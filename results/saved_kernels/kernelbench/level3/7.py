import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

# ---------------------------
# Triton-accelerated average pooling that writes fp16 outputs directly.
# This reduces memory traffic by producing (N, C) in fp16 which is then consumed
# by a fp16 GEMM (tensor cores).
# ---------------------------
AUTOTUNE_CONFIGS_AVG_FP16 = [
    triton.Config({"BLOCK_C": 128, "BLOCK_HW": 49},  num_warps=4, num_stages=2),
    triton.Config({"BLOCK_C": 256, "BLOCK_HW": 49},  num_warps=8, num_stages=2),
    triton.Config({"BLOCK_C": 512, "BLOCK_HW": 49},  num_warps=8, num_stages=3),
    triton.Config({"BLOCK_C": 256, "BLOCK_HW": 64},  num_warps=8, num_stages=3),
]


@triton.autotune(configs=AUTOTUNE_CONFIGS_AVG_FP16, key=['N', 'C', 'HW'])
@triton.jit
def _global_avgpool_fp16_kernel(
    x_ptr,          # pointer to input x flattened as (N*C*HW)
    out_ptr,        # pointer to output (N*C) fp16
    N,              # batch size
    C,              # channels
    HW,             # spatial size H*W
    BLOCK_C: tl.constexpr,   # number of channels processed per program
    BLOCK_HW: tl.constexpr,  # (unused as a vector-width here) kept for autotune key compatibility
):
    """
    Each program handles BLOCK_C channels for one batch element n.
    It computes the mean over HW for each channel (in fp32 accumulation), then casts to fp16 for storage.
    Grid dimensions: (N, ceildiv(C, BLOCK_C))
    Note: the inner spatial dimension is iterated scalarly to avoid 2D loads that caused compilation issues.
    """
    n = tl.program_id(0)
    c_block = tl.program_id(1)

    c_start = c_block * BLOCK_C
    offs_c = tl.arange(0, BLOCK_C)
    c_idx = c_start + offs_c                 # (BLOCK_C,)
    mask_c = c_idx < C

    # base pointer for each channel in this block: (BLOCK_C,)
    base = (n * C + c_idx) * HW

    # accumulate in fp32 per channel
    sum_c = tl.zeros((BLOCK_C,), dtype=tl.float32)

    # Iterate over spatial elements one-by-one to avoid 2D loads / tl.arange over BLOCK_HW.
    hw_off = 0
    while hw_off < HW:
        # ptrs is (BLOCK_C,) : pointer for each channel at this spatial index
        ptrs = x_ptr + base + hw_off
        vals = tl.load(ptrs, mask=mask_c, other=0.0)  # fp32 loads per channel
        sum_c += vals
        hw_off += 1

    inv_HW = 1.0 / HW
    avg_c = sum_c * inv_HW  # fp32 (BLOCK_C,)

    # cast to fp16 and store
    avg_fp16 = tl.cast(avg_c, tl.float16)
    out_ptrs = out_ptr + n * C + c_idx
    tl.store(out_ptrs, avg_fp16, mask=mask_c)


def triton_global_avg_pool_fp16(x: torch.Tensor):
    """
    Compute global average pool (N, C, H, W) -> (N, C) in fp16 using Triton.
    The kernel accumulates in fp32 for numerical correctness, then writes fp16.
    Returns a tensor of dtype torch.float16 shaped (N, C).
    """
    assert x.is_cuda, "Input must be CUDA"
    assert x.dtype == torch.float32, "Input must be float32"
    x = x.contiguous()
    N, C, H, W = x.shape
    HW = H * W

    out = torch.empty((N, C), device=x.device, dtype=torch.float16)

    # Flatten input to 1D pointer view for pointer arithmetic in kernel
    x_flat = x.view(-1)

    grid = lambda meta: (N, (C + meta["BLOCK_C"] - 1) // meta["BLOCK_C"])
    _global_avgpool_fp16_kernel[grid](x_flat, out, N, C, HW)
    return out


# ---------------------------
# Weight fp16 cache
# ---------------------------
_WEIGHT_FP16_CACHE = {}


def _get_cached_fp16_weights(weight: torch.Tensor):
    """
    Cache both (K, C) fp16 and its transpose (C, K) fp16 for reuse.
    """
    key = (weight.data_ptr(), weight.shape, weight.device)
    cached = _WEIGHT_FP16_CACHE.get(key)
    if cached is None:
        weight_fp16_KC = weight.half().contiguous()         # (K, C)
        weight_fp16_CK = weight_fp16_KC.t().contiguous()    # (C, K)
        _WEIGHT_FP16_CACHE[key] = (weight_fp16_KC, weight_fp16_CK)
        return weight_fp16_KC, weight_fp16_CK
    return cached


# ---------------------------
# Fused avgpool (fp16) + fp16 GEMM path
# ---------------------------
def fused_avgpool_linear(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor = None):
    """
    Efficient fused path optimized for the common final shape of this model:
      - Uses a Triton kernel to compute (N, C) averages directly in fp16 (accumulate in fp32).
      - Uses cached fp16 weights and a single fp16 GEMM (cuBLAS / tensor cores) to compute logits.
    This reduces memory traffic by keeping the intermediate (N, C) in fp16 and leveraging tensor cores.
    """
    assert x.is_cuda and weight.is_cuda, "tensors must be CUDA"
    assert x.dtype == torch.float32 and weight.dtype == torch.float32
    if bias is not None:
        assert bias.is_cuda and bias.dtype == torch.float32

    # ensure contiguous
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is None:
        bias = torch.zeros((weight.shape[0],), device=weight.device, dtype=torch.float32)
    else:
        bias = bias.contiguous()

    N, C, H, W = x.shape
    K, C_w = weight.shape
    assert C == C_w, "weight in_features must match x channels"

    HW = H * W

    # For the shapes in this model (HW = 49, K = 1000), the fp16 two-step path is highly favorable:
    # - Compute avg in fp16 via Triton kernel (accumulating in fp32)
    # - Use fp16 matmul with cached fp16 weights to utilize tensor cores
    # Get cached fp16 weights (K, C) and (C, K)
    weight_fp16_KC, weight_fp16_CK = _get_cached_fp16_weights(weight)

    # Stage A: compute avg in fp16 using Triton kernel
    x_avg_h = triton_global_avg_pool_fp16(x)  # (N, C) fp16

    # Stage B: fp16 GEMM via torch.matmul under autocast to ensure tensor-core path
    # We use weight_fp16_CK (C, K) so matmul is (N, C) @ (C, K) -> (N, K)
    with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
        out_h = torch.matmul(x_avg_h, weight_fp16_CK)  # (N, K) fp16

    out = out_h.float()  # cast back to fp32 for consistency with original model
    if bias is not None:
        out = out + bias.unsqueeze(0)
    return out


# ---------------------------
# Inception Module (unchanged semantics)
# ---------------------------
class InceptionModule(nn.Module):
    def __init__(self, in_channels, out_1x1, reduce_3x3, out_3x3, reduce_5x5, out_5x5, pool_proj):
        """
        :param in_channels: Number of input channels
        :param out_1x1: Number of output channels for the 1x1 convolution
        :param reduce_3x3: Number of output channels for the 1x1 reduction before 3x3 convolution
        :param out_3x3: Number of output channels for the 3x3 convolution
        :param reduce_5x5: Number of output channels for the 1x1 reduction before 5x5 convolution
        :param out_5x5: Number of output channels for the 5x5 convolution
        :param pool_proj: Number of output channels for the pooling projection
        """
        super(InceptionModule, self).__init__()
        
        # 1x1 convolution branch
        self.branch1x1 = nn.Conv2d(in_channels, out_1x1, kernel_size=1)
        
        # 3x3 convolution branch
        self.branch3x3 = nn.Sequential(
            nn.Conv2d(in_channels, reduce_3x3, kernel_size=1),
            nn.Conv2d(reduce_3x3, out_3x3, kernel_size=3, padding=1)
        )
        
        # 5x5 convolution branch
        self.branch5x5 = nn.Sequential(
            nn.Conv2d(in_channels, reduce_5x5, kernel_size=1),
            nn.Conv2d(reduce_5x5, out_5x5, kernel_size=5, padding=2)
        )
        
        # Max pooling branch
        self.branch_pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, pool_proj, kernel_size=1)
        )
    
    def forward(self, x):
        """
        :param x: Input tensor, shape (batch_size, in_channels, height, width)
        :return: Output tensor, shape (batch_size, out_channels, height, width)
        """
        branch1x1 = self.branch1x1(x)
        branch3x3 = self.branch3x3(x)
        branch5x5 = self.branch5x5(x)
        branch_pool = self.branch_pool(x)
        
        outputs = [branch1x1, branch3x3, branch5x5, branch_pool]
        return torch.cat(outputs, 1)


# ---------------------------
# ModelNew: Use Triton-accelerated avgpool->fp16 GEMM path for final classifier.
# ---------------------------
class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        """
        Optimized Model:
         - Keeps original convolutional/Inception blocks.
         - Replaces AdaptiveAvgPool2d+flatten+fc with a Triton-accelerated path that:
             * computes per-(N,C) averages in fp16 (with fp32 accumulation) via Triton,
             * performs a single fp16 GEMM (tensor-core) using cached fp16 weights,
             * casts outputs back to fp32 and adds bias.
        """
        super(ModelNew, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = nn.MaxPool2d(3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=1)
        self.conv3 = nn.Conv2d(64, 192, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool2d(3, stride=2, padding=1)

        self.inception3a = InceptionModule(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = InceptionModule(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(3, stride=2, padding=1)

        self.inception4a = InceptionModule(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = InceptionModule(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = InceptionModule(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = InceptionModule(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = InceptionModule(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(3, stride=2, padding=1)

        self.inception5a = InceptionModule(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = InceptionModule(832, 384, 192, 384, 48, 128, 128)

        # Keep dropout (p=0.0 per spec) and final fc for parameterization
        self.dropout = nn.Dropout(0.0)
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        """
        :param x: Input tensor, shape (batch_size, 3, height, width)
        :return: Output tensor, shape (batch_size, num_classes)
        """
        x = self.maxpool1(F.relu(self.conv1(x)))
        x = F.relu(self.conv2(x))
        x = self.maxpool2(F.relu(self.conv3(x)))

        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)

        x = self.inception4a(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        x = self.inception4e(x)
        x = self.maxpool4(x)

        x = self.inception5a(x)
        x = self.inception5b(x)

        # Use the optimized Triton path for avgpool + final linear
        logits = fused_avgpool_linear(x, self.fc.weight, self.fc.bias)
        return logits