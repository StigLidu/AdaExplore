import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

# Tuned autotune configs targeting NVIDIA A6000 (Ampere)
AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK": 256},  num_warps=4, num_stages=2),
    triton.Config({"BLOCK": 512},  num_warps=8, num_stages=2),
    triton.Config({"BLOCK": 1024}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK": 2048}, num_warps=8, num_stages=3),
]


@triton.autotune(configs=AUTOTUNE_CONFIGS, key=["B", "C", "Hout", "Wout"])
@triton.jit
def bn_relu_avgpool2x2_kernel(
    x_ptr,          # pointer to input tensor (B, C, H, W), flattened
    out_ptr,        # pointer to output tensor (B, C, Hout, Wout), flattened
    gamma_ptr,      # bn weight (C,)
    beta_ptr,       # bn bias (C,)
    mean_ptr,       # running mean or batch mean (C,)
    var_ptr,        # running var or batch var (C,)
    B, C, H, W,     # input dims
    Hout, Wout,     # output spatial dims (H//2, W//2)
    eps,
    BLOCK: tl.constexpr   # block size over spatial positions
):
    """
    Each program handles one (b, c) plane and a contiguous block of output spatial positions.
    For each output position we load the 2x2 input region, apply BN affine, ReLU, then average.
    """
    bc = tl.program_id(0)  # index over batch*channel = b * C + c
    block_id = tl.program_id(1)
    HoutWout = Hout * Wout

    # compute offsets for spatial positions handled by this program
    block_start = block_id * BLOCK
    offs = block_start + tl.arange(0, BLOCK)
    mask = offs < HoutWout

    # decode spatial indices i (row) and j (col) for each offset (vectorized)
    i = offs // Wout
    j = offs % Wout

    # compute input h,w coordinates for the 2x2 block
    h0 = 2 * i
    w0 = 2 * j
    h1 = h0 + 1
    w1 = w0 + 1

    # base addresses
    base_in = bc * (H * W)
    base_out = bc * (HoutWout)

    # compute element addresses for the four pixels of each pooled output
    addr00 = base_in + h0 * W + w0
    addr10 = base_in + h1 * W + w0
    addr01 = base_in + h0 * W + w1
    addr11 = base_in + h1 * W + w1

    # load input values (masked)
    x00 = tl.load(x_ptr + addr00, mask=mask, other=0.0)
    x10 = tl.load(x_ptr + addr10, mask=mask, other=0.0)
    x01 = tl.load(x_ptr + addr01, mask=mask, other=0.0)
    x11 = tl.load(x_ptr + addr11, mask=mask, other=0.0)

    # Load BN params for this channel (scalar)
    c_idx = bc % C
    gamma = tl.load(gamma_ptr + c_idx)
    beta = tl.load(beta_ptr + c_idx)
    mean = tl.load(mean_ptr + c_idx)
    var = tl.load(var_ptr + c_idx)
    invstd = 1.0 / tl.sqrt(var + eps)

    # Apply BN (affine) then ReLU on each of the four values, vectorized
    # y = gamma * (x - mean) * invstd + beta
    y00 = gamma * (x00 - mean) * invstd + beta
    y10 = gamma * (x10 - mean) * invstd + beta
    y01 = gamma * (x01 - mean) * invstd + beta
    y11 = gamma * (x11 - mean) * invstd + beta

    # ReLU (vectorized)
    y00 = tl.maximum(y00, 0.0)
    y10 = tl.maximum(y10, 0.0)
    y01 = tl.maximum(y01, 0.0)
    y11 = tl.maximum(y11, 0.0)

    # Average pooling (2x2)
    pooled = 0.25 * (y00 + y10 + y01 + y11)

    # store results into output
    out_addr = base_out + offs
    tl.store(out_ptr + out_addr, pooled, mask=mask)


def triton_bn_relu_avgpool2x2(x: torch.Tensor, bn: nn.BatchNorm2d, eps: float = 1e-5):
    """
    x: (B, C, H, W) float32 CUDA tensor
    bn: nn.BatchNorm2d instance (we will read its parameters/buffers)
    Returns: (B, C, H//2, W//2) float32 CUDA tensor
    """
    assert x.is_cuda, "Input must be on CUDA"
    B, C, H, W = x.shape
    assert H % 2 == 0 and W % 2 == 0, "H and W must be divisible by 2 for 2x2 pooling"

    Hout = H // 2
    Wout = W // 2
    HoutWout = Hout * Wout

    # Ensure contiguous for efficient pointer arithmetic
    x = x.contiguous()

    # Prepare output
    out = torch.empty((B, C, Hout, Wout), device=x.device, dtype=x.dtype)

    # Flattened pointers (Triton accepts tensors; we pass flattened views)
    x_ptr = x.view(-1)
    out_ptr = out.view(-1)

    # BN params (ensure contiguous on device)
    gamma = bn.weight.contiguous()
    beta = bn.bias.contiguous()

    # If BatchNorm is in training mode, compute batch (per-channel) mean and var
    # across batch and spatial dims, to match PyTorch's BatchNorm behavior.
    if bn.training:
        # mean/var over dims (0,2,3): (B, C, H, W) -> (C,)
        # Use unbiased=False to match running var update semantics
        mean = x.mean(dim=(0, 2, 3)).contiguous()
        var = x.var(dim=(0, 2, 3), unbiased=False).contiguous()
    else:
        mean = bn.running_mean.contiguous()
        var = bn.running_var.contiguous()

    # Move BN params to CUDA if necessary
    if not gamma.is_cuda:
        gamma = gamma.cuda()
    if not beta.is_cuda:
        beta = beta.cuda()
    if not mean.is_cuda:
        mean = mean.cuda()
    if not var.is_cuda:
        var = var.cuda()

    # Grid: (B*C, ceil(HoutWout / BLOCK))
    def grid(meta):
        BLOCK = meta["BLOCK"]
        return (B * C, (HoutWout + BLOCK - 1) // BLOCK)

    bn_relu_avgpool2x2_kernel[grid](
        x_ptr, out_ptr,
        gamma, beta, mean, var,
        B, C, H, W, Hout, Wout,
        float(eps)
    )

    return out


class ModelNew(nn.Module):
    def __init__(self, num_input_features: int, num_output_features: int):
        super(ModelNew, self).__init__()
        # Keep original module for state_dict compatibility
        self.transition = nn.Sequential(
            nn.BatchNorm2d(num_input_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_input_features, num_output_features, kernel_size=1, bias=False),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
        # Expose bn and conv for the fused implementation
        self.bn = self.transition[0]
        self.conv = self.transition[2]

    def forward(self, x):
        """
        Optimized forward:
          1) Fused BN + ReLU + 2x2 AvgPool implemented in Triton (reduces memory traffic).
          2) Apply 1x1 convolution on the downsampled feature map using PyTorch's optimized conv2d.
        This ordering reduces the spatial domain before the 1x1 convolution, saving compute and memory.
        """
        if x.device.type == "cuda":
            x = x.contiguous()
            pooled = triton_bn_relu_avgpool2x2(x, self.bn)
            # Use PyTorch conv2d for 1x1 conv (highly optimized)
            out = F.conv2d(pooled, self.conv.weight, bias=None, stride=1, padding=0)
            return out
        else:
            # CPU fallback to original sequential module
            return self.transition(x)


# Keep the same helper functions signature as the original
batch_size = 128
num_input_features = 32
num_output_features = 64
height, width = 256, 256


def get_inputs():
    return [torch.rand(batch_size, num_input_features, height, width).cuda()]


def get_init_inputs():
    return [num_input_features, num_output_features]