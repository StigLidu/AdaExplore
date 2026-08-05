import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

# Autotune configurations exploring BLOCK (spatial reduction chunk) and C_BLOCK (channels per program)
# Tuned for Ampere A6000: larger C_BLOCK (32/64) and BLOCK values that are multiples of warp size (128/256).
# Heavier configs use more warps/stages to better utilize SM resources.
AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK": 256, "C_BLOCK": 32},  num_warps=4, num_stages=2),
    triton.Config({"BLOCK": 256, "C_BLOCK": 64},  num_warps=8, num_stages=3),
    triton.Config({"BLOCK": 512, "C_BLOCK": 32},  num_warps=8, num_stages=3),
    triton.Config({"BLOCK": 1024, "C_BLOCK": 32}, num_warps=8, num_stages=3),
]

@triton.autotune(
    configs=AUTOTUNE_CONFIGS,
    key=["N", "C", "HW"],
)
@triton.jit
def _global_avgpool_kernel(
    x_ptr,            # pointer to input (flattened N*C*H*W)
    out_ptr,          # pointer to output (flattened N*C)
    N,                # batch size
    C,                # channels
    HW,               # H*W
    stride_n,         # stride to advance one sample (in elements)
    stride_c,         # stride to advance one channel (in elements)
    BLOCK: tl.constexpr,    # spatial chunk size (constexpr)
    C_BLOCK: tl.constexpr,  # number of channels processed per program (constexpr)
):
    """
    Each program handles a tile: one batch element (n) and C_BLOCK consecutive channels.
    It reduces over the HW spatial elements in chunks of size BLOCK, accumulating per-channel sums.
    """

    n = tl.program_id(0)                      # batch index
    c_block_idx = tl.program_id(1)            # index of channel block

    # channel offsets handled by this program: (c_block_idx * C_BLOCK) + [0..C_BLOCK-1]
    c_offsets = c_block_idx * C_BLOCK + tl.arange(0, C_BLOCK)
    mask_c = c_offsets < C  # mask for channels that actually exist

    # Base pointer for each channel in the block at spatial offset 0:
    # (n * stride_n) + (c_offsets * stride_c)
    base_ptrs = x_ptr + n * stride_n + c_offsets * stride_c  # shape: (C_BLOCK,)

    # accumulator per channel
    acc = tl.zeros((C_BLOCK,), dtype=tl.float32)

    # Loop over spatial positions in chunks of BLOCK
    # For each chunk, we build a pointer matrix of shape (C_BLOCK, BLOCK) and load values
    for off_start in range(0, HW, BLOCK):
        offs = off_start + tl.arange(0, BLOCK)  # shape: (BLOCK,)
        mask_hw = offs < HW                      # shape: (BLOCK,)

        # Create (C_BLOCK, BLOCK) pointer matrix: base_ptrs[:, None] + offs[None, :]
        ptrs = base_ptrs[:, None] + offs[None, :]

        # Combined mask: channel mask broadcasted over BLOCK and spatial mask broadcasted over C_BLOCK
        mask = mask_c[:, None] & mask_hw[None, :]

        vals = tl.load(ptrs, mask=mask, other=0.0)  # shape (C_BLOCK, BLOCK)
        acc += tl.sum(vals, axis=1)

    # Compute average
    avg = acc / HW  # shape (C_BLOCK,)

    # Store results into out_ptr at positions (n * C + c_offsets)
    out_ptrs = out_ptr + n * C + c_offsets
    tl.store(out_ptrs, avg, mask=mask_c)


def triton_global_avg_pool(x: torch.Tensor):
    """
    x: Tensor of shape (N, C, H, W), contiguous, cuda float32
    returns: Tensor of shape (N, C) with global average pooled values
    """
    assert x.is_cuda and x.dtype == torch.float32, "Triton global avg pool requires CUDA float32 tensor."
    x = x.contiguous()
    N, C, H, W = x.shape
    HW = H * W

    out = torch.empty((N, C), device=x.device, dtype=x.dtype)

    stride_n = x.stride(0)
    stride_c = x.stride(1)

    # Grid: one program per (n, channel_block)
    grid = lambda meta: (N, (C + meta["C_BLOCK"] - 1) // meta["C_BLOCK"])

    _global_avgpool_kernel[grid](
        x,
        out,
        N,
        C,
        HW,
        stride_n,
        stride_c,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        """
        EfficientNetB1-like model where the global average pooling has been replaced
        with an optimized Triton kernel that processes multiple channels per program
        and vectorizes the spatial reduction for better GPU utilization.
        """
        super(ModelNew, self).__init__()

        # Initial convolutional layer
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)

        # MBConv blocks (kept as PyTorch modules for correctness and to avoid
        # reimplementing convolutional primitives in Triton here)
        self.mbconv1 = self._make_mbconv_block(32, 16, 1, 1)
        self.mbconv2 = self._make_mbconv_block(16, 24, 2, 6)
        self.mbconv3 = self._make_mbconv_block(24, 40, 2, 6)
        self.mbconv4 = self._make_mbconv_block(40, 80, 2, 6)
        self.mbconv5 = self._make_mbconv_block(80, 112, 1, 6)
        self.mbconv6 = self._make_mbconv_block(112, 192, 2, 6)
        self.mbconv7 = self._make_mbconv_block(192, 320, 1, 6)

        # Final convolutional layer
        self.conv2 = nn.Conv2d(320, 1280, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(1280)

        # Fully connected layer: we will compute global average pooling first (with Triton) and then apply fc.
        # This reduces the amount of work compared to applying a per-class 1x1 conv before pooling.
        self.fc = nn.Linear(1280, num_classes)

    def _make_mbconv_block(self, in_channels, out_channels, stride, expand_ratio):
        hidden_dim = round(in_channels * expand_ratio)
        return nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        """
        Forward pass. Uses Triton-optimized global average pooling for the final pooling step.
        The rest of the network uses standard PyTorch layers.
        """
        x = F.relu(self.bn1(self.conv1(x)))

        x = self.mbconv1(x)
        x = self.mbconv2(x)
        x = self.mbconv3(x)
        x = self.mbconv4(x)
        x = self.mbconv5(x)
        x = self.mbconv6(x)
        x = self.mbconv7(x)

        x = F.relu(self.bn2(self.conv2(x)))

        # Compute global average pooling first (Triton-optimized), then apply the final linear layer.
        # This reduces computation from O(HW * C * num_classes) to O(C * num_classes) per sample.
        x = x.contiguous()
        pooled = triton_global_avg_pool(x)  # shape (N, C=1280)

        # Apply the linear layer using a matmul for efficiency.
        out = pooled.matmul(self.fc.weight.t())
        if self.fc.bias is not None:
            out = out + self.fc.bias.unsqueeze(0)
        return out