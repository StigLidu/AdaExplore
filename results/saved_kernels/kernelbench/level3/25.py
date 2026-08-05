import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

# Autotune configs tuned for Ampere-class GPUs (A6000)
AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK_SIZE": 256},  num_warps=4, num_stages=2),
    triton.Config({"BLOCK_SIZE": 512},  num_warps=4, num_stages=2),
    triton.Config({"BLOCK_SIZE": 1024}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_SIZE": 2048}, num_warps=8, num_stages=3),
]

@triton.autotune(
    configs=AUTOTUNE_CONFIGS,
    key=['N', 'C', 'H', 'W', 'G'],
)
@triton.jit
def _depthwise_conv3x3_shuffled_kernel(
    x_ptr,         # input pointer (flattened N*C*H*W)
    w_ptr,         # weight pointer (flattened C * 1 * KH * KW)
    out_ptr,       # output pointer (flattened N*C*H*W) -- will be in shuffled channel order
    total_elems,   # total number of output elements
    N,             # batch size
    C,             # channels (mid_channels)
    H,             # height
    W,             # width
    KH,            # kernel height (3)
    KW,            # kernel width (3)
    pad_h,         # padding height (1)
    pad_w,         # padding width (1)
    G,             # groups (for shuffle)
    K,             # channels per group = C // G
    BLOCK_SIZE: tl.constexpr
):
    """
    Each program processes BLOCK_SIZE linear output elements corresponding to the original
    channel ordering (cs = source channel). For each cs, compute depthwise conv output then
    write it to the shuffled destination channel index so a separate channel-shuffle pass is eliminated.
    """
    offs = tl.arange(0, BLOCK_SIZE) + tl.program_id(0) * BLOCK_SIZE
    mask = offs < total_elems

    CHW = C * H * W
    HW = H * W

    # recover n, cs, h, w from linear offset
    n = offs // CHW
    r = offs - n * CHW

    cs = r // HW
    r2 = r - cs * HW

    h = r2 // W
    w = r2 - h * W

    # accumulator
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    # compute depthwise convolution (3x3)
    # iterate kernel rows and cols
    for kh in range(3):
        in_h = h + kh - pad_h
        valid_h = (in_h >= 0) & (in_h < H)
        for kw in range(3):
            in_w = w + kw - pad_w
            valid_w = (in_w >= 0) & (in_w < W)

            valid = mask & valid_h & valid_w

            # input index: ((n * C + cs) * H + in_h) * W + in_w
            in_idx = ((n * C + cs) * H + in_h) * W + in_w
            val = tl.load(x_ptr + in_idx, mask=valid, other=0.0)

            # weight index for depthwise: cs*(KH*KW) + kh*KW + kw
            w_idx = cs * (KH * KW) + kh * KW + kw
            wval = tl.load(w_ptr + w_idx, mask=mask, other=0.0)

            acc = acc + val * wval

    # compute shuffled destination channel index for cs:
    # cs = g * K + k  => g = cs // K, k = cs % K
    g = cs // K
    k = cs - g * K
    c_dest = k * G + g

    # destination linear offset: ((n * C + c_dest) * H + h) * W + w
    dest_offs = ((n * C + c_dest) * H + h) * W + w

    tl.store(out_ptr + dest_offs, acc, mask=mask)


def triton_depthwise_conv3x3_shuffled(x: torch.Tensor, weight: torch.Tensor, groups_for_shuffle: int, padding: int = 1):
    """
    Triton-backed depthwise 3x3 convolution that writes its output in the channel-shuffled order.
    This eliminates a separate channel-shuffle pass and reduces memory traffic.
    - x: (N, C, H, W), contiguous CUDA float32
    - weight: (C, 1, 3, 3), contiguous CUDA float32
    - groups_for_shuffle: the groups parameter used by ChannelShuffle
    """
    if not x.is_cuda:
        # CPU fallback: compute conv then apply CPU shuffle
        out = F.conv2d(x, weight, bias=None, stride=1, padding=padding, groups=x.shape[1])
        batch_size, channels, height, width = out.size()
        groups = groups_for_shuffle
        channels_per_group = channels // groups
        out = out.view(batch_size, groups, channels_per_group, height, width)
        out = out.transpose(1, 2).contiguous()
        return out.view(batch_size, channels, height, width)

    assert x.is_contiguous(), "Input must be contiguous"
    assert weight.is_contiguous(), "Weight must be contiguous"
    assert x.dtype == torch.float32 and weight.dtype == torch.float32

    N, C, H, W = x.shape
    KH, KW = 3, 3
    pad_h = padding
    pad_w = padding
    G = groups_for_shuffle
    assert C % G == 0, "channels must be divisible by groups"
    K = C // G

    total_elems = N * C * H * W
    out = torch.empty_like(x)

    # grid depends on the selected BLOCK_SIZE from autotune
    grid = lambda meta: ((total_elems + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    _depthwise_conv3x3_shuffled_kernel[grid](
        x, weight, out,
        total_elems,
        N, C, H, W,
        KH, KW,
        pad_h, pad_w,
        G, K,
    )

    return out


class ModelNew(nn.Module):
    """
    Optimized ShuffleNet unit:
      - Uses PyTorch/cuDNN optimized grouped 1x1 conv for conv1 and conv3 (keeps highly-optimized GEMM path).
      - Replaces depthwise 3x3 conv + channel shuffle with a single Triton kernel that computes the
        depthwise convolution and writes its outputs already shuffled, eliminating a separate channel shuffle pass.
      - Keeps BatchNorm and ReLU placements equivalent to the original for correctness.
    """
    def __init__(self, in_channels, out_channels, groups=3):
        super(ModelNew, self).__init__()

        assert out_channels % 4 == 0
        mid_channels = out_channels // 4

        # First 1x1 group convolution
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=1, padding=0, groups=groups, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)

        # Depthwise 3x3 convolution (we will use its weight with Triton, and Triton will write shuffled output)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=1, padding=1, groups=mid_channels, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)

        # Second 1x1 group convolution
        self.conv3 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, stride=1, padding=0, groups=groups, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        # store groups for shuffle operation
        self.groups = groups

        # Shortcut connection if input and output channels are the same
        if in_channels == out_channels:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        # conv1 + bn + relu (use PyTorch/cuDNN grouped 1x1 conv)
        out = F.relu(self.bn1(self.conv1(x)))

        # depthwise conv via Triton that writes its output in shuffled channel order
        w = self.conv2.weight  # shape (C, 1, 3, 3)
        if out.is_cuda:
            if not w.is_cuda:
                w = w.to(out.device)
            out = out.contiguous()
            w = w.contiguous()
            out = triton_depthwise_conv3x3_shuffled(out, w, self.groups, padding=1)
        else:
            # CPU fallback
            out = triton_depthwise_conv3x3_shuffled(out, w, self.groups, padding=1)

        out = self.bn2(out)

        # conv3 + bn + relu
        out = F.relu(self.bn3(self.conv3(out)))

        out = out + self.shortcut(x)
        return out


# Helper functions consistent with the original interface
batch_size = 10
input_channels = 240
out_channels = 480
groups = 3
height = 224
width = 224
num_classes = 1000

def get_inputs():
    # Return CUDA tensor for GPU workloads
    return [torch.rand(batch_size, input_channels, height, width).cuda()]

def get_init_inputs():
    return [input_channels, out_channels, groups]