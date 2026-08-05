import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

# Autotune configs expanded to include a spatial tile size (BLOCK_HW).
# These combinations favor coalesced spatial loads on Ampere (A6000).
# Added larger spatial tiles and a wider channel tile to bias the search
# toward higher memory/SIMT utilization on A6000 (see reviser guidance).
POOL_AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK_C": 32,  "BLOCK_HW": 8},  num_warps=4,  num_stages=2),
    triton.Config({"BLOCK_C": 32,  "BLOCK_HW": 16}, num_warps=4,  num_stages=2),
    triton.Config({"BLOCK_C": 32,  "BLOCK_HW": 32}, num_warps=8,  num_stages=3),
    triton.Config({"BLOCK_C": 64,  "BLOCK_HW": 8},  num_warps=4,  num_stages=2),
    triton.Config({"BLOCK_C": 64,  "BLOCK_HW": 16}, num_warps=8,  num_stages=3),
    triton.Config({"BLOCK_C": 64,  "BLOCK_HW": 32}, num_warps=8,  num_stages=3),
    triton.Config({"BLOCK_C": 128, "BLOCK_HW": 8},  num_warps=8,  num_stages=3),
    triton.Config({"BLOCK_C": 128, "BLOCK_HW": 16}, num_warps=8,  num_stages=4),
    triton.Config({"BLOCK_C": 128, "BLOCK_HW": 32}, num_warps=8,  num_stages=4),
    # Wider spatial tiles to enable larger contiguous loads (suits 224*224 HW)
    triton.Config({"BLOCK_C": 128, "BLOCK_HW": 64},  num_warps=8,  num_stages=3),
    # Very large spatial tile: favor more warps and stages to utilize compute/SIMT
    triton.Config({"BLOCK_C": 128, "BLOCK_HW": 128}, num_warps=16, num_stages=4),
]

@triton.autotune(configs=POOL_AUTOTUNE_CONFIGS, key=["N", "C", "HW"])
@triton.jit
def _global_avgpool_kernel(
    x_ptr,          # pointer to input flattened tensor (N*C*H*W)
    out_ptr,        # pointer to output flattened tensor (N*C)
    N,              # number of batches
    C,              # number of channels
    H,              # height
    W,              # width
    HW,             # H * W
    inv_HW,         # inverse of H*W (passed from host)
    BLOCK_C: tl.constexpr,  # block size in channels
    BLOCK_HW: tl.constexpr, # block size in spatial dimension (H*W)
):
    """
    Triton kernel to compute global average pooling over spatial dims.
    Each program handles a (n, cb) block where:
      - n is batch index (program_id(0))
      - cb is channel-block index (program_id(1))
    We load tiles of shape (BLOCK_C, BLOCK_HW) for coalesced spatial access,
    then reduce along the spatial axis.
    """

    n = tl.program_id(0)
    cb = tl.program_id(1)

    c_start = cb * BLOCK_C
    offs_c = c_start + tl.arange(0, BLOCK_C)
    mask_c = offs_c < C  # (BLOCK_C,)

    # Base pointer for each channel in this block at spatial position 0
    # Flattened layout: ((n * C + c) * HW) + p
    base = (n * C + offs_c) * HW  # shape: (BLOCK_C,)

    # accumulator for each channel in the block
    acc = tl.zeros((BLOCK_C,), dtype=tl.float32)

    # Iterate over spatial dimension in tiles of BLOCK_HW for coalesced loads.
    # BLOCK_HW is a constexpr so this loop has a known step at compile time.
    for p0 in range(0, HW, BLOCK_HW):
        # spatial offsets for this tile: shape (BLOCK_HW,)
        offs_p = p0 + tl.arange(0, BLOCK_HW)
        mask_p = offs_p < HW  # (BLOCK_HW,)

        # compute full (BLOCK_C, BLOCK_HW) offsets and mask
        offs = base[:, None] + offs_p[None, :]  # (BLOCK_C, BLOCK_HW)
        mask = mask_c[:, None] & mask_p[None, :]  # (BLOCK_C, BLOCK_HW)

        # Load a tile and reduce across spatial axis (axis=1)
        vals = tl.load(x_ptr + offs, mask=mask, other=0.0)  # (BLOCK_C, BLOCK_HW)
        # Sum over spatial positions in the tile -> (BLOCK_C,)
        acc += tl.sum(vals, axis=1)

    # compute mean using inv_HW passed from host to avoid casting inside Triton kernel
    acc = acc * inv_HW

    # store into output flattened at index n*C + c
    out_offsets = n * C + offs_c
    tl.store(out_ptr + out_offsets, acc, mask=mask_c)


def triton_global_avgpool(x: torch.Tensor):
    """
    x: (N, C, H, W), contiguous CUDA float32 tensor
    returns: (N, C) tensor with global average pooled values
    """
    assert x.is_cuda and x.dtype == torch.float32, "Input must be CUDA float32 tensor"
    N, C, H, W = x.shape
    HW = H * W

    # Prepare output
    out = torch.empty((N, C), device=x.device, dtype=x.dtype)

    # Ensure contiguous and flattened input for pointer arithmetic in Triton kernel
    x_flat = x.contiguous().view(-1)
    out_flat = out.view(-1)

    # grid: (N, num_channel_blocks)
    def grid(meta):
        return (N, (C + meta["BLOCK_C"] - 1) // meta["BLOCK_C"])

    # Precompute inverse HW on host to avoid calling float() inside Triton kernel
    inv_HW = 1.0 / float(HW)

    # Launch kernel
    _global_avgpool_kernel[grid](x_flat, out_flat, N, C, H, W, HW, inv_HW)

    return out

# ---- Model definitions ----

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        """
        :param in_channels: Number of input channels
        :param out_channels: Number of output channels
        :param stride: Stride for the first convolutional layer
        :param downsample: Downsample layer for the shortcut connection
        """
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        """
        :param x: Input tensor, shape (batch_size, in_channels, height, width)
        :return: Output tensor, shape (batch_size, out_channels, height, width)
        """
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        """
        Optimized ResNet-like model where global average pooling is implemented
        via a custom Triton kernel for improved performance on GPU.
        """
        super(ModelNew, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(BasicBlock, 64, 2, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2)

        # Keep the fully-connected layer in PyTorch (leverages cuBLAS)
        self.fc = nn.Linear(512 * BasicBlock.expansion, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        """
        :param x: Input tensor, shape (batch_size, 3, height, width)
        :return: Output tensor, shape (batch_size, num_classes)
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Use Triton-based fused global average pooling
        # x: (N, C, H, W) -> pooled: (N, C)
        pooled = triton_global_avgpool(x)

        x = pooled  # (N, C)
        x = self.fc(x)

        return x

# For compatibility with original script structure
batch_size = 2
num_classes = 1000
input_shape = (batch_size, 3, 224, 224)

def get_inputs():
    # Ensure inputs are on CUDA since Triton kernels expect CUDA tensors
    return [torch.rand(input_shape).cuda()]

def get_init_inputs():
    return [num_classes]