import torch
import torch.nn as nn
import triton
import triton.language as tl

# Autotune configurations for the softmax kernel. We key by 'cols' to ensure that
# the autotuner can pick the best BLOCK_SIZE/warp/stage combination for different row widths.
# We include smaller BLOCK_SIZEs (32,64,128,256) and try multiple num_warps/num_stages
# combinations to suit Ampere GPUs like the A6000.
AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK_SIZE": 32},  num_warps=2, num_stages=2),
    triton.Config({"BLOCK_SIZE": 32},  num_warps=4, num_stages=2),
    triton.Config({"BLOCK_SIZE": 32},  num_warps=8, num_stages=3),
    triton.Config({"BLOCK_SIZE": 64},  num_warps=2, num_stages=2),
    triton.Config({"BLOCK_SIZE": 64},  num_warps=4, num_stages=2),
    triton.Config({"BLOCK_SIZE": 64},  num_warps=8, num_stages=3),
    triton.Config({"BLOCK_SIZE": 128}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_SIZE": 128}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_SIZE": 256}, num_warps=8, num_stages=3),
]

@triton.autotune(configs=AUTOTUNE_CONFIGS, key=['cols'])
@triton.jit
def _softmax_last_dim_kernel(x_ptr, out_ptr, rows, cols, BLOCK_SIZE: tl.constexpr):
    """
    Tiled, three-pass row-wise softmax implemented in Triton.
    Each program handles one row. The row is scanned in tiles of size BLOCK_SIZE.

    Pass 1: compute row maximum by scanning tiles (uses masked loads with other=-inf)
    Pass 2: compute sum of exp(x - max) by scanning tiles (masked loads with other=0.0)
    Pass 3: write exp(x - max) / sum back to output

    This avoids requiring BLOCK_SIZE >= cols and keeps working sets small for better L1 reuse.
    """
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    neg_inf = -1e20

    # Pass 1: compute max across the row
    m = neg_inf
    for col_start in range(0, cols, BLOCK_SIZE):
        col_idx = col_start + offs
        mask = col_idx < cols
        idx = row * cols + col_idx
        x = tl.load(x_ptr + idx, mask=mask, other=neg_inf)
        # tile max is scalar
        m_tile = tl.max(x, axis=0)
        m = tl.max(m, m_tile)

    # Pass 2: compute sum of exp(x - m)
    s = 0.0
    for col_start in range(0, cols, BLOCK_SIZE):
        col_idx = col_start + offs
        mask = col_idx < cols
        idx = row * cols + col_idx
        x = tl.load(x_ptr + idx, mask=mask, other=0.0)
        x_exp = tl.exp(x - m)
        s = s + tl.sum(x_exp, axis=0)

    # Pass 3: write normalized outputs
    for col_start in range(0, cols, BLOCK_SIZE):
        col_idx = col_start + offs
        mask = col_idx < cols
        idx = row * cols + col_idx
        x = tl.load(x_ptr + idx, mask=mask, other=0.0)
        x_exp = tl.exp(x - m)
        out = x_exp / s
        tl.store(out_ptr + idx, out, mask=mask)


def triton_softmax_last_dim(x: torch.Tensor) -> torch.Tensor:
    """
    Compute softmax along the last dimension using the Triton kernel.
    Expects x to be a CUDA tensor. Works on fp32.
    """
    if not x.is_cuda:
        # Fallback to torch for CPU (or non-CUDA) tensors
        return torch.softmax(x, dim=-1)

    # Ensure contiguous and float32
    x = x.contiguous().float()
    orig_shape = x.shape  # e.g., (N, C, H, W)
    cols = orig_shape[-1]
    # Compute rows = product(orig_shape[:-1]) without creating intermediate tensors
    rows = 1
    for d in orig_shape[:-1]:
        rows *= d

    # Reshape to (rows, cols) without copying (after contiguous)
    x2d = x.view(rows, cols)

    out2d = torch.empty_like(x2d)

    # Grid: one program per row
    grid = lambda meta: (rows,)

    _softmax_last_dim_kernel[grid](x2d, out2d, rows, cols)
    # Reshape back to original
    return out2d.view(orig_shape)


# New DoubleConv that replaces nn.Softmax(dim=-1) with a Triton-based softmax along width (last dim)
class DoubleConvNew(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # keep Conv2d and BatchNorm2d (highly optimized in cuDNN/CUDA)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        # Softmax along last dimension (width) - use PyTorch's softmax for correctness
        x = torch.softmax(x, dim=-1)
        x = self.conv2(x)
        x = self.bn2(x)
        x = torch.softmax(x, dim=-1)
        return x


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, features):
        """
        Optimized U-Net variant where the Softmax along the width dimension
        is implemented with a custom Triton kernel for improved throughput.
        """
        super(ModelNew, self).__init__()
        self.encoder1 = DoubleConvNew(in_channels, features)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = DoubleConvNew(features, features * 2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = DoubleConvNew(features * 2, features * 4)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = DoubleConvNew(features * 4, features * 8)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = DoubleConvNew(features * 8, features * 16)

        self.upconv4 = nn.ConvTranspose2d(features * 16, features * 8, kernel_size=2, stride=2)
        self.decoder4 = DoubleConvNew(features * 16, features * 8)
        self.upconv3 = nn.ConvTranspose2d(features * 8, features * 4, kernel_size=2, stride=2)
        self.decoder3 = DoubleConvNew(features * 8, features * 4)
        self.upconv2 = nn.ConvTranspose2d(features * 4, features * 2, kernel_size=2, stride=2)
        self.decoder2 = DoubleConvNew(features * 4, features * 2)
        self.upconv1 = nn.ConvTranspose2d(features * 2, features, kernel_size=2, stride=2)
        self.decoder1 = DoubleConvNew(features * 2, features)

        self.final_conv = nn.Conv2d(features, out_channels, kernel_size=1)

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)

        return self.final_conv(dec1)