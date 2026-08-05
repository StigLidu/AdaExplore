import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

# Autotune configurations for different block sizes / warps
AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK": 512},  num_warps=4, num_stages=2),
    triton.Config({"BLOCK": 1024}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK": 2048}, num_warps=8, num_stages=2),
]

# A simple per-batch chunked copy + ReLU kernel.
# It copies elems_src_per_batch contiguous elements per batch from src_ptr to
# dst_ptr at a fixed dst offset within each batch (dst_offset_within_batch),
# applying ReLU in-place during the copy. BLOCK is a constexpr vector length.
@triton.autotune(configs=AUTOTUNE_CONFIGS, key=['elems_src_per_batch', 'elems_dst_per_batch'])
@triton.jit
def _copy_relu_kernel(src_ptr, dst_ptr, elems_src_per_batch, elems_dst_per_batch, dst_offset_within_batch, BLOCK: tl.constexpr):
    batch = tl.program_id(0)
    chunk = tl.program_id(1)

    start = chunk * BLOCK
    offs = start + tl.arange(0, BLOCK)
    mask = offs < elems_src_per_batch

    # source index within flattened source: batch * elems_src_per_batch + offs
    src_idx = batch * elems_src_per_batch + offs
    # destination index within flattened output: batch * elems_dst_per_batch + dst_offset_within_batch + offs
    dst_idx = batch * elems_dst_per_batch + dst_offset_within_batch + offs

    vals = tl.load(src_ptr + src_idx, mask=mask, other=0.0)
    relu_vals = tl.where(vals > 0.0, vals, 0.0)
    tl.store(dst_ptr + dst_idx, relu_vals, mask=mask)


def triton_relu_concat(x1: torch.Tensor, x2: torch.Tensor):
    """
    Fuses ReLU on x1 and x2 and concatenates them along the channel dimension.
    Expects x1 and x2 to be CUDA tensors with same batch, height, width shapes.
    This implementation launches two contiguous copy+ReLU kernels (one per input)
    that write into the correct per-batch region of the output, avoiding per-element
    index decoding and branching.
    """
    # If not on CUDA, fallback to pure PyTorch for correctness
    if not x1.is_cuda or not x2.is_cuda:
        return torch.cat([F.relu(x1), F.relu(x2)], dim=1)

    assert x1.dtype == torch.float32 and x2.dtype == torch.float32, "Only fp32 supported in Triton kernel."
    assert x1.ndim == 4 and x2.ndim == 4, "Expected tensors in NCHW format."
    assert x1.shape[0] == x2.shape[0] and x1.shape[2] == x2.shape[2] and x1.shape[3] == x2.shape[3], "Batch/Spatial dims must match."

    # Ensure contiguous flat layout
    x1c = x1.contiguous()
    x2c = x2.contiguous()

    N, C1, H, W = x1c.shape
    _, C2, _, _ = x2c.shape

    spatial = H * W
    elems_src_per_batch_1 = C1 * spatial
    elems_src_per_batch_2 = C2 * spatial
    elems_dst_per_batch = (C1 + C2) * spatial

    # Prepare output
    out = torch.empty((N, C1 + C2, H, W), device=x1c.device, dtype=x1c.dtype)

    # Flatten everything
    x1_flat = x1c.view(-1)
    x2_flat = x2c.view(-1)
    out_flat = out.view(-1)

    # Launch kernel for x1 -> out (x1 occupies the first C1 channels of each batch in out)
    grid1 = lambda meta: (N, (elems_src_per_batch_1 + meta["BLOCK"] - 1) // meta["BLOCK"])
    # dst offset within each batch for x1 is 0
    _copy_relu_kernel[grid1](x1_flat, out_flat, elems_src_per_batch_1, elems_dst_per_batch, 0)

    # Launch kernel for x2 -> out (x2 occupies channels starting at C1 within each batch)
    grid2 = lambda meta: (N, (elems_src_per_batch_2 + meta["BLOCK"] - 1) // meta["BLOCK"])
    dst_offset_for_x2 = C1 * spatial  # within a batch offset
    _copy_relu_kernel[grid2](x2_flat, out_flat, elems_src_per_batch_2, elems_dst_per_batch, dst_offset_for_x2)

    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, squeeze_channels, expand1x1_channels, expand3x3_channels):
        super(ModelNew, self).__init__()
        # Keep convolution modules in PyTorch; fuse activation+concat with Triton kernel.
        self.squeeze = nn.Conv2d(in_channels, squeeze_channels, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)

        self.expand1x1 = nn.Conv2d(squeeze_channels, expand1x1_channels, kernel_size=1)
        self.expand3x3 = nn.Conv2d(squeeze_channels, expand3x3_channels, kernel_size=3, padding=1)

    def forward(self, x):
        # Squeeze conv + activation stays in PyTorch (inplace relu for memory)
        x = self.squeeze_activation(self.squeeze(x))

        # Expand convolutions (pre-activation)
        e1 = self.expand1x1(x)
        e2 = self.expand3x3(x)

        # Fused ReLU + concat using Triton kernels (fast GPU path)
        return triton_relu_concat(e1, e2)