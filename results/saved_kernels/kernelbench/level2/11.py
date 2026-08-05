import torch
import torch.nn as nn
import triton
import triton.language as tl

# Autotune configs for the fused kernel
AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK": 128},  num_warps=4, num_stages=2),
    triton.Config({"BLOCK": 256},  num_warps=4, num_stages=2),
    triton.Config({"BLOCK": 512},  num_warps=8, num_stages=2),
    triton.Config({"BLOCK": 1024}, num_warps=8, num_stages=3),
]

@triton.autotune(configs=AUTOTUNE_CONFIGS, key=["N", "C", "H", "W", "H2", "W2"])
@triton.jit
def _bn_tanh_maxpool_kernel(
    x_ptr,           # input tensor pointer (N, C, H, W)
    out_ptr,         # output pointer (N, C, H2, W2)
    weight_ptr,      # batchnorm weight per channel (C,)
    bias_ptr,        # batchnorm bias per channel (C,)
    mean_ptr,        # running mean per channel (C,)
    var_ptr,         # running var per channel (C,)
    N, C, H, W, H2, W2, eps,
    BLOCK: tl.constexpr
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)  # linear indices into output flattened tensor
    total_outputs = N * C * H2 * W2
    mask = offsets < total_outputs

    # compute multi-dim indices from linear offsets: idx -> (n, c, h2, w2)
    idx = offsets
    # compute n
    tmp = idx // (C * H2 * W2)
    n = tmp
    rem = idx - tmp * (C * H2 * W2)
    # compute c
    tmp = rem // (H2 * W2)
    c = tmp
    rem = rem - tmp * (H2 * W2)
    # compute h2, w2
    h2 = rem // W2
    w2 = rem - h2 * W2

    # convert to input top-left coords
    h0 = h2 * 2
    w0 = w2 * 2

    # Compute input flattened indices for the four 2x2 positions
    # input_index = ((n * C + c) * H + h) * W + w
    nc_base = (n * C + c)  # vector
    in_idx0 = nc_base * H * W + (h0 * W + w0)
    in_idx1 = nc_base * H * W + (h0 * W + (w0 + 1))
    in_idx2 = nc_base * H * W + (((h0 + 1) * W) + w0)
    in_idx3 = nc_base * H * W + (((h0 + 1) * W) + (w0 + 1))

    # masks for input loads (in case H/W not divisible by 2). Here general code handles boundaries.
    mask0 = mask & (h0 < H) & (w0 < W)
    mask1 = mask & (h0 < H) & ((w0 + 1) < W)
    mask2 = mask & ((h0 + 1) < H) & (w0 < W)
    mask3 = mask & ((h0 + 1) < H) & ((w0 + 1) < W)

    # Load the four input values (as fp32). Provide other=0.0 for out-of-bounds.
    x0 = tl.load(x_ptr + in_idx0, mask=mask0, other=0.0)
    x1 = tl.load(x_ptr + in_idx1, mask=mask1, other=0.0)
    x2 = tl.load(x_ptr + in_idx2, mask=mask2, other=0.0)
    x3 = tl.load(x_ptr + in_idx3, mask=mask3, other=0.0)

    # Load batchnorm parameters per channel
    c_offsets = c
    param_mask = c_offsets < C
    w = tl.load(weight_ptr + c_offsets, mask=param_mask, other=1.0)
    b = tl.load(bias_ptr + c_offsets, mask=param_mask, other=0.0)
    m = tl.load(mean_ptr + c_offsets, mask=param_mask, other=0.0)
    v = tl.load(var_ptr + c_offsets, mask=param_mask, other=1.0)

    # compute inverse std
    inv_std = 1.0 / tl.sqrt(v + eps)

    # apply batchnorm: y = (x - mean) * inv_std * weight + bias
    y0 = (x0 - m) * inv_std * w + b
    y1 = (x1 - m) * inv_std * w + b
    y2 = (x2 - m) * inv_std * w + b
    y3 = (x3 - m) * inv_std * w + b

    # compute tanh(y) using a numerically stable formulation:
    # tanh(y) = sign(y) * (1 - exp(-2 * |y|)) / (1 + exp(-2 * |y|))
    # compute exp_term = exp(-2*abs(y))
    ay0 = tl.abs(y0); ay1 = tl.abs(y1); ay2 = tl.abs(y2); ay3 = tl.abs(y3)
    e0 = tl.exp(-2.0 * ay0); e1 = tl.exp(-2.0 * ay1); e2 = tl.exp(-2.0 * ay2); e3 = tl.exp(-2.0 * ay3)
    tanh0 = tl.where(y0 >= 0.0, (1.0 - e0) / (1.0 + e0), (e0 - 1.0) / (1.0 + e0))
    tanh1 = tl.where(y1 >= 0.0, (1.0 - e1) / (1.0 + e1), (e1 - 1.0) / (1.0 + e1))
    tanh2 = tl.where(y2 >= 0.0, (1.0 - e2) / (1.0 + e2), (e2 - 1.0) / (1.0 + e2))
    tanh3 = tl.where(y3 >= 0.0, (1.0 - e3) / (1.0 + e3), (e3 - 1.0) / (1.0 + e3))

    # compute max-pool across the 2x2 window
    m01 = tl.maximum(tanh0, tanh1)
    m23 = tl.maximum(tanh2, tanh3)
    mout = tl.maximum(m01, m23)

    # store output flattened (same linear idx)
    tl.store(out_ptr + offsets, mout, mask=mask)

def triton_bn_tanh_maxpool(x: torch.Tensor, batch_norm: nn.BatchNorm2d):
    """
    Fuses BatchNorm2d (using running stats and affine), tanh, and 2x2 maxpool (stride 2).
    x: (N, C, H, W) float32 CUDA tensor
    Returns: (N, C, H//2, W//2) float32 CUDA tensor
    """
    assert x.is_cuda, "Input must be on CUDA"
    assert x.dtype == torch.float32, "Only float32 supported"
    N, C, H, W = x.shape
    H2 = H // 2
    W2 = W // 2

    x = x.contiguous()

    # Prepare output
    out = torch.empty((N, C, H2, W2), dtype=x.dtype, device=x.device)

    # extract batchnorm params (use running stats and affine params).
    # When in training mode, BatchNorm uses batch statistics; reproduce that for correctness.
    # Ensure tensors are on GPU
    if batch_norm.weight is None:
        weight = torch.ones(C, device=x.device, dtype=x.dtype)
    else:
        weight = batch_norm.weight.to(device=x.device, dtype=x.dtype).contiguous()
    if batch_norm.bias is None:
        bias = torch.zeros(C, device=x.device, dtype=x.dtype)
    else:
        bias = batch_norm.bias.to(device=x.device, dtype=x.dtype).contiguous()
    eps = float(batch_norm.eps)

    if batch_norm.training:
        # compute per-channel batch mean and variance over N,H,W (like PyTorch's BatchNorm forward)
        # unbiased=False to match BatchNorm implementation
        mean = x.mean(dim=(0, 2, 3)).to(device=x.device, dtype=x.dtype).contiguous()
        var  = x.var(dim=(0, 2, 3), unbiased=False).to(device=x.device, dtype=x.dtype).contiguous()
    else:
        mean = batch_norm.running_mean.to(device=x.device, dtype=x.dtype).contiguous()
        var  = batch_norm.running_var.to(device=x.device, dtype=x.dtype).contiguous()

    # Flattened pointers are handled by Triton when passing tensors directly.
    n_elements = N * C * H2 * W2
    # grid: number of blocks to cover all output elements
    grid = lambda meta: ((n_elements + meta["BLOCK"] - 1) // meta["BLOCK"],)

    # launch kernel
    _bn_tanh_maxpool_kernel[grid](
        x, out,
        weight, bias, mean, var,
        N, C, H, W, H2, W2, eps
    )
    return out

class ModelNew(nn.Module):
    """
    Optimized model:
      - Uses PyTorch ConvTranspose2d as-is.
      - Replaces BatchNorm2d -> Tanh -> MaxPool2d sequence with a fused Triton kernel.
      - Leaves GroupNorm as PyTorch module (to preserve its running/affine behavior).
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups, num_groups):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        # Keep original batch_norm module (we will use its running stats & affine parameters in the fused kernel)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        # Keep group norm as PyTorch module
        self.group_norm = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)

    def forward(self, x):
        # conv transpose using PyTorch (highly optimized)
        x = self.conv_transpose(x)
        # fused BatchNorm (running stats & affine) + tanh + 2x2 maxpool
        x = triton_bn_tanh_maxpool(x, self.batch_norm)
        # group normalization using PyTorch
        x = self.group_norm(x)
        return x