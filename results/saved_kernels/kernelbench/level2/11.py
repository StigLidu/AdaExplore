import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def _bn_tanh_maxpool2x2_kernel(
    x_ptr,            # input pointer (N, C, H, W), contiguous
    out_ptr,          # output pointer (N, C, H//2, W//2), contiguous
    mean_ptr,         # running mean per channel (C,)
    var_ptr,          # running var per channel (C,)
    weight_ptr,       # bn weight per channel (C,)
    bias_ptr,         # bn bias per channel (C,)
    N, C, H, W, H_out, W_out, eps,
    BLOCK: tl.constexpr
):
    # Each program handles BLOCK output elements
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    idx = pid * BLOCK + offs  # flattened index over output elements
    n_output = N * C * H_out * W_out
    mask = idx < n_output

    # Decompose flattened index into (n, c, h_out, w_out)
    tmp = idx
    w_o = tmp % W_out
    tmp = tmp // W_out
    h_o = tmp % H_out
    tmp = tmp // H_out
    c = tmp % C
    n = tmp // C

    # Compute input top-left coords for the 2x2 window
    h_in = h_o * 2
    w_in = w_o * 2

    # Compute base addresses for the four elements in the 2x2 window:
    # address = ((n * C + c) * H + h_in) * W + w_in
    base = ((n * C + c) * H + h_in) * W + w_in

    addr00 = base
    addr01 = base + 1
    addr10 = base + W
    addr11 = base + W + 1

    # Load the four values (use a large negative value as "other" for masked loads)
    neg_inf = -1e9
    x00 = tl.load(x_ptr + addr00, mask=mask, other=neg_inf)
    x01 = tl.load(x_ptr + addr01, mask=mask, other=neg_inf)
    x10 = tl.load(x_ptr + addr10, mask=mask, other=neg_inf)
    x11 = tl.load(x_ptr + addr11, mask=mask, other=neg_inf)

    # Load BN params for the channel (broadcasted)
    mean = tl.load(mean_ptr + c, mask=mask, other=0.0)
    var = tl.load(var_ptr + c, mask=mask, other=1.0)
    weight = tl.load(weight_ptr + c, mask=mask, other=1.0)
    bias = tl.load(bias_ptr + c, mask=mask, other=0.0)

    # Compute normalization: (x - mean) / sqrt(var + eps) * weight + bias
    inv_std = 1.0 / tl.sqrt(var + eps)

    # Inline BN + tanh computation for each of the four positions to avoid nested function
    y00 = (x00 - mean) * inv_std * weight + bias
    e00 = tl.exp(-2.0 * y00)
    t00 = (1.0 - e00) / (1.0 + e00)

    y01 = (x01 - mean) * inv_std * weight + bias
    e01 = tl.exp(-2.0 * y01)
    t01 = (1.0 - e01) / (1.0 + e01)

    y10 = (x10 - mean) * inv_std * weight + bias
    e10 = tl.exp(-2.0 * y10)
    t10 = (1.0 - e10) / (1.0 + e10)

    y11 = (x11 - mean) * inv_std * weight + bias
    e11 = tl.exp(-2.0 * y11)
    t11 = (1.0 - e11) / (1.0 + e11)

    # Max pool over the 2x2 window
    m0 = tl.maximum(t00, t01)
    m1 = tl.maximum(t10, t11)
    mout = tl.maximum(m0, m1)

    # Compute output flattened index and store
    out_idx = idx  # same layout ordering as computed
    tl.store(out_ptr + out_idx, mout, mask=mask)


class ModelNew(nn.Module):
    """
    Optimized model: keep ConvTranspose2d in PyTorch, fuse BatchNorm2d + Tanh + MaxPool2d(2x2)
    into a Triton kernel for speed, then apply GroupNorm via PyTorch.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups, num_groups):
        super(ModelNew, self).__init__()
        # Keep conv transpose in PyTorch for correctness and simplicity
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        # Keep BatchNorm module to hold parameters (we will read them in the Triton kernel)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        # Keep GroupNorm module to perform the final normalization
        self.group_norm = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)
        # Store eps for batchnorm
        self._bn_eps = float(self.batch_norm.eps)

    def forward(self, x):
        # conv_transpose (PyTorch)
        x = self.conv_transpose(x)  # shape: (N, C, H, W)
        # If not CUDA or we're in training mode, fall back to PyTorch ops to maintain correctness.
        # In training mode BatchNorm uses per-batch statistics (not running stats), so we cannot
        # safely fuse BN into the Triton kernel without a separate reduction to compute per-channel stats.
        if not x.is_cuda or self.batch_norm.training:
            x = self.batch_norm(x)
            x = torch.tanh(x)
            x = nn.functional.max_pool2d(x, kernel_size=2, stride=2)
            x = self.group_norm(x)
            return x

        x = x.contiguous()

        N, C, H, W = x.shape
        # Output spatial dims after 2x2 maxpool stride 2
        H_out = H // 2
        W_out = W // 2

        # Prepare output tensor
        out = torch.empty((N, C, H_out, W_out), device=x.device, dtype=x.dtype)

        # Prepare BN parameter tensors (ensure contiguous and on same device)
        running_mean = self.batch_norm.running_mean.contiguous().to(x.device)
        running_var = self.batch_norm.running_var.contiguous().to(x.device)
        weight = self.batch_norm.weight.contiguous().to(x.device)
        bias = self.batch_norm.bias.contiguous().to(x.device)
        eps = float(self.batch_norm.eps)

        # Total number of output elements
        n_output = N * C * H_out * W_out

        # Launch Triton kernel
        BLOCK = 1024
        grid = ( (n_output + BLOCK - 1) // BLOCK, )

        _bn_tanh_maxpool2x2_kernel[grid](
            x, out,
            running_mean, running_var, weight, bias,
            N, C, H, W, H_out, W_out, eps,
            BLOCK=BLOCK
        )

        # Finally apply GroupNorm using PyTorch
        out = self.group_norm(out)
        return out