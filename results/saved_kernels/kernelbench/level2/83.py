import torch
import torch.nn as nn
import triton
import triton.language as tl

# Autotune a few block sizes/warp choices to hit best performance on A6000.
AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK": 8192},    num_warps=4, num_stages=2),
    triton.Config({"BLOCK": 16384},   num_warps=4, num_stages=2),
    triton.Config({"BLOCK": 32768},   num_warps=8, num_stages=2),
    triton.Config({"BLOCK": 65536},   num_warps=8, num_stages=2),
    triton.Config({"BLOCK": 131072},  num_warps=8, num_stages=2),
    triton.Config({"BLOCK": 262144},  num_warps=8, num_stages=2),
]

@triton.autotune(configs=AUTOTUNE_CONFIGS, key=['n_elements'])
@triton.jit
def _const_dropout_kernel(
    out_ptr,        # pointer to output float32
    n_elements,     # total number of elements (int)
    val,            # scalar float32 value to write for kept lanes (min_value * scale)
    keep_thresh,    # scalar int32/uint32 threshold: floor(keep_prob * 2**32)
    seed,           # scalar int32 seed
    BLOCK: tl.constexpr
):
    """
    Write a tensor of length n_elements where each element is either 'val' (kept if RNG < keep_thresh)
    or 0.0 (dropped). RNG is generated per-lane using a cheap 32-bit LCG/xorshift mix.
    Comparison is performed in integer space to avoid expensive uint->float casts.
    """
    pid = tl.program_id(0)
    start = pid * BLOCK
    offs = start + tl.arange(0, BLOCK)
    mask = offs < n_elements

    # 32-bit unsigned values for RNG
    offs_u = tl.cast(offs, tl.uint32)
    seed_u = tl.cast(seed, tl.uint32)
    pid_u = tl.cast(pid, tl.uint32)

    # Simple LCG-like mix
    s = offs_u * tl.cast(1664525, tl.uint32) + tl.cast(1013904223, tl.uint32) + seed_u + pid_u
    # xorshift-like further mixing
    s = s ^ (s >> tl.cast(16, tl.uint32))
    s = s * tl.cast(22695477, tl.uint32) + tl.cast(1, tl.uint32)

    # integer-threshold compare (avoid uint->float cast)
    keep = s < tl.cast(keep_thresh, tl.uint32)
    val_f = tl.cast(val, tl.float32)
    out_vals = tl.where(keep, val_f, 0.0)

    tl.store(out_ptr + offs, out_vals, mask=mask)


def _make_const_dropout_output(x: torch.Tensor, out_shape, min_value: float, dropout_p: float, training: bool):
    """
    Produce the final tensor for the sequence:
      x = conv(...) ; x = norm(x)
      x = torch.min(x, min_value)
      x = torch.clamp(x, min=min_value, max=max_value)
      x = dropout(x)

    Observations:
      - torch.min(x, min_value) followed by clamp(min_value, ...) yields a tensor
        where every element equals min_value, independent of x.
      - Therefore, we can synthesize the final tensor directly, applying dropout (with scaling) if training.
      - Special-case: if min_value == 0.0, the final tensor is all zeros both in train and eval,
        and dropout on zeros yields zeros -> short-circuit to a zero tensor.
    """
    device = x.device
    dtype = x.dtype

    # If min_value is exactly 0.0, the collapsed tensor is all zeros; dropout doesn't change zeros.
    if float(min_value) == 0.0:
        return torch.zeros(out_shape, device=device, dtype=dtype)

    # If not training or no dropout, final tensor is constant min_value.
    if (not training) or (dropout_p == 0.0):
        return torch.full(out_shape, float(min_value), device=device, dtype=dtype)

    # Allocate output tensor (contiguous)
    out = torch.empty(out_shape, device=device, dtype=dtype)

    n_elements = out.numel()
    if n_elements == 0:
        return out

    keep_prob = 1.0 - float(dropout_p)
    # guard against degenerate keep_prob
    if keep_prob <= 0.0:
        # all dropped -> zeros
        out.zero_()
        return out

    scale = 1.0 / keep_prob
    val = float(min_value) * float(scale)

    # random seed for kernel
    seed = int(torch.randint(0, 1 << 30, (1,), device=device, dtype=torch.int32).item())

    # Precompute integer threshold on host to avoid uint->float casts in the kernel.
    # Map keep_prob in [0,1] to integer threshold in [0, 2**32 - 1].
    raw_thresh = int(max(0.0, min(keep_prob, 1.0)) * (1 << 32))
    keep_thresh = min(raw_thresh, (1 << 32) - 1)

    # grid depends on the BLOCK selected by autotune; use a meta-aware grid lambda
    grid = lambda meta: ((n_elements + meta["BLOCK"] - 1) // meta["BLOCK"],)

    _const_dropout_kernel[grid](
        out,
        n_elements,
        float(val),
        int(keep_thresh),
        int(seed),
    )

    return out


class ModelNew(nn.Module):
    """
    Optimized model:
      - Keep Conv3d and GroupNorm modules as attributes so parameters/shapes remain consistent.
      - Avoid executing expensive conv/norm since subsequent operations collapse the output to a constant.
      - Synthesize the final tensor directly. In training, apply dropout by generating the mask inside a Triton kernel.
      - Special-case when min_value == 0.0 to return zeros immediately (common in many configs).
    """
    def __init__(self, in_channels, out_channels, kernel_size, groups, min_value, max_value, dropout_p):
        super(ModelNew, self).__init__()
        # Keep original modules so parameters exist and shapes are consistent
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.norm = nn.GroupNorm(groups, out_channels)
        # Keep a Dropout module for API/semantics, but we perform the actual operation in our fused generator.
        self.dropout = nn.Dropout(dropout_p)

        self.min_value = float(min_value)
        self.max_value = float(max_value)
        self.dropout_p = float(dropout_p)

    def forward(self, x):
        # Compute expected output shape of the Conv3d (without performing the conv)
        N, C_in, D_in, H_in, W_in = x.shape
        conv = self.conv

        def out_dim(L_in, k, p, d, s):
            return (L_in + 2 * p - d * (k - 1) - 1) // s + 1

        # kernel_size, padding, dilation, stride may be ints or tuples
        kD, kH, kW = conv.kernel_size if isinstance(conv.kernel_size, tuple) else (conv.kernel_size,)*3
        pD, pH, pW = conv.padding if isinstance(conv.padding, tuple) else (conv.padding,)*3
        dD, dH, dW = conv.dilation if isinstance(conv.dilation, tuple) else (conv.dilation,)*3
        sD, sH, sW = conv.stride if isinstance(conv.stride, tuple) else (conv.stride,)*3

        D_out = out_dim(D_in, kD, pD, dD, sD)
        H_out = out_dim(H_in, kH, pH, dH, sH)
        W_out = out_dim(W_in, kW, pW, dW, sW)

        out_shape = (N, conv.out_channels, D_out, H_out, W_out)

        # Use the fused generator that writes the constant (and dropout mask) directly.
        out = _make_const_dropout_output(x, out_shape, self.min_value, self.dropout_p, self.training)
        return out