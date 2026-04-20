import torch
import torch.nn as nn
import triton
import triton.language as tl

# Fused Triton kernel that tiles both batch and channels (multiple groups per program).
# Each program processes BLOCK_B rows and BLOCK_G groups (each group has BLOCK_CH channels).
# BLOCK_B, BLOCK_G, BLOCK_CH must be provided as constexpr kernel launch parameters.
@triton.jit
def _gn_lrelu_add_kernel(
    x_ptr,        # input pointer (B, C) row-major
    gamma_ptr,    # scale (C,)
    beta_ptr,     # bias (C,)
    out_ptr,      # output pointer (B, C)
    B,            # batch size (python int)
    C,            # num channels (python int)
    num_groups,   # number of groups (python int)
    eps,          # float eps for numerical stability
    negative_slope,  # float for LeakyReLU
    BLOCK_B: tl.constexpr,   # rows per program
    BLOCK_G: tl.constexpr,   # groups per program
    BLOCK_CH: tl.constexpr,  # channels per group (channels_per_group)
):
    pid_c = tl.program_id(0)  # group-block index (each program handles BLOCK_G groups)
    pid_b = tl.program_id(1)  # batch-block index (each program handles BLOCK_B rows)

    # batch rows handled by this program
    row_base = pid_b * BLOCK_B
    offs_b = tl.arange(0, BLOCK_B)                       # constexpr
    rows = row_base + offs_b
    mask_b = rows < B                                    # shape [BLOCK_B]

    # for each group handled by this program
    group_base = pid_c * BLOCK_G
    offs_ch = tl.arange(0, BLOCK_CH)                     # channels per group (constexpr)
    # Precompute row_offsets to linearize indexing: rows[:, None] * C
    row_offsets = rows[:, None] * C                      # shape [BLOCK_B, 1]

    # Loop over the groups handled by this program (BLOCK_G is constexpr so this loop is unrolled)
    for gi in range(0, BLOCK_G):
        group_id = group_base + gi
        # Guard: only process valid groups
        if group_id < num_groups:
            ch_base = group_id * BLOCK_CH
            ch_idx = ch_base + offs_ch                     # shape [BLOCK_CH]
            mask_ch = ch_idx < C                            # shape [BLOCK_CH]

            # linearized column offsets broadcasted across rows
            col_offsets = ch_idx[None, :]                   # shape [1, BLOCK_CH]
            idx = row_offsets + col_offsets                 # shape [BLOCK_B, BLOCK_CH]

            # combined mask for loads/stores
            mask = mask_b[:, None] & mask_ch[None, :]       # shape [BLOCK_B, BLOCK_CH]

            # load x tile (masked)
            x_tile = tl.load(x_ptr + idx, mask=mask, other=0.0)  # shape [BLOCK_B, BLOCK_CH]

            # compute number of valid channels in this group (scalar)
            count_ch = tl.sum(tl.where(mask_ch, 1.0, 0.0))  # scalar
            # inv_count per row (zero for fully-padded rows)
            inv_count = tl.where(mask_b, 1.0 / tl.where(count_ch > 0.0, count_ch, 1.0), 0.0)  # [BLOCK_B]

            # compute sums and sums-of-squares using masked load (invalid lanes already zero)
            sum_x = tl.sum(x_tile, 1)                    # [BLOCK_B]
            sum_x2 = tl.sum(x_tile * x_tile, 1)          # [BLOCK_B]

            # mean and variance per row
            mean = sum_x * inv_count                     # [BLOCK_B]
            var = sum_x2 * inv_count - mean * mean      # [BLOCK_B]
            invstd = tl.rsqrt(var + eps)                 # [BLOCK_B]

            # normalize: (x - mean) * invstd
            normalized = (x_tile - mean[:, None]) * invstd[:, None]  # [BLOCK_B, BLOCK_CH]

            # load affine params for this group's channels and broadcast to rows
            g_vals = tl.load(gamma_ptr + ch_idx, mask=mask_ch, other=1.0)  # [BLOCK_CH]
            b_vals = tl.load(beta_ptr + ch_idx, mask=mask_ch, other=0.0)   # [BLOCK_CH]

            g_vals = g_vals[None, :]  # [1, BLOCK_CH]
            b_vals = b_vals[None, :]  # [1, BLOCK_CH]

            # affine transform
            out_vals = normalized * g_vals + b_vals

            # LeakyReLU
            out_vals = tl.where(out_vals >= 0.0, out_vals, out_vals * negative_slope)

            # final elementwise add x + x => multiply by 2
            out_vals = out_vals * 2.0

            # store result (masked)
            tl.store(out_ptr + idx, out_vals, mask=mask)


def triton_groupnorm_lrelu_add(x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, num_groups: int, eps: float, negative_slope: float):
    """
    Wrapper to launch the Triton kernel that fuses GroupNorm, LeakyReLU, and doubling (x + x).
    x: (B, C)
    gamma, beta: (C,)
    num_groups: number of groups for GroupNorm (must divide C)
    """
    assert x.is_cuda and gamma.is_cuda and beta.is_cuda, "All tensors must be on CUDA."
    assert x.dtype == torch.float32 and gamma.dtype == torch.float32 and beta.dtype == torch.float32, "Only fp32 supported."

    x = x.contiguous()
    gamma = gamma.contiguous()
    beta = beta.contiguous()

    B, C = x.shape
    assert C % num_groups == 0, "num_groups must divide channels"
    channels_per_group = C // num_groups

    # Tiling parameters chosen to increase work per program and reduce kernel-launch overhead.
    # BLOCK_B: rows per program; BLOCK_G: groups per program; BLOCK_CH: channels per group (constexpr)
    BLOCK_B = 16
    BLOCK_G = 4
    BLOCK_CH = channels_per_group

    # Grid: number of group-blocks, number of batch-blocks
    grid = ( (num_groups + BLOCK_G - 1) // BLOCK_G, (B + BLOCK_B - 1) // BLOCK_B )

    out = torch.empty_like(x)
    # Launch kernel. Provide constexpr args BLOCK_B, BLOCK_G, BLOCK_CH.
    _gn_lrelu_add_kernel[grid](
        x, gamma, beta, out,
        B, C, num_groups, float(eps), float(negative_slope),
        BLOCK_B=BLOCK_B, BLOCK_G=BLOCK_G, BLOCK_CH=BLOCK_CH
    )
    return out


class ModelNew(nn.Module):
    """
    Optimized model that uses a fused Triton kernel to perform GroupNorm, LeakyReLU, and the final elementwise add.
    The linear layer (fc) is left as a standard PyTorch nn.Linear (uses cuBLAS), while the subsequent ops are fused.
    """
    def __init__(self, input_size, hidden_size, num_groups, eps=1e-5, negative_slope=0.01):
        super(ModelNew, self).__init__()
        self.fc = nn.Linear(input_size, hidden_size)
        self.num_groups = num_groups
        self.eps = eps
        self.negative_slope = negative_slope

        # GroupNorm affine parameters (per-channel)
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=torch.float32))
        self.bias = nn.Parameter(torch.zeros(hidden_size, dtype=torch.float32))

    def forward(self, x):
        # x: (B, input_size)
        x = self.fc(x)  # (B, hidden_size)
        # fused groupnorm + leakyrelu + add
        x = triton_groupnorm_lrelu_add(x, self.weight, self.bias, self.num_groups, self.eps, self.negative_slope)
        return x


# Keep the same helper functions for input generation as the original script
batch_size = 1024
input_size = 8192
hidden_size = 8192
num_groups = 512

def get_inputs():
    return [torch.rand(batch_size, input_size).cuda().float()]

def get_init_inputs():
    return [input_size, hidden_size, num_groups]