import torch
import torch.nn as nn
import triton
import triton.language as tl

# Autotune configurations tuned for Ampere (NVIDIA A6000). We tune BLOCK_B (batch tile)
# and try different num_warps / num_stages to find the best setting at runtime.
AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK_B": 8},  num_warps=2, num_stages=2),
    triton.Config({"BLOCK_B": 8},  num_warps=4, num_stages=2),
    triton.Config({"BLOCK_B": 8},  num_warps=8, num_stages=3),
    triton.Config({"BLOCK_B": 16}, num_warps=2, num_stages=2),
    triton.Config({"BLOCK_B": 16}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_B": 16}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_B": 32}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_B": 32}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_B": 64}, num_warps=8, num_stages=3),
]

@triton.autotune(
    configs=AUTOTUNE_CONFIGS,
    key=["B", "C", "G", "NBLOCK_B"],
)
@triton.jit
def _gn_lrelu_kernel(
    inp_ptr,            # input pointer (B, C)
    out_ptr,            # output pointer (B, C)
    gamma_ptr,          # gamma pointer (C,)
    beta_ptr,           # beta pointer (C,)
    B,                  # batch size
    C,                  # num channels
    G,                  # num groups
    NBLOCK_B,           # number of blocks along batch dim (ceildiv(B, BLOCK_B))
    eps,                # eps for groupnorm
    neg_slope,          # leaky relu negative slope
    BLOCK_C: tl.constexpr,  # group size (constexpr)
    BLOCK_B: tl.constexpr   # batch tile size (constexpr)
):
    """
    Each Triton program handles one (group_idx, batch_block) tile:
      - group_idx selects which group of channels is processed (size BLOCK_C)
      - batch_block selects a contiguous block of batch rows (size BLOCK_B)

    The kernel computes mean+var across the BLOCK_C channels for each sample in the batch block,
    applies scale/shift (fused), LeakyReLU, and writes result. The final doubling (x+x) is
    pre-folded into gamma and beta on host side to avoid an extra per-element op.
    """

    pid = tl.program_id(0)
    # decode (group_idx, block_b)
    group_idx = pid // NBLOCK_B
    block_b = pid - group_idx * NBLOCK_B
    b_start = block_b * BLOCK_B

    c_start = group_idx * BLOCK_C
    offs_c = c_start + tl.arange(0, BLOCK_C)
    mask_c = offs_c < C

    # Load gamma and beta for this group (masked)
    gamma = tl.load(gamma_ptr + offs_c, mask=mask_c, other=1.0)
    beta = tl.load(beta_ptr + offs_c, mask=mask_c, other=0.0)

    # Process a BLOCK_B x BLOCK_C tile in one shot (vectorized over the batch-tile)
    # Build row and channel offsets for the 2D tile
    offs_b = b_start + tl.arange(0, BLOCK_B)                # (BLOCK_B,)
    mask_b = offs_b < B                                     # (BLOCK_B,)
    inp_offs_b = offs_b[:, None] * C                        # (BLOCK_B, 1)
    inp_offs_c = offs_c[None, :]                            # (1, BLOCK_C)
    inp_offsets = inp_offs_b + inp_offs_c                   # (BLOCK_B, BLOCK_C)
    mask2 = mask_b[:, None] & mask_c[None, :]               # (BLOCK_B, BLOCK_C)

    # load tile (BLOCK_B, BLOCK_C)
    x = tl.load(inp_ptr + inp_offsets, mask=mask2, other=0.0)

    # per-row mean and variance (reduce over channels -> axis=1)
    s = tl.sum(x, axis=1)                                   # (BLOCK_B,)
    mean = s / BLOCK_C                                      # (BLOCK_B,)

    x_centered = x - mean[:, None]                          # broadcast to (BLOCK_B, BLOCK_C)
    sq = x_centered * x_centered
    var = tl.sum(sq, axis=1) / BLOCK_C                      # (BLOCK_B,)

    invstd = 1.0 / tl.sqrt(var + eps)                       # (BLOCK_B,)

    # fuse scale and shift with proper broadcasting across rows
    scale = invstd[:, None] * gamma[None, :]                # (BLOCK_B, BLOCK_C)
    shift = beta[None, :] - mean[:, None] * scale           # (BLOCK_B, BLOCK_C)
    out = x * scale + shift

    # LeakyReLU (broadcasting works across the tile)
    out = tl.where(out > 0.0, out, out * neg_slope)

    # store the whole tile
    out_offsets = inp_offsets
    tl.store(out_ptr + out_offsets, out, mask=mask2)


def triton_groupnorm_leaky_relu_double(x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor,
                                       num_groups: int, eps: float, negative_slope: float):
    """
    Host wrapper that:
      - ensures contiguity,
      - folds the final doubling (x + x) into gamma and beta,
      - launches the autotuned Triton kernel.
    """
    assert x.is_cuda and gamma.is_cuda and beta.is_cuda, "Tensors must be on CUDA."
    assert x.dtype == torch.float32 and gamma.dtype == torch.float32 and beta.dtype == torch.float32, "Only fp32 supported."

    B, C = x.shape
    G = num_groups
    assert C % G == 0, "num_channels must be divisible by num_groups"
    group_size = C // G

    # make contiguous and prepare output
    x_contig = x.contiguous()
    out = torch.empty_like(x_contig)

    # fold doubling into gamma and beta to avoid extra per-element arithmetic in kernel
    gamma_contig = (gamma.contiguous() * 2.0)
    beta_contig = (beta.contiguous() * 2.0)

    # compute NBLOCK_B given the BLOCK_B chosen by autotuner (we don't know it here, but
    # grid size uses NBLOCK_B computed with the chosen BLOCK_B during kernel launch).
    # For grid we supply G * NBLOCK_B where NBLOCK_B = ceildiv(B, BLOCK_B)
    # We pass BLOCK_C=group_size and BLOCK_B as constexpr via autotuned configs.
    # To compute NBLOCK_B generically we set it to ceildiv(B, smallest possible BLOCK_B)=B/64 ceil max,
    # but the autoscheduler uses the key and will provide proper mapping. To be safe, compute for BLOCK_B=1 and
    # pass actual NBLOCK_B; Triton will accept a NBLOCK_B value larger than needed (unused programs masked).
    # Simpler: choose NBLOCK_B = ceildiv(B, 1) = B (safe upper bound). However, that would create too large grid.
    # Correct approach: compute NBLOCK_B for a typical BLOCK_B value (we'll compute using 1 and let unreachable pids mask).
    # Instead, we compute NBLOCK_B using the minimum BLOCK_B in the configs (8) to keep grid reasonable.
    min_block_b = 8
    NBLOCK_B = (B + min_block_b - 1) // min_block_b

    # Grid: one program per (group, batch-block) with NBLOCK_B based on min BLOCK_B.
    # Triton's autotuner will pick an actual BLOCK_B from configs; extra program ids (if any) will be masked inside kernel.
    grid = (G * NBLOCK_B,)

    # Launch kernel with BLOCK_C set to group_size (constexpr)
    _gn_lrelu_kernel[grid](
        x_contig, out, gamma_contig, beta_contig,
        B, C, G, NBLOCK_B, eps, negative_slope,
        BLOCK_C=group_size
    )

    return out


class ModelNew(nn.Module):
    """
    Optimized Model that uses PyTorch's fast Linear (cuBLAS/cuDNN) for the matmul + bias,
    and a Triton fused kernel for GroupNorm + LeakyReLU + final doubling (x + x folded into params).
    """
    def __init__(self, input_size, hidden_size, num_groups, eps=1e-5, negative_slope=0.01):
        super(ModelNew, self).__init__()
        self.fc = nn.Linear(input_size, hidden_size)
        # keep GroupNorm module only to hold learnable parameters (weight, bias) and metadata
        self.gn = nn.GroupNorm(num_groups=num_groups, num_channels=hidden_size, eps=eps)
        self._eps = eps
        self._negative_slope = negative_slope

    def forward(self, x):
        # 1) linear layer: use optimized PyTorch implementation (cuBLAS/cuDNN)
        x = self.fc(x)

        # 2) fused GroupNorm + LeakyReLU + doubling via Triton
        out = triton_groupnorm_leaky_relu_double(
            x,
            self.gn.weight,
            self.gn.bias,
            num_groups=self.gn.num_groups,
            eps=self._eps,
            negative_slope=self._negative_slope,
        )
        return out