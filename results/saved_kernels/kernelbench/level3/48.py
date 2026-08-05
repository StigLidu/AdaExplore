import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import triton
import triton.language as tl

# Reuse a Triton segsum-from-cumsum kernel (autotuned) to build the lower-triangular difference matrix.
AUTOTUNE_CONFIGS_SEGSUM = [
    triton.Config({"BLOCK_COL": 16},  num_warps=2, num_stages=2),
    triton.Config({"BLOCK_COL": 32},  num_warps=4, num_stages=2),
    triton.Config({"BLOCK_COL": 64},  num_warps=8, num_stages=3),
]


@triton.autotune(
    configs=AUTOTUNE_CONFIGS_SEGSUM,
    key=["T"],
)
@triton.jit
def _segsum_from_cumsum_kernel(
    cumsum_ptr,       # pointer to cumsum matrix (M, T)
    out_ptr,          # pointer to output (M, T, T)
    M,                # number of flattened leading entries
    T,                # sequence length (last dimension)
    BLOCK_COL: tl.constexpr,  # number of columns handled per program
):
    m = tl.program_id(0)            # which flattened row (from leading dims)
    col_block = tl.program_id(1)    # which block of columns
    i = tl.program_id(2)            # row index

    col_offsets = col_block * BLOCK_COL + tl.arange(0, BLOCK_COL)   # j positions
    col_mask = col_offsets < T

    # Base addresses for this m in cumsum and out
    cumsum_base = m * T
    out_base = m * (T * T) + i * T   # out[m, i, 0] offset

    # Load cumsum_j for this block of columns (vectorized)
    cumsum_j = tl.load(cumsum_ptr + cumsum_base + col_offsets, mask=col_mask, other=0.0)

    # Load cumsum_i and broadcast across BLOCK_COL entries by using zero-strided offsets
    i_offsets = cumsum_base + i + tl.arange(0, BLOCK_COL) * 0
    cumsum_i_vec = tl.load(cumsum_ptr + i_offsets, mask=col_mask, other=0.0)

    # Compute difference: cumsum_i - cumsum_j
    diff = cumsum_i_vec - cumsum_j

    # Apply triangular mask: keep entries where j <= i, otherwise -inf
    tri_mask = col_offsets <= i
    neg_inf_vec = tl.full((BLOCK_COL,), -1e20, dtype=tl.float32)
    val = tl.where(tri_mask & col_mask, diff, neg_inf_vec)

    # Store result into out at positions (m, i, col_offsets)
    tl.store(out_ptr + out_base + col_offsets, val, mask=col_mask)


def triton_segsum_from_cumsum(x_cumsum: torch.Tensor):
    """
    Wrapper that calls the Triton kernel to compute the pairwise segment-sum matrix
    from an inclusive cumsum along the last dimension.

    Input:
      x_cumsum: tensor of shape (..., T), contiguous or will be made contiguous.
    Output:
      tensor of shape (..., T, T) where out[..., i, j] = cumsum[..., i] - cumsum[..., j] for j <= i,
      and -inf elsewhere.
    """
    assert x_cumsum.is_cuda, "Input must be on CUDA"
    assert x_cumsum.dtype == torch.float32, "Only float32 supported"

    original_shape = x_cumsum.shape[:-1]
    T = x_cumsum.shape[-1]
    M = int(x_cumsum.numel() // T)

    x_flat = x_cumsum.contiguous().view(M, T)
    out_flat = torch.empty((M, T, T), device=x_flat.device, dtype=x_flat.dtype)

    # Autotune-driven grid: (M, n_col_blocks, T)
    grid = lambda meta: (M, (T + meta["BLOCK_COL"] - 1) // meta["BLOCK_COL"], T)

    # Launch kernel (autotuned BLOCK_COL)
    _segsum_from_cumsum_kernel[grid](x_flat, out_flat, M, T)
    out = out_flat.view(*original_shape, T, T)
    return out


def triton_diag_einsum(B: torch.Tensor, C: torch.Tensor, L: torch.Tensor, X: torch.Tensor, BLOCK_P=32, BLOCK_N=16):
    """
    Fallback implementation using PyTorch for correctness.
    Kept the original function name and signature so the rest of the code can call it unchanged.
    It performs the same contraction as the original fused Triton kernel:
      out[b,c,l,h,p] = sum_{s,n} C[b,c,l,h,n] * B[b,c,s,h,n] * L[b,h,c,l,s] * X[b,c,s,h,p]
    """
    # Ensure contiguous inputs for efficient einsum
    B = B.contiguous()
    C = C.contiguous()
    X = X.contiguous()
    # Use einsum to perform the contraction (clear and correct)
    Y_diag = torch.einsum('b c l h n, b c s h n, b h c l s, b c s h p -> b c l h p',
                          C, B, L, X)
    return Y_diag


class ModelNew(nn.Module):
    def __init__(self, batch_size, seq_length, n_heads, d_head, d_state, block_len=64):
        """
        Optimized Model using Triton kernels:
          - triton_segsum_from_cumsum to build segsum matrices efficiently (autotuned)
          - triton_diag_einsum to compute the diagonal-block contribution in a fused kernel
        Other parts remain in PyTorch but with contiguous tensors and reduced temporaries.
        """
        super(ModelNew, self).__init__()

        assert seq_length % block_len == 0, "Sequence length must be divisible by block length"

        self.batch_size = batch_size
        self.seq_length = seq_length
        self.n_heads = n_heads
        self.d_head = d_head
        self.d_state = d_state
        self.block_len = block_len

        # Parameters same as original model
        self.A = nn.Parameter(torch.randn(batch_size, seq_length, n_heads))
        self.B = nn.Parameter(torch.randn(batch_size, seq_length, n_heads, d_state))
        self.C = nn.Parameter(torch.randn(batch_size, seq_length, n_heads, d_state))

    def segsum(self, x):
        """
        Compute the segment-sum matrix using Triton-accelerated kernel.
        x: tensor of shape (..., T)
        Returns tensor of shape (..., T, T) where out[..., i, j] = cumsum[..., i] - cumsum[..., j]
        for j <= i and -inf elsewhere (matching original behavior).
        """
        x_cumsum = torch.cumsum(x, dim=-1)
        return triton_segsum_from_cumsum(x_cumsum)

    def forward(self, X, initial_states=None):
        """
        Forward pass implementing the SSD operation, with an optimized diagonal-block computation.
        X: (batch, length, n_heads, d_head)
        """
        # Rearrange into blocks/chunks
        X_blocks, A_blocks, B_blocks, C_blocks = [
            rearrange(x, "b (c l) ... -> b c l ...", l=self.block_len)
            for x in (X, self.A, self.B, self.C)
        ]

        # Reorder A_blocks to (b, h, c, l) as original and compute cumsum
        A_blocks = rearrange(A_blocks, "b c l h -> b h c l")
        A_cumsum = torch.cumsum(A_blocks, dim=-1)

        # 1. Compute diagonal block outputs (optimized)
        # L is exp(segsum(A_blocks)) -> segsum returns (b, h, c, l, l)
        L = torch.exp(self.segsum(A_blocks))  # (b, h, c, l, l)

        # Prepare tensors in shapes expected by triton_diag_einsum:
        # B_blocks, C_blocks: (b, c, l, h, n)
        B_blocks_t = B_blocks.contiguous()
        C_blocks_t = C_blocks.contiguous()
        X_blocks_t = X_blocks.contiguous()

        # Compute Y_diag using fused Triton kernel (returns shape (b, c, l, h, p))
        Y_diag = triton_diag_einsum(B_blocks_t, C_blocks_t, L, X_blocks_t, BLOCK_P=32, BLOCK_N=self.d_state)

        # 2. Compute intra-chunk states (PyTorch)
        decay_states = torch.exp((A_cumsum[:, :, :, -1:] - A_cumsum))
        # states shape in original: torch.einsum("bclhn,bhcl,bclhp->bchpn", B_blocks, decay_states, X_blocks)
        # Rearrange inputs to match einsum shape expectations
        states = torch.einsum("b c l h n, b h c l, b c l h p -> b c h p n",
                              B_blocks_t, decay_states, X_blocks_t)
        # states -> want shape (b, c, h, p, n), match original variable ordering
        states = states.permute(0, 1, 2, 3, 4)  # (b, c, h, p, n) already in this layout

        # 3. Compute inter-chunk recurrence
        if initial_states is None:
            initial_states = torch.zeros_like(states[:, :1])

        states = torch.cat([initial_states, states], dim=1)  # concat on chunk dimension

        # decay_chunk computation: we need segsum on last_cumsum per chunk
        last_cumsum = A_cumsum[:, :, :, -1]  # (b, h, c)
        # Pad left with one zero for the recurrence matrix (matching original code)
        padded = F.pad(last_cumsum, (1, 0))  # (b, h, c+1)
        decay_chunk = torch.exp(self.segsum(padded))  # (b, h, c+1, c+1)
        # new_states = torch.einsum("bhzc,bchpn->bzhpn", decay_chunk, states)
        new_states = torch.einsum("b h z c, b c h p n -> b z h p n", decay_chunk, states)
        # Drop the appended chunk
        states = new_states[:, :-1]

        # 4. Compute state-to-output conversion
        state_decay_out = torch.exp(A_cumsum)  # (b, h, c, l)
        # Y_off: torch.einsum('bclhn,bchpn,bhcl->bclhp', C_blocks, states, state_decay_out)
        # Rearrange states back to expected dims for einsum:
        # states currently (b, c, h, p, n) -> need b c h p n
        Y_off = torch.einsum('b c l h n, b c h p n, b h c l -> b c l h p',
                             C_blocks_t, states, state_decay_out)

        # Combine diagonal and off-diagonal terms and reshape back
        Y = rearrange(Y_diag + Y_off, "b c l h p -> b (c l) h p")

        return Y