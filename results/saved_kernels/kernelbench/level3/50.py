import math
import torch
import torch.nn as nn
import triton
import triton.language as tl

# Triton kernel: apply causal mask (j > i_global -> 0) and ReLU in-place on a scores block tensor
# scores layout: (N, BQ, BK) where BK == i1 (number of keys considered for this query block)
# We launch the kernel with grid = (num_row_tiles, num_col_tiles, N)
@triton.jit
def _mask_and_relu_scores_kernel(
    scores_ptr,      # pointer to scores tensor (N, BQ, BK) in fp16
    N,               # number of combined heads (B * NH)
    BQ,              # number of query positions in this block
    BK,              # number of key positions in this block (i1)
    i0,              # global start index for this query block (so global_i = i0 + i_rel)
    stride0,         # scores.stride(0)
    stride1,         # scores.stride(1)
    stride2,         # scores.stride(2)
    BLOCK_M: tl.constexpr,  # tile rows (over BQ)
    BLOCK_N: tl.constexpr,  # tile cols (over BK)
):
    # program ids
    row_block = tl.program_id(0)
    col_block = tl.program_id(1)
    n = tl.program_id(2)

    row_start = row_block * BLOCK_M
    col_start = col_block * BLOCK_N

    rows = row_start + tl.arange(0, BLOCK_M)   # shape (BLOCK_M,)
    cols = col_start + tl.arange(0, BLOCK_N)   # shape (BLOCK_N,)

    row_in_bounds = rows < BQ
    col_in_bounds = cols < BK

    # pairwise i_rel (rows) and j_rel (cols)
    i_rel = rows[:, None]      # (BLOCK_M, 1)
    j_rel = cols[None, :]      # (1, BLOCK_N)

    # global i indices = i0 + i_rel
    # causal condition: j_rel <= (i0 + i_rel)
    # Note: j_rel contains key indices in [0, BK-1] relative to the current key block whose global j = j_rel (since keys start at 0 each block)
    tri_mask = j_rel <= (i0 + i_rel)  # (BLOCK_M, BLOCK_N)

    # valid bounds mask
    valid_mask = (row_in_bounds[:, None]) & (col_in_bounds[None, :])  # (BLOCK_M, BLOCK_N)

    # compute flattened offsets for load/store
    base = n * stride0
    offs = base + i_rel * stride1 + j_rel * stride2  # (BLOCK_M, BLOCK_N)
    offs_flat = tl.reshape(offs, (BLOCK_M * BLOCK_N,))

    mask_flat = tl.reshape(valid_mask, (BLOCK_M * BLOCK_N,))

    # Fast path: if this tile is fully above diagonal (all j > i), we can store zeros
    # That happens when col_start > (i0 + row_start + BLOCK_M - 1)
    if col_start > (i0 + row_start + BLOCK_M - 1):
        zeros = tl.zeros((BLOCK_M * BLOCK_N,), dtype=tl.float16)
        tl.store(scores_ptr + offs_flat, zeros, mask=mask_flat)
        return

    # Load values (out-of-bounds filled with 0.0)
    vals = tl.load(scores_ptr + offs_flat, mask=mask_flat, other=0.0)  # fp16 flat
    vals = tl.reshape(vals, (BLOCK_M, BLOCK_N))  # (BLOCK_M, BLOCK_N)

    # Apply triangular causal mask: keep entries where tri_mask is true, else zero
    tri_mask_f = tri_mask & valid_mask
    zeros_f = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float16)
    vals = tl.where(tri_mask_f, vals, zeros_f)

    # Apply ReLU: convert to fp32 for comparison, then back to fp16
    vals_f32 = vals.to(tl.float32)
    vals_f32 = tl.where(vals_f32 > 0.0, vals_f32, 0.0)
    vals = vals_f32.to(tl.float16)

    # store back
    vals_flat = tl.reshape(vals, (BLOCK_M * BLOCK_N,))
    tl.store(scores_ptr + offs_flat, vals_flat, mask=mask_flat)


def triton_causal_mask_relu_scores_(scores: torch.Tensor, i0: int, BLOCK_M: int = 64, BLOCK_N: int = 64):
    """
    In-place apply causal mask and ReLU to scores tensor of shape (N, BQ, BK)
    where BK equals the number of keys considered (i1), and i0 is the global start index for queries.
    This function launches a Triton kernel to efficiently zero out j > (i0 + i_rel) and apply ReLU.
    """
    assert scores.is_cuda and scores.dtype == torch.float16
    assert scores.dim() == 3
    N, BQ, BK = scores.shape

    scores = scores.contiguous()
    stride0, stride1, stride2 = scores.stride()
    num_row = (BQ + BLOCK_M - 1) // BLOCK_M
    num_col = (BK + BLOCK_N - 1) // BLOCK_N

    grid = (num_row, num_col, N)
    _mask_and_relu_scores_kernel[grid](
        scores, N, BQ, BK, i0,
        stride0, stride1, stride2,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N
    )


class ModelNew(nn.Module):
    """
    Optimized multi-head causal attention with ReLU using blocked streaming + Triton-assisted
    in-place causal masking and ReLU on score blocks.

    Approach:
    - Project inputs to Q,K,V and reshape to (N=B*NH, T, HS).
    - Use FP16 for heavy GEMMs to utilize Tensor Cores.
    - Process queries in blocks (BQ). For each query block [i0:i1), compute scores = q_block @ k_all^T
      (where k_all are keys for j in [0, i1)), then call a Triton kernel to apply causal mask & ReLU
      in-place on the scores tensor, and finally do out_block = scores @ v_all to accumulate results.
    - This avoids materializing a full (N, T, T) attention matrix and reduces memory traffic,
      while using a Triton kernel to efficiently apply masking + ReLU per score block.
    """

    def __init__(self, n_embd, n_head, max_seqlen):
        super().__init__()
        assert n_embd % n_head == 0
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        # keep c_proj for API parity (original forward didn't use it, but included earlier)
        self.c_proj = nn.Linear(n_embd, n_embd)
        self.register_buffer("bias", torch.tril(torch.ones(max_seqlen, max_seqlen))
                                     .view(1, 1, max_seqlen, max_seqlen))
        self.n_head = n_head
        self.n_embd = n_embd

        # Tunable block sizes. These are chosen to balance GEMM sizes for Tensor Cores
        # and the overhead of launching kernels / allocating temporaries.
        self.block_size_q = 256  # number of query positions per block
        self.block_size_k = 256  # number of key positions considered per block (i1)

    def forward(self, x):
        # x: (B, T, C)
        assert x.is_cuda, "This optimized implementation expects CUDA tensors."
        B, T, C = x.size()
        assert C == self.n_embd

        # Project to q, k, v
        qkv = self.c_attn(x)  # (B, T, 3C)
        q, k, v = qkv.split(self.n_embd, dim=2)  # each (B, T, C)

        HS = C // self.n_head
        N = B * self.n_head

        # reshape to (N, T, HS)
        def to_heads(z):
            z = z.view(B, T, self.n_head, HS).transpose(1, 2).contiguous()
            return z.view(N, T, HS)

        qh = to_heads(q)
        kh = to_heads(k)
        vh = to_heads(v)

        device = qh.device
        dtype_fp16 = torch.float16

        # Use mixed precision: scale Q and cast to fp16 for GEMMs
        scale = 1.0 / math.sqrt(HS)
        qh = (qh * scale).half().contiguous()   # (N, T, HS) fp16
        kh = kh.half().contiguous()
        vh = vh.half().contiguous()

        # Pre-transpose kh to (N, HS, T) once to avoid repeated transposes
        kh_T = kh.transpose(1, 2).contiguous()  # (N, HS, T)

        BQ = self.block_size_q
        # Prepare output accumulator in fp16
        out = torch.zeros((N, T, HS), dtype=dtype_fp16, device=device)

        # Precompute index tensors on device to build masks when needed (but main masking done in Triton)
        idx_k_full = torch.arange(T, device=device)

        # Loop over query blocks
        for i0 in range(0, T, BQ):
            i1 = min(i0 + BQ, T)
            bq = i1 - i0  # actual query block size

            # q_block: (N, bq, HS)
            q_block = qh[:, i0:i1, :]  # fp16

            # keys considered: j in [0, i1)
            k_all_T = kh_T[:, :, :i1]  # (N, HS, i1)
            v_all = vh[:, :i1, :]      # (N, i1, HS)

            # scores: (N, bq, i1) = q_block @ k_all_T
            scores = torch.bmm(q_block, k_all_T)  # fp16, contiguous

            # In-place apply causal mask and ReLU on scores using Triton kernel.
            # Provide i0 so Triton can compute triangular masking: keep j <= (i0 + i_rel)
            triton_causal_mask_relu_scores_(scores, i0, BLOCK_M=64, BLOCK_N=64)

            # Accumulate: out_block = scores @ v_all  -> (N, bq, HS)
            out_block = torch.bmm(scores, v_all)  # fp16

            # Store block
            out[:, i0:i1, :] = out_block

        # reshape back to (B, T, C) in fp32
        out = out.view(B, self.n_head, T, HS).transpose(1, 2).contiguous().view(B, T, C).float()
        return out


# Provide NewGELU in case external code expects it (original model defined it).
class NewGELU(nn.Module):
    def __init__(self):
        super(NewGELU, self).__init__()

    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))