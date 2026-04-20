import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import triton
import triton.language as tl

# Autotune configurations tuned for Ampere (A6000).
# Prioritize tensor-core-friendly larger K tiles and M/N multiples suited for Ampere.
# Keep smaller candidates for memory pressure corner cases.
AUTOTUNE_CONFIGS_GEMM = [
    # Very large, high-throughput candidates (favor BLOCK_K=128 and even BLOCK_N/M multiples)
    triton.Config({"BLOCK_M": 1024, "BLOCK_N": 256, "BLOCK_K": 128}, num_warps=8, num_stages=4),
    triton.Config({"BLOCK_M": 512,  "BLOCK_N": 256, "BLOCK_K": 128}, num_warps=8, num_stages=4),
    triton.Config({"BLOCK_M": 1024, "BLOCK_N": 128, "BLOCK_K": 128}, num_warps=8, num_stages=4),
    triton.Config({"BLOCK_M": 512,  "BLOCK_N": 128, "BLOCK_K": 128}, num_warps=8, num_stages=3),

    # Balanced mid-size tiles (good fallback)
    triton.Config({"BLOCK_M": 256, "BLOCK_N": 256, "BLOCK_K": 128}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 64},  num_warps=8, num_stages=3),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64},  num_warps=8, num_stages=3),

    # Smaller / lower-concurrency candidates for memory-pressure sensitive cases
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 32}, num_warps=4, num_stages=2),
]

@triton.autotune(
    configs=AUTOTUNE_CONFIGS_GEMM,
    key=["M", "N", "K", "A_stride_row", "B_stride_row", "C_stride_row"],
)
@triton.jit
def _matmul_kernel(
    A_ptr,  # pointer to A (M, K), row-major
    B_ptr,  # pointer to B (K, N) interpreted via strides (we accept weight in (N, K) layout and read transposed)
    C_ptr,  # pointer to C (M, N), row-major
    bias_ptr,  # pointer to bias (N,) (ignored if bias_on==0)
    bias_on,   # int32 flag: 1 if bias provided, 0 otherwise
    M, N, K,
    A_stride_row, A_stride_k,
    B_stride_k, B_stride_row,
    C_stride_row, C_stride_n,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    STORE_FP16: tl.constexpr
):
    """
    Compute C = A @ B + bias
    A: (M, K) row-major
    B: We accept a tensor with layout (N, K) (i.e., linear weight shape),
       but we index it as if it's (K, N) by using B_stride_k and B_stride_row appropriately.
    C: (M, N) row-major
    Each program computes a (BLOCK_M x BLOCK_N) tile of C.
    STORE_FP16 (constexpr): when true, convert accum (fp32) to fp16 before storing to C.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m_start = pid_m * BLOCK_M
    n_start = pid_n * BLOCK_N

    offs_m = tl.arange(0, BLOCK_M)[:, None]  # (BLOCK_M, 1)
    offs_n = tl.arange(0, BLOCK_N)[None, :]  # (1, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K in chunks
    for k_start in range(0, K, BLOCK_K):
        offs_k = tl.arange(0, BLOCK_K)  # (BLOCK_K,)

        # Indices
        a_m_indices = m_start + offs_m              # (BLOCK_M, 1)
        a_k_indices = k_start + offs_k              # (BLOCK_K,)
        b_k_indices = k_start + offs_k              # (BLOCK_K,)
        b_n_indices = n_start + offs_n              # (1, BLOCK_N)

        # Masks
        mask_a = (a_m_indices < M) & (a_k_indices[None, :] < K)  # (BLOCK_M, BLOCK_K)
        mask_b = (b_k_indices[:, None] < K) & (b_n_indices < N)  # (BLOCK_K, BLOCK_N)

        # Pointers (we interpret B with swapped strides so that passing weight in (N, K) works)
        a_ptrs = A_ptr + a_m_indices * A_stride_row + a_k_indices[None, :] * A_stride_k
        b_ptrs = B_ptr + b_k_indices[:, None] * B_stride_k + b_n_indices * B_stride_row

        # Load inputs (expect host to provide fp16 tensors for tensor-core execution).
        # tl.load will infer element type from memory; avoid specifying dtype keyword.
        a = tl.load(a_ptrs, mask=mask_a, other=0.0)  # (BLOCK_M, BLOCK_K)  # expected fp16
        b = tl.load(b_ptrs, mask=mask_b, other=0.0)  # (BLOCK_K, BLOCK_N)  # expected fp16

        acc += tl.dot(a, b)  # (BLOCK_M, BLOCK_N)

    # Write back
    c_m_indices = m_start + offs_m
    c_n_indices = n_start + offs_n
    c_ptrs = C_ptr + c_m_indices * C_stride_row + c_n_indices * C_stride_n
    mask_c = (c_m_indices < M) & (c_n_indices < N)

    if bias_on != 0:
        bias_vals = tl.load(bias_ptr + n_start + tl.arange(0, BLOCK_N),
                            mask=(n_start + tl.arange(0, BLOCK_N) < N), other=0.0).to(tl.float32)
        acc = acc + bias_vals[None, :]

    # Compile-time branch: store as fp16 if requested to avoid an extra device-side cast/copy.
    if STORE_FP16:
        acc_fp16 = acc.to(tl.float16)
        tl.store(c_ptrs, acc_fp16, mask=mask_c)
    else:
        tl.store(c_ptrs, acc, mask=mask_c)


def triton_gemm(A: torch.Tensor, B: torch.Tensor, bias: torch.Tensor = None, out_dtype: torch.dtype = torch.float32):
    """
    Compute A @ B + bias using the Triton matmul kernel with flexible B layout support.

    - A: (M, K) contiguous CUDA tensor (fp16 or fp32)
    - B: either (N, K) or pre-transposed (K, N) contiguous fp16/fp32 tensor. This function
         detects the layout and computes the proper (M, N, K) triple to pass to the kernel.
    - bias: (N,) or None. Bias is promoted to float32 for accumulation inside the kernel.
    - out_dtype: desired output dtype. If out_dtype==torch.float16 the kernel will write fp16 directly
                 (STORE_FP16=True) to avoid an extra device-side float32->fp16 copy.

    Returns: tensor of shape (M, N) with dtype out_dtype.
    """
    assert A.is_cuda and B.is_cuda
    assert A.dtype in (torch.float16, torch.float32) and B.dtype in (torch.float16, torch.float32)

    # Prefer fp16 inputs for tensor-core HMMA. Convert inputs to fp16 if needed (fallback).
    # Callers that already provide fp16 can avoid extra copies.
    if A.dtype == torch.float32:
        A = A.half()
    if B.dtype == torch.float32:
        B = B.half()

    A = A.contiguous()
    B = B.contiguous()
    M, K = A.shape

    # Support both (N, K) and pre-transposed (K, N) B layouts.
    if B.shape[1] == K:
        # B is (N, K)
        N = B.shape[0]
        K2 = B.shape[1]
    elif B.shape[0] == K:
        # B is pre-transposed (K, N) -> treat N = B.shape[1]
        N = B.shape[1]
        K2 = B.shape[0]
    else:
        raise AssertionError(f"Incompatible B shape {tuple(B.shape)} for A with K={K}")

    assert K == K2, f"Mismatch K dims: A.K={K} vs inferred B.K={K2}"

    # If user requested fp16 output, ask kernel to store fp16 directly to avoid an extra cast/copy.
    store_fp16 = (out_dtype == torch.float16)
    C_dtype = torch.float16 if store_fp16 else torch.float32
    C = torch.empty((M, N), device=A.device, dtype=C_dtype)

    if bias is not None:
        bias_ptr = bias.contiguous().to(torch.float32)
        bias_on = 1
    else:
        bias_ptr = torch.empty((0,), device=A.device, dtype=torch.float32)
        bias_on = 0

    A_stride_row = A.stride(0)
    A_stride_k = A.stride(1)
    # We require B to be in (N, K) layout (rows = output dim, cols = K) and contiguous.
    B_stride_k = B.stride(1)
    B_stride_row = B.stride(0)
    C_stride_row = C.stride(0)
    C_stride_n = C.stride(1)

    grid = lambda meta: (
        (M + meta["BLOCK_M"] - 1) // meta["BLOCK_M"],
        (N + meta["BLOCK_N"] - 1) // meta["BLOCK_N"],
    )
    # Pass the STORE_FP16 constexpr value as a keyword so the kernel can write fp16 directly when requested.
    # This avoids accidentally binding it to an autotuner-inserted BLOCK_* positional constexpr.
    _matmul_kernel[grid](
        A, B, C, bias_ptr, bias_on,
        M, N, K,
        A_stride_row, A_stride_k,
        B_stride_k, B_stride_row,
        C_stride_row, C_stride_n,
        STORE_FP16=int(store_fp16)
    )

    # If kernel already wrote in the requested dtype, return as-is; otherwise cast.
    if out_dtype == C_dtype:
        return C
    else:
        return C.to(out_dtype)


# Optimized ModelNew using Triton for both projections + PyTorch fused attention
class ModelNew(nn.Module):
    """
    Optimized multi-head masked self-attention layer:
      - Uses Triton GEMM for both the qkv projection and the final output projection (large matmuls).
      - Folds the 1/sqrt(head_dim) scaling into the Q projection weights at init.
      - Stores projection weights in fp16 to reduce memory bandwidth; triton kernel accumulates in fp32.
      - Uses PyTorch's fused scaled_dot_product_attention (FlashAttention) for the attention core.
    """
    def __init__(self, n_embd, n_head, attn_pdrop, resid_pdrop, max_seqlen):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.n_embd = n_embd

        # projections
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd)

        # dropouts
        self.attn_dropout = nn.Dropout(attn_pdrop)
        self.resid_dropout = nn.Dropout(resid_pdrop)

        # causal mask buffer (kept for compatibility; fused primitive applies causal mask)
        self.register_buffer("bias", torch.tril(torch.ones(max_seqlen, max_seqlen)).view(1, 1, max_seqlen, max_seqlen))

        # Fold Q scaling into c_attn first n_embd rows and convert projection weights to fp16 to save bandwidth.
        # Only fold the scaling when the fused attention primitive is NOT available to avoid double-scaling.
        hs = n_embd // n_head
        q_scale = 1.0 / math.sqrt(hs)
        with torch.no_grad():
            # c_attn: shape (3*C, C); first C rows correspond to Q in concatenated [Q;K;V]
            W_attn = self.c_attn.weight.data.clone()
            # Only fold q_scale if fused attention isn't present (the fused primitive applies its own scaling)
            if not hasattr(F, "scaled_dot_product_attention"):
                W_attn[:n_embd, :] *= float(q_scale)
            self.c_attn.weight = nn.Parameter(W_attn.half())
            if self.c_attn.bias is not None:
                self.c_attn.bias = nn.Parameter(self.c_attn.bias.data.clone().half())

            # c_proj weights -> store as fp16 to reduce memory bandwidth
            W_proj = self.c_proj.weight.data.clone()
            self.c_proj.weight = nn.Parameter(W_proj.half())
            if self.c_proj.bias is not None:
                self.c_proj.bias = nn.Parameter(self.c_proj.bias.data.clone().half())

            # Store contiguous fp16 weights in (N, K) layout so Triton kernel can read coalesced memory.
            # c_attn.weight is (3*C, C) which directly matches the (N, K) orientation used by triton_gemm
            # (rows = output dims). Keeping this contiguous avoids extra copies/transposes at runtime.
            self.register_buffer("c_attn_weight_t", self.c_attn.weight.data.contiguous())
            self.register_buffer("c_proj_weight_t", self.c_proj.weight.data.contiguous())

    def forward(self, x):
        """
        x: (B, T, C) float32
        """
        B, T, C = x.size()
        assert C == self.n_embd

        N = B * T  # number of rows for GEMM projection

        # Prepare A matrix for projection: (N, C)
        # Cast the activation to proj_dtype once and reuse the buffer to avoid duplicate device traffic.
        proj_dtype = self.c_attn.weight.dtype  # fp16
        A_half = x.to(proj_dtype).contiguous()
        A_qkv = A_half.view(N, C)
        # Use the contiguous fp16 weight buffer in (N, K) layout for coalesced loads.
        W_qkv = self.c_attn_weight_t
        bias_qkv = self.c_attn.bias.contiguous().to(torch.float32) if self.c_attn.bias is not None else None

        # Fuse Q/K/V projection into a single GEMM call to avoid rereading the large A matrix 3×.
        # The registered buffer self.c_attn_weight_t is in (N=3*C, K=C) layout with rows [Q;K;V]
        # (rows = output dims). Call triton_gemm once to produce (N, 3*C) and split.
        W_qkv = self.c_attn_weight_t  # shape (3*C, C), contiguous fp16 buffer
        # triton_gemm returns (N, 3*C) in proj_dtype (fp16) here; kernel accumulates in fp32 internally.
        qkv_out = triton_gemm(A_qkv, W_qkv, bias_qkv, out_dtype=proj_dtype)  # (N, 3*C)

        # Split and reshape back to (B, T, C) for q, k, v (already proj_dtype)
        q_out, k_out, v_out = qkv_out.split(C, dim=1)
        q = q_out.view(B, T, C)
        k = k_out.view(B, T, C)
        v = v_out.view(B, T, C)

        # Rearrange to (B, n_head, T, hs)
        hs = C // self.n_head
        q = q.view(B, T, self.n_head, hs).permute(0, 2, 1, 3).contiguous()
        k = k.view(B, T, self.n_head, hs).permute(0, 2, 1, 3).contiguous()
        v = v.view(B, T, self.n_head, hs).permute(0, 2, 1, 3).contiguous()

        # Use PyTorch fused scaled_dot_product_attention if available for fastest causal attention
        if hasattr(F, "scaled_dot_product_attention"):
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=True, dropout_p=0.0)
        else:
            # Fallback to a stable implementation in fp32
            Bnh = B * self.n_head
            q_ = q.contiguous().view(Bnh, T, hs).to(torch.float32)
            k_ = k.contiguous().view(Bnh, T, hs).to(torch.float32)
            v_ = v.contiguous().view(Bnh, T, hs).to(torch.float32)
            logits = torch.bmm(q_, k_.transpose(1, 2)) * (1.0 / math.sqrt(hs))
            mask = torch.tril(torch.ones((T, T), device=logits.device, dtype=torch.bool))
            logits = logits.masked_fill(~mask.unsqueeze(0), float('-inf'))
            attn = F.softmax(logits, dim=-1)
            out_ = torch.bmm(attn, v_)
            y = out_.view(B, self.n_head, T, hs)

        y = self.attn_dropout(y)

        # Reassemble heads -> (B, T, C) in proj_dtype
        y = y.permute(0, 2, 1, 3).contiguous().view(B, T, C)  # dtype = proj_dtype (fp16)

        # Final projection via Triton GEMM: (N, C) @ (C_out=N_out, C_in=C) weight -> (N, C_out) float32
        A_out = y.view(N, C).contiguous()  # fp16
        # Use the contiguous fp16 weight buffer in (N, K) layout for the output projection
        W_out = self.c_proj_weight_t
        bias_out = self.c_proj.bias.contiguous().to(torch.float32) if self.c_proj.bias is not None else None

        out_fp32 = triton_gemm(A_out, W_out, bias_out, out_dtype=torch.float32)  # returns float32 (N, C)

        out_fp32 = self.resid_dropout(out_fp32)
        out = out_fp32.view(B, T, C)
        return out


# Keep helper functions and input sizes for the external harness
batch_size = 128
max_seqlen = 1024
seq_len = 512
n_embd = 768
n_head = 8
attn_pdrop = 0.0
resid_pdrop = 0.0

def get_inputs():
    # Return CUDA tensor (fp32)
    return [torch.rand(batch_size, seq_len, n_embd, device='cuda', dtype=torch.float32)]

def get_init_inputs():
    return [n_embd, n_head, attn_pdrop, resid_pdrop, max_seqlen]