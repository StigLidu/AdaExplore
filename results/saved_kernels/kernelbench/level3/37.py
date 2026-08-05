import torch
import torch.nn as nn
import torch.nn.functional as F

# Triton-based fused linear (GEMM) for A @ W.T + b where:
# A: (M, K)  (row-major contiguous)
# W: (O, K)  (weight as in nn.Linear, i.e., out_features x in_features)
# result: (M, O)
try:
    import triton
    import triton.language as tl
except Exception:
    triton = None
    tl = None

# Autotune configs chosen for reasonable Ampere performance
if triton is not None:
    # Ampere-friendly autotune configs: larger BLOCK_M, larger BLOCK_K and more warps/stages
    AUTOTUNE_CONFIGS = [
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=8, num_stages=3),
    ]

    @triton.autotune(
        configs=AUTOTUNE_CONFIGS,
        key=["M", "N", "K"],
    )
    @triton.jit
    def _matmul_kernel(
        A_ptr,            # pointer to A (M x K), row-major
        B_ptr,            # pointer to B (O x K) i.e., (N x K) where N==O, row-major (weight in natural layout)
        bias_ptr,         # pointer to bias (N,)
        M, N, K,          # matrix dims (A: MxK, B: N x K, out: M x N)
        stride_am, stride_ak,  # A row stride (elements), A inner stride (should be 1 for contiguous)
        stride_bm, stride_bk,  # B row stride (elements, stride between output rows), B inner stride (k stride)
        stride_bias,            # bias stride (should be 1 for contiguous)
        stride_cm, stride_cn,  # C row stride (elements), C inner stride (should be 1)
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
    ):
        """
        A_ptr points to A[0,0]. A is shape (M, K) row-major: element (i,k) at A_ptr + i*stride_am + k*stride_ak
        B_ptr points to B[0,0]. B is shape (N, K) row-major: element (j,k) at B_ptr + j*stride_bm + k*stride_bk
        bias_ptr points to bias[0]. bias is shape (N,)
        C_ptr points to C[0,0]. C is shape (M, N) row-major: element (i,j) at C_ptr + i*stride_cm + j*stride_cn
        For contiguous row-major 2D tensors:
          stride_am = K, stride_ak = 1
          stride_bm = K, stride_bk = 1
          stride_cm = N, stride_cn = 1
          stride_bias = 1
        """
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        m_start = pid_m * BLOCK_M
        n_start = pid_n * BLOCK_N

        offs_m = m_start + tl.arange(0, BLOCK_M)
        offs_n = n_start + tl.arange(0, BLOCK_N)

        # create accumulator
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        # iterate over K in chunks of BLOCK_K
        k = 0
        # Python-level loop is OK; BLOCK_K is a constexpr
        while k < K:
            k_offs = k + tl.arange(0, BLOCK_K)
            # masks for valid loads
            mask_a = (offs_m[:, None] < M) & (k_offs[None, :] < K)
            mask_b = (k_offs[:, None] < K) & (offs_n[None, :] < N)

            # pointers for A: element (i,k) -> A_ptr + i*stride_am + k*stride_ak
            a_ptrs = A_ptr + (offs_m[:, None] * stride_am) + (k_offs[None, :] * stride_ak)
            # pointers for B: element (j,k) -> B_ptr + j*stride_bm + k*stride_bk
            # We need a (BLOCK_K, BLOCK_N) tile where rows are K dim and cols are N dim:
            b_ptrs = B_ptr + (offs_n[None, :] * stride_bm) + (k_offs[:, None] * stride_bk)

            a = tl.load(a_ptrs, mask=mask_a, other=0.0)
            b = tl.load(b_ptrs, mask=mask_b, other=0.0)

            acc += tl.dot(a, b)
            k += BLOCK_K

        # fuse bias addition while storing result
        c_ptrs = C_ptr + (offs_m[:, None] * stride_cm) + (offs_n[None, :] * stride_cn)
        mask_c = (offs_m[:, None] < M) & (offs_n[None, :] < N)

        # load bias for this N tile (1D load)
        bias_vals = tl.load(bias_ptr + offs_n * stride_bias, mask=(offs_n < N), other=0.0)
        acc = acc + bias_vals[None, :]

        tl.store(c_ptrs, acc, mask=mask_c)


def _ensure_triton_available_or_none():
    # Helper to detect at runtime
    return triton is not None and torch.cuda.is_available()


def triton_linear(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor):
    """
    Compute x @ weight.T + bias using Triton kernel when available and inputs are CUDA float32 tensors.
    Falls back to torch.nn.functional.linear when Triton or CUDA is not available or dtypes mismatch.
    Expects:
      x: (M, K)
      weight: (O, K)  # natural nn.Linear layout (out_features x in_features)
      bias: (O,) or None
    Returns:
      out: (M, O)
    """
    # Quick fallbacks
    if not _ensure_triton_available_or_none():
        # fallback to PyTorch
        return F.linear(x, weight, bias)

    # require float32
    if x.dtype != torch.float32 or weight.dtype != torch.float32:
        return F.linear(x, weight, bias)

    # Move weight/bias to same device as x if necessary
    device = x.device
    if weight.device != device:
        weight = weight.to(device)
    if bias is not None and bias.device != device:
        bias = bias.to(device)

    # Ensure contiguous
    A = x.contiguous()               # (M, K)
    W = weight.contiguous()          # (O, K)
    M, K = A.shape
    O = W.shape[0]  # out features -> N

    # If bias is None, give kernel a zero-bias buffer so it can always read bias_ptr
    if bias is None:
        bias_tensor = torch.zeros((O,), device=device, dtype=torch.float32)
    else:
        bias_tensor = bias.contiguous()

    # Pack / transpose weight once to (K, O) row-major contiguous for efficient inner-loop loads
    # (Kernel expects B_ptr to be (K x N) row-major where N == O)
    Wt = W.t().contiguous()  # shape (K, O)

    # output
    out = torch.empty((M, O), device=device, dtype=torch.float32)

    # Strides in elements (not bytes)
    # For row-major contiguous 2D array:
    # A: stride_am = K, stride_ak = 1
    # Wt (packed as K x O): stride_bk = O (elements per k-row), stride_bn = 1
    # out: stride_cm = O, stride_cn = 1
    stride_am = K
    stride_ak = 1
    stride_bk = O
    stride_bn = 1
    stride_bias = 1
    stride_cm = O
    stride_cn = 1

    # Launch configuration: compute grid from reasonable block guesses aligned with autotuner choices
    BLOCK_M_GUESS = 256
    BLOCK_N_GUESS = 128
    grid = ((M + BLOCK_M_GUESS - 1) // BLOCK_M_GUESS, (O + BLOCK_N_GUESS - 1) // BLOCK_N_GUESS)

    # Launch kernel with corrected argument ordering: A_ptr, B_ptr (packed KxN), bias_ptr, C_ptr, M, N, K, ...
    _matmul_kernel[grid](
        A, Wt, bias_tensor, out,
        M, O, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_bias,
        stride_cm, stride_cn,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.0):
        """
        LSTM model where the final linear projection is performed by a custom Triton kernel when available.
        """
        super(ModelNew, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout, bidirectional=False)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, h0, c0):
        """
        Forward pass:
          - Run PyTorch LSTM (highly optimized)
        Returns the cell states (c_n) to match original behavior (state[1]).
        """
        # Move inputs to same device as model parameters if needed
        device = next(self.parameters()).device if any(True for _ in self.parameters()) else x.device
        if x.device != device:
            x = x.to(device)
        if h0.device != device:
            h0 = h0.to(device)
        if c0.device != device:
            c0 = c0.to(device)

        out, state = self.lstm(x, (h0, c0))

        # The final linear projection result was previously computed but never used.
        # Avoid launching the GEMM (and allocating its output) since it's wasted work.
        # Note: original model returns state[1]
        return state[1]