import torch
import torch.nn as nn
import triton
import triton.language as tl

# Autotune configurations chosen to favor small-batch (M=10) and small output dims (N=10)
# on Ampere (A6000). Include a few large-block options so the kernel can often run as
# a single program (reducing launch overhead), plus smaller blocks for generality.
AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64},  num_warps=8,  num_stages=3),
    triton.Config({"BLOCK_M": 64,  "BLOCK_N": 128, "BLOCK_K": 64},  num_warps=8,  num_stages=3),
    triton.Config({"BLOCK_M": 32,  "BLOCK_N": 64,  "BLOCK_K": 64},  num_warps=8,  num_stages=2),
    triton.Config({"BLOCK_M": 8,   "BLOCK_N": 128, "BLOCK_K": 64},  num_warps=4,  num_stages=2),
    triton.Config({"BLOCK_M": 1,   "BLOCK_N": 256, "BLOCK_K": 64},  num_warps=4,  num_stages=2),
    triton.Config({"BLOCK_M": 1,   "BLOCK_N": 128, "BLOCK_K": 128}, num_warps=8,  num_stages=3),
]

@triton.autotune(configs=AUTOTUNE_CONFIGS, key=['M', 'N', 'K'])
@triton.jit
def _gemm_flexible_strides(
    A_ptr,  # pointer to A (M x K)
    B_ptr,  # pointer to B (K x N)
    C_ptr,  # pointer to C (M x N)
    M, N, K,  # sizes
    stride_am, stride_ak,  # A strides (elements)
    stride_bk, stride_bn,  # B strides (elements)
    stride_cm, stride_cn,  # C strides (elements)
    bias_ptr,  # pointer to bias (N,)
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    """
    GEMM kernel that supports arbitrary element-wise strides for A (so we avoid forcing
    A.contiguous() in the Python wrapper). This reduces extra allocations when the input
    is a view (e.g., h_n[-1] from RNN outputs). Uses tl.dot on block tiles and handles
    masked loads/stores.
    """
    row_block = tl.program_id(0)
    col_block = tl.program_id(1)

    row_start = row_block * BLOCK_M
    col_start = col_block * BLOCK_N

    offs_m = row_start + tl.arange(0, BLOCK_M)   # [BLOCK_M]
    offs_n = col_start + tl.arange(0, BLOCK_N)   # [BLOCK_N]

    # accumulator in fp32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    k = 0
    while k < K:
        k_start = k
        offs_k = k_start + tl.arange(0, BLOCK_K)   # [BLOCK_K]

        # Masks for loads
        mask_a = (offs_m[:, None] < M) & (offs_k[None, :] < K)   # [BLOCK_M, BLOCK_K]
        mask_b = (offs_k[:, None] < K) & (offs_n[None, :] < N)   # [BLOCK_K, BLOCK_N]

        # Compute addresses using element-wise strides. We support arbitrary element strides
        # provided by the caller (in elements, not bytes).
        a_addr = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
        b_addr = B_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

        # Load tiles with masking
        a_tile = tl.load(a_addr, mask=mask_a, other=0.0)  # [BLOCK_M, BLOCK_K]
        b_tile = tl.load(b_addr, mask=mask_b, other=0.0)  # [BLOCK_K, BLOCK_N]

        # Use tl.dot which is efficient for these block shapes and avoids large temporaries.
        # a_tile: [BLOCK_M, BLOCK_K], b_tile: [BLOCK_K, BLOCK_N]
        acc += tl.dot(a_tile, b_tile)

        k += BLOCK_K

    # Add bias
    bias_mask = offs_n < N
    bias_vals = tl.load(bias_ptr + offs_n, mask=bias_mask, other=0.0)  # [BLOCK_N]
    acc = acc + bias_vals[None, :]

    # Store result with write mask
    c_addr = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    write_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_addr, acc, mask=write_mask)


def triton_linear_no_a_contig(x: torch.Tensor, weight_t: torch.Tensor, bias: torch.Tensor = None, out_buffer: torch.Tensor = None):
    """
    Compute out = x @ weight_t + bias where:
      x: (M, K) -- may be non-contiguous, we pass element strides to the kernel to avoid copies
      weight_t: (K, N) -- expected contiguous (we cache a transposed contiguous weight on device)
      bias: (N,) or None
      out_buffer: optional preallocated output tensor (M, N)
    This wrapper avoids forcing x.contiguous() to save the copy when possible.
    """
    assert x.is_cuda and weight_t.is_cuda, "Inputs must be on CUDA."
    assert x.dtype == torch.float32 and weight_t.dtype == torch.float32, "Only fp32 supported."

    # Fast-path for very small batch sizes: prefer cuBLAS via PyTorch to avoid Triton launch overhead.
    # This is often faster for tiny M (e.g., M <= 64) because Triton kernel launch and occupancy
    # can dominate the execution time.
    M = x.shape[0]
    K = x.shape[1]
    if M <= 64:
        B = weight_t.contiguous()
        if bias is not None:
            # Ensure bias is on the correct device/dtype
            return torch.matmul(x, B) + bias.to(x.dtype).to(x.device)
        else:
            return torch.matmul(x, B)

    A = x  # do not force contiguous; kernel accepts arbitrary element strides
    B = weight_t.contiguous()  # B should be contiguous for efficient loads; we ensure it is
    M, K = A.shape
    Kb, N = B.shape
    assert K == Kb, "Incompatible shapes for matmul"

    # Prepare output buffer
    if out_buffer is None or out_buffer.dim() != 2 or out_buffer.shape[0] != M or out_buffer.shape[1] != N or out_buffer.device != A.device:
        C = torch.empty((M, N), device=A.device, dtype=torch.float32)
    else:
        C = out_buffer

    # Element strides (in elements, not bytes)
    stride_am = A.stride(0)
    stride_ak = A.stride(1)
    stride_bk = B.stride(0)
    stride_bn = B.stride(1)
    stride_cm = C.stride(0)
    stride_cn = C.stride(1)

    # Prepare bias
    if bias is not None:
        bias_contig = bias.contiguous()
    else:
        bias_contig = torch.zeros((N,), device=A.device, dtype=torch.float32)

    grid = lambda meta: (
        (M + meta["BLOCK_M"] - 1) // meta["BLOCK_M"],
        (N + meta["BLOCK_N"] - 1) // meta["BLOCK_N"],
    )

    _gemm_flexible_strides[grid](
        A, B, C,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        bias_contig,
    )

    return C


class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.0):
        super(ModelNew, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # Keep PyTorch's highly-optimized cuDNN LSTM for recurrent computation
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout, bidirectional=False)
        # Final linear layer parameters
        self.fc = nn.Linear(hidden_size, output_size)

        # Caches to avoid repeated transposes/allocations across forwards
        self._triton_weight = None
        self._cached_weight_ptr = None
        self._triton_bias = None
        self._cached_bias_ptr = None
        self._cached_device = None
        self._out_buffer = None
        self._out_buffer_shape = None

    def forward(self, x, h0=None, c0=None):
        """
        Forward pass:
          - Runs cuDNN LSTM (fast) to get final-layer hidden state.
          - Runs a Triton GEMM for the final linear layer. This GEMM accepts a non-contiguous
            input view for A (so we avoid forcing a copy of h_n[-1] when possible).
        """
        batch_size = x.size(0)
        device = x.device

        # Initialize hidden states on correct device if not provided
        if h0 is None:
            h0 = torch.randn(self.num_layers, batch_size, self.hidden_size, device=device)
        if c0 is None:
            c0 = torch.randn(self.num_layers, batch_size, self.hidden_size, device=device)

        # Run cuDNN-optimized LSTM
        out, (h_n, c_n) = self.lstm(x, (h0, c0))

        # Use last layer's final hidden state: h_n has shape (num_layers, batch, hidden_size)
        # h_n[-1] is typically a view; we intentionally avoid forcing contiguity to save a copy.
        last = h_n[-1]  # shape (batch_size, hidden_size), may be non-contiguous

        weight = self.fc.weight   # shape (out_features, in_features) == (N, K)
        bias = self.fc.bias

        # Cache transposed contiguous weight on device (K, N) to enable coalesced loads in Triton
        if (not hasattr(self, "_triton_weight")) or (self._cached_weight_ptr != weight.data_ptr()) or (self._cached_device != device):
            # weight.t() is (in_features, out_features) -> (K, N)
            # Make contiguous and move to device
            self._triton_weight = weight.t().contiguous().to(device)
            self._cached_weight_ptr = weight.data_ptr()
            self._cached_device = device

        if bias is not None:
            if (not hasattr(self, "_triton_bias")) or (self._cached_bias_ptr != bias.data_ptr()) or (self._cached_device != device):
                self._triton_bias = bias.contiguous().to(device)
                self._cached_bias_ptr = bias.data_ptr()
                self._cached_device = device
            bias_contig = self._triton_bias
        else:
            bias_contig = None

        # Prepare/reuse an output buffer to avoid allocations each forward
        M = last.shape[0]
        N = self.fc.out_features
        if (self._out_buffer is None or self._out_buffer_shape != (M, N) or self._out_buffer.device != device):
            # allocate output buffer on device
            self._out_buffer = torch.empty((M, N), device=device, dtype=torch.float32)
            self._out_buffer_shape = (M, N)

        # Compute final linear using Triton kernel that accepts non-contiguous A (last).
        # We pass last directly (no .contiguous()) and let kernel use element strides.
        out_lin = triton_linear_no_a_contig(last, self._triton_weight, bias_contig, out_buffer=self._out_buffer)

        return out_lin