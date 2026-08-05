import torch
import torch.nn as nn

# Optional Triton acceleration for tensor copy when beneficial.
# If Triton isn't available, or tensors are on CPU, fall back to PyTorch operations.
try:
    import triton
    import triton.language as tl

    # Autotune configs tuned for Ampere (A6000).
    # Include smaller block sizes and a variety of warp choices so autotune can
    # pick a good tile/warp combination for different workloads.
    AUTOTUNE_CONFIGS = [
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_SIZE": 8192}, num_warps=8, num_stages=2),
    ]

    @triton.autotune(configs=AUTOTUNE_CONFIGS, key=['n_elements'])
    @triton.jit
    def _copy_kernel(src_ptr, dst_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        start = pid * BLOCK_SIZE
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_elements
        vals = tl.load(src_ptr + offs, mask=mask, other=0.0)
        tl.store(dst_ptr + offs, vals, mask=mask)

    def triton_copy(tensor: torch.Tensor) -> torch.Tensor:
        """
        Copy helper that avoids copies when unnecessary and uses Triton for large CUDA copies.
        - If tensor is on CPU: return a contiguous clone (preserve semantics).
        - If tensor is CUDA and already contiguous: return it directly (no copy, preserves autograd).
        - Otherwise, for sufficiently large tensors, use Triton kernel to copy; for smaller ones, use contiguous().clone().
        """
        assert isinstance(tensor, torch.Tensor)

        # CPU path: return a contiguous clone to preserve semantics and device.
        if not tensor.is_cuda:
            return tensor.contiguous().clone()

        # Fast-path: if already contiguous on CUDA, return directly (preserve autograd).
        if tensor.is_contiguous():
            return tensor

        # Determine size before any possible contiguous() to avoid double-copy.
        n_elements = tensor.numel()

        # Threshold (elements) below which torch.clone() is likely lower overhead than a kernel launch.
        # Keep conservative to avoid kernel launch overhead for small tensors (typical h_n here is small).
        TRITON_USE_THRESHOLD = 256 * 1024  # 256K elements (~1MB for fp32)

        # For small non-contiguous tensors, do a single contiguous().clone() (one copy).
        if n_elements < TRITON_USE_THRESHOLD:
            return tensor.contiguous().clone()

        # For large non-contiguous CUDA tensors, launch the Triton copy kernel.
        x = tensor.contiguous()
        out = torch.empty_like(x)
        # Use triton.cdiv to compute grid size robustly from the autotuned BLOCK_SIZE.
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
        _copy_kernel[grid](x, out, n_elements)
        return out

except Exception:
    # Triton not available -> simple fallback implementation
    def triton_copy(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.contiguous().clone()


class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.0):
        """
        Optimized LSTM wrapper.

        Key optimizations compared to the original:
        - Avoid the unused final fully-connected layer computation in forward.
        - Use a small Triton kernel to accelerate returning/copying the hidden state
          tensor when running on CUDA and the tensor is large enough.
        """
        super(ModelNew, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout, bidirectional=False)
        # Keep attribute for compatibility, but we won't compute it in forward (it was unused).
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, h0, c0):
        """
        Forward pass: run LSTM and return the hidden state h_n (state[0]).
        The previous model computed a final linear on the last time step but returned h_n;
        that linear computation is unnecessary and is omitted here to save time.
        """
        # Ensure inputs are on the same device and in float32 with a single copy each.
        device = x.device
        x = x.to(device=device, dtype=torch.float32)
        h0 = h0.to(device=device, dtype=torch.float32)
        c0 = c0.to(device=device, dtype=torch.float32)

        out, state = self.lstm(x, (h0, c0))  # state is (h_n, c_n)
        h_n = state[0]

        # Use Triton-accelerated copy for CUDA tensors when available; otherwise return a contiguous clone.
        return triton_copy(h_n)