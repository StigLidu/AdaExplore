import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

# Triton kernel for in-place bias add + ReLU over a (M x N) row-major matrix.
@triton.jit
def _bias_add_relu_inplace_kernel(x_ptr, bias_ptr, M, N, BLOCK: tl.constexpr, ROW_BLOCK: tl.constexpr):
    """
    In-place: for each row r and column c: x[r, c] = max(x[r, c] + bias[c], 0)
    Grid: ((ceil(M / ROW_BLOCK)), ceil(N / BLOCK))
    Each program handles ROW_BLOCK rows (ROW_BLOCK is a constexpr).
    """
    row_block_idx = tl.program_id(0)
    block_id = tl.program_id(1)

    col_start = block_id * BLOCK
    cols = col_start + tl.arange(0, BLOCK)
    mask = cols < N

    row_start = row_block_idx * ROW_BLOCK

    # Load bias tile once and reuse for rows in this row-block
    b = tl.load(bias_ptr + cols, mask=mask, other=0.0)

    # Iterate over the rows handled by this program
    for i in range(ROW_BLOCK):
        row = row_start + i
        if row < M:
            offs = row * N + cols
            x = tl.load(x_ptr + offs, mask=mask, other=0.0)
            y = x + b
            y = tl.where(y > 0.0, y, 0.0)
            tl.store(x_ptr + offs, y, mask=mask)


# Triton kernel for in-place bias add (no activation)
@triton.jit
def _bias_add_inplace_kernel(x_ptr, bias_ptr, M, N, BLOCK: tl.constexpr, ROW_BLOCK: tl.constexpr):
    row_block_idx = tl.program_id(0)
    block_id = tl.program_id(1)

    col_start = block_id * BLOCK
    cols = col_start + tl.arange(0, BLOCK)
    mask = cols < N

    row_start = row_block_idx * ROW_BLOCK

    # Load bias tile once and reuse for rows in this row-block
    b = tl.load(bias_ptr + cols, mask=mask, other=0.0)

    for i in range(ROW_BLOCK):
        row = row_start + i
        if row < M:
            offs = row * N + cols
            x = tl.load(x_ptr + offs, mask=mask, other=0.0)
            y = x + b
            tl.store(x_ptr + offs, y, mask=mask)


def triton_bias_add_relu_inplace(mat: torch.Tensor, bias: torch.Tensor, BLOCK: int = 256, ROW_BLOCK: int = 4):
    """
    In-place broadcast add of bias to each row of mat and apply ReLU, using Triton kernel.
    mat: (M, N) contiguous CUDA fp32
    bias: (N,) contiguous CUDA fp32
    BLOCK and ROW_BLOCK are constexpr tile sizes passed to the Triton kernel.
    """
    assert mat.is_cuda and bias.is_cuda, "Triton kernel requires CUDA tensors."
    assert mat.dtype == torch.float32 and bias.dtype == torch.float32, "Only fp32 supported."
    M, N = mat.shape
    if not mat.is_contiguous():
        mat = mat.contiguous()
    if not bias.is_contiguous():
        bias = bias.contiguous()
    grid = ((M + ROW_BLOCK - 1) // ROW_BLOCK, (N + BLOCK - 1) // BLOCK)
    _bias_add_relu_inplace_kernel[grid](mat, bias, M, N, BLOCK=BLOCK, ROW_BLOCK=ROW_BLOCK)
    return mat


def triton_bias_add_inplace(mat: torch.Tensor, bias: torch.Tensor, BLOCK: int = 256, ROW_BLOCK: int = 4):
    """
    In-place broadcast add of bias to each row of mat (no activation), using Triton kernel.
    """
    assert mat.is_cuda and bias.is_cuda, "Triton kernel requires CUDA tensors."
    assert mat.dtype == torch.float32 and bias.dtype == torch.float32, "Only fp32 supported."
    M, N = mat.shape
    if not mat.is_contiguous():
        mat = mat.contiguous()
    if not bias.is_contiguous():
        bias = bias.contiguous()
    grid = ((M + ROW_BLOCK - 1) // ROW_BLOCK, (N + BLOCK - 1) // BLOCK)
    _bias_add_inplace_kernel[grid](mat, bias, M, N, BLOCK=BLOCK, ROW_BLOCK=ROW_BLOCK)
    return mat


class ModelNew(nn.Module):
    """
    Optimized Model replacement for the original architecture.

    Key optimizations:
      - Cache device-resident, contiguous transposed weights (weight.T) to avoid repeated transposes.
      - Cache device-resident contiguous biases to avoid repeated .to() calls.
      - Preallocate a single workspace sized to the maximum layer width and reuse it across layers.
      - Use torch.matmul(..., out=...) to let cuBLAS write directly into workspace.
      - Fuse bias add + ReLU (hidden layers) and bias add (final layer) with Triton in-place kernels.
      - Adaptive Triton block size selection to reduce kernel launch overhead on very wide layers.
    """
    def __init__(self, input_size, layer_sizes, output_size):
        super(ModelNew, self).__init__()
        # Build layers explicitly (keeps same parameterization as original Model)
        self.hidden_linears = nn.ModuleList()
        current = input_size
        for sz in layer_sizes:
            lin = nn.Linear(current, sz)
            self.hidden_linears.append(lin)
            current = sz
        self.final_linear = nn.Linear(current, output_size)

        # Caches for device-resident tensors to avoid repeated transfers/transposes
        self._wt_cache = {}      # weight.data_ptr() -> (ptr, transposed_contiguous_on_device)
        self._bias_cache = {}    # bias.data_ptr() -> (ptr, contiguous_on_device)

        # Reusable workspaces to avoid repeated allocations
        self._workspace = None
        self._workspace_meta = None  # (device, dtype, batch, width)

        # Temporary buffer for matmul when input aliases the workspace slice
        self._matmul_tmp = None
        self._matmul_tmp_meta = None  # (device, dtype, batch, width)

    def _ensure_workspace(self, x: torch.Tensor, required_width: int):
        device = x.device
        dtype = x.dtype
        batch = x.shape[0]
        meta = (device, dtype, batch, required_width)
        if self._workspace is not None and self._workspace_meta == meta:
            return self._workspace
        # allocate workspace sized to (batch, required_width)
        self._workspace = torch.empty((batch, required_width), device=device, dtype=dtype)
        self._workspace_meta = meta
        return self._workspace

    def _ensure_matmul_tmp(self, x: torch.Tensor, required_width: int):
        device = x.device
        dtype = x.dtype
        batch = x.shape[0]
        meta = (device, dtype, batch, required_width)
        if self._matmul_tmp is not None and self._matmul_tmp_meta == meta:
            return self._matmul_tmp
        self._matmul_tmp = torch.empty((batch, required_width), device=device, dtype=dtype)
        self._matmul_tmp_meta = meta
        return self._matmul_tmp

    def _get_weight_t_on_device(self, weight: torch.nn.Parameter, device: torch.device):
        """
        Return a cached transposed contiguous copy of weight on `device`.
        Cache validated by weight.data_ptr() to reflect parameter updates.
        """
        key = int(weight.data_ptr())
        entry = self._wt_cache.get(key)
        current_ptr = int(weight.data_ptr())
        if entry is not None:
            cached_ptr, cached_wt = entry
            if cached_ptr == current_ptr and cached_wt.device == device:
                return cached_wt
        wt = weight.detach().t().contiguous().to(device)
        self._wt_cache[key] = (current_ptr, wt)
        return wt

    def _get_bias_on_device(self, bias: torch.nn.Parameter, device: torch.device):
        """
        Return a cached contiguous bias tensor on `device`.
        """
        if bias is None:
            return None
        key = int(bias.data_ptr())
        entry = self._bias_cache.get(key)
        current_ptr = int(bias.data_ptr())
        if entry is not None:
            cached_ptr, cached_b = entry
            if cached_ptr == current_ptr and cached_b.device == device:
                return cached_b
        b = bias.detach().contiguous().to(device)
        self._bias_cache[key] = (current_ptr, b)
        return b

    def _choose_block(self, N: int) -> int:
        """
        Adaptive BLOCK selection: prefer modest power-of-two tile widths for Ampere.
        Smaller BLOCK values (128/256) encourage good vectorization and lower register/shared memory pressure.
        """
        if N >= 8192:
            return 1024
        if N >= 4096:
            return 512
        if N >= 2048:
            return 256
        return 128

    def _choose_row_block(self, N: int) -> int:
        """
        Choose how many rows each Triton program handles; small tile of rows to increase
        work per program while avoiding excessive register pressure.
        """
        # Conservative default that works well across batch sizes on A6000.
        return 4

    def forward(self, x: torch.Tensor):
        out = x
        device = x.device
        dtype = x.dtype
        batch = x.shape[0]

        # compute maximum required width among all layers to size workspace once
        max_width = 0
        for lin in self.hidden_linears:
            max_width = max(max_width, lin.out_features)
        max_width = max(max_width, self.final_linear.out_features)

        workspace = self._ensure_workspace(x, max_width)

        # Hidden layers: matmul (cuBLAS) into workspace then fuse bias+ReLU with Triton inplace kernel
        for lin in self.hidden_linears:
            weight = lin.weight  # (out_f, in_f)
            bias = lin.bias
            out_features = weight.shape[0]

            # cached transposed weight (in_f, out_f) and bias on the correct device
            wt = self._get_weight_t_on_device(weight, device)
            b_dev = self._get_bias_on_device(bias, device)

            out_buf = workspace[:, :out_features]
            # Ensure out_buf is contiguous; workspace allocation ensures this, but be defensive
            if not out_buf.is_contiguous():
                out_buf = out_buf.contiguous()

            # Avoid aliasing: if out and out_buf share same data ptr, write into temporary then copy
            with torch.no_grad():
                if int(out.data_ptr()) == int(out_buf.data_ptr()):
                    tmp = self._ensure_matmul_tmp(out, out_features)[:, :out_features]
                    torch.matmul(out, wt, out=tmp)
                    out_buf.copy_(tmp)
                else:
                    torch.matmul(out, wt, out=out_buf)

            # Fuse bias add + ReLU using Triton when on CUDA
            if out_buf.is_cuda and b_dev is not None and b_dev.device == device and out_buf.dtype == torch.float32:
                BLOCK = self._choose_block(out_features)
                ROW_BLOCK = self._choose_row_block(out_features)
                # ensure bias is contiguous on device (cached)
                triton_bias_add_relu_inplace(out_buf, b_dev, BLOCK=BLOCK, ROW_BLOCK=ROW_BLOCK)
                out = out_buf
            else:
                if b_dev is not None:
                    out_buf = out_buf + b_dev.to(out_buf.device)
                out = F.relu(out_buf)

        # Final linear (no ReLU): matmul then fuse bias add (no activation)
        final_w = self.final_linear.weight
        final_b = self.final_linear.bias
        final_out_features = final_w.shape[0]

        wt_final = self._get_weight_t_on_device(final_w, device)
        b_final_dev = self._get_bias_on_device(final_b, device)

        out_buf = workspace[:, :final_out_features]
        if not out_buf.is_contiguous():
            out_buf = out_buf.contiguous()

        with torch.no_grad():
            if int(out.data_ptr()) == int(out_buf.data_ptr()):
                tmp = self._ensure_matmul_tmp(out, final_out_features)[:, :final_out_features]
                torch.matmul(out, wt_final, out=tmp)
                out_buf.copy_(tmp)
            else:
                torch.matmul(out, wt_final, out=out_buf)

        if out_buf.is_cuda and b_final_dev is not None and b_final_dev.device == device and out_buf.dtype == torch.float32:
            BLOCK = self._choose_block(final_out_features)
            ROW_BLOCK = self._choose_row_block(final_out_features)
            triton_bias_add_inplace(out_buf, b_final_dev, BLOCK=BLOCK, ROW_BLOCK=ROW_BLOCK)
            return out_buf
        else:
            if b_final_dev is not None:
                out_buf = out_buf + b_final_dev.to(out_buf.device)
            return out_buf