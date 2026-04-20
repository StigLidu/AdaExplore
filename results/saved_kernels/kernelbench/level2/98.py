import torch
import torch.nn as nn
import triton
import triton.language as tl

# Autotune different BLOCK and ROWS configurations to find the best-performing kernel
# Tailored for NVIDIA A6000 (Ampere): favor larger tiles and multi-row workloads.
# Keep num_warps as powers of two and narrow search to hardware-friendly candidates.
# Added larger BLOCK sizes to amortize the cost of tl.exp and reduce loop iterations.
AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK": 256, "ROWS": 4}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK": 512, "ROWS": 4}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK": 256, "ROWS": 8}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK": 512, "ROWS": 8}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK": 256, "ROWS": 8}, num_warps=16, num_stages=3),
    triton.Config({"BLOCK": 512, "ROWS": 8}, num_warps=16, num_stages=3),
    # Larger blocks to amortize expensive elementwise ops (e.g., exp)
    triton.Config({"BLOCK": 1024, "ROWS": 2}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK": 1024, "ROWS": 4}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK": 1024, "ROWS": 8}, num_warps=16, num_stages=3),
    triton.Config({"BLOCK": 2048, "ROWS": 2}, num_warps=16, num_stages=3),
    triton.Config({"BLOCK": 2048, "ROWS": 4}, num_warps=16, num_stages=3),
    triton.Config({"BLOCK": 2048, "ROWS": 8}, num_warps=16, num_stages=4),
]


@triton.autotune(configs=AUTOTUNE_CONFIGS, key=["M","N_COLS"])
@triton.jit
def _gelu_scaled_row_max_kernel(
    input_ptr,      # pointer to input matrix (row-major) -- expected fp16 memory
    output_ptr,     # pointer to output vector (one max per row) -- fp32 memory
    N_COLS,         # number of columns in the input (pooled length)
    scale,          # scaling factor (float)
    M,              # number of rows (batch size)
    BLOCK: tl.constexpr,  # columns per block (constexpr)
    ROWS: tl.constexpr,   # rows handled per program (constexpr)
):
    # Each program handles a contiguous block of ROWS rows
    row_block = tl.program_id(0)
    row_start = row_block * ROWS

    offs = tl.arange(0, BLOCK)  # column offsets inside a block
    neg_inf = -1e30

    # Initialize accumulators per row handled by this program (accumulate in fp32)
    acc0 = neg_inf
    acc1 = neg_inf
    acc2 = neg_inf
    acc3 = neg_inf
    acc4 = neg_inf
    acc5 = neg_inf
    acc6 = neg_inf
    acc7 = neg_inf

    # Iterate over columns in chunks
    for col_start in range(0, N_COLS, BLOCK):
        cols = col_start + offs
        mask = cols < N_COLS

        # For each row handled by this program, load (fp16), cast to fp32 and process its block
        if ROWS >= 1:
            r0 = row_start + 0
            ptr0 = input_ptr + r0 * N_COLS + cols
            vals0 = tl.load(ptr0, mask=mask, other=-1e30)         # loaded in memory dtype (fp16)
            vals0 = tl.cast(vals0, tl.float32)
            # GELU approx: x * sigmoid(1.702 * x)
            sig0 = 1.0 / (1.0 + tl.exp(-1.702 * vals0))
            gated0 = vals0 * sig0 * scale
            local_max0 = tl.max(gated0)
            acc0 = tl.maximum(acc0, local_max0)

        if ROWS >= 2:
            r1 = row_start + 1
            ptr1 = input_ptr + r1 * N_COLS + cols
            vals1 = tl.load(ptr1, mask=mask, other=-1e30)
            vals1 = tl.cast(vals1, tl.float32)
            sig1 = 1.0 / (1.0 + tl.exp(-1.702 * vals1))
            gated1 = vals1 * sig1 * scale
            local_max1 = tl.max(gated1)
            acc1 = tl.maximum(acc1, local_max1)

        if ROWS >= 3:
            r2 = row_start + 2
            ptr2 = input_ptr + r2 * N_COLS + cols
            vals2 = tl.load(ptr2, mask=mask, other=-1e30)
            vals2 = tl.cast(vals2, tl.float32)
            sig2 = 1.0 / (1.0 + tl.exp(-1.702 * vals2))
            gated2 = vals2 * sig2 * scale
            local_max2 = tl.max(gated2)
            acc2 = tl.maximum(acc2, local_max2)

        if ROWS >= 4:
            r3 = row_start + 3
            ptr3 = input_ptr + r3 * N_COLS + cols
            vals3 = tl.load(ptr3, mask=mask, other=-1e30)
            vals3 = tl.cast(vals3, tl.float32)
            sig3 = 1.0 / (1.0 + tl.exp(-1.702 * vals3))
            gated3 = vals3 * sig3 * scale
            local_max3 = tl.max(gated3)
            acc3 = tl.maximum(acc3, local_max3)

        if ROWS >= 5:
            r4 = row_start + 4
            ptr4 = input_ptr + r4 * N_COLS + cols
            vals4 = tl.load(ptr4, mask=mask, other=-1e30)
            vals4 = tl.cast(vals4, tl.float32)
            sig4 = 1.0 / (1.0 + tl.exp(-1.702 * vals4))
            gated4 = vals4 * sig4 * scale
            local_max4 = tl.max(gated4)
            acc4 = tl.maximum(acc4, local_max4)

        if ROWS >= 6:
            r5 = row_start + 5
            ptr5 = input_ptr + r5 * N_COLS + cols
            vals5 = tl.load(ptr5, mask=mask, other=-1e30)
            vals5 = tl.cast(vals5, tl.float32)
            sig5 = 1.0 / (1.0 + tl.exp(-1.702 * vals5))
            gated5 = vals5 * sig5 * scale
            local_max5 = tl.max(gated5)
            acc5 = tl.maximum(acc5, local_max5)

        if ROWS >= 7:
            r6 = row_start + 6
            ptr6 = input_ptr + r6 * N_COLS + cols
            vals6 = tl.load(ptr6, mask=mask, other=-1e30)
            vals6 = tl.cast(vals6, tl.float32)
            sig6 = 1.0 / (1.0 + tl.exp(-1.702 * vals6))
            gated6 = vals6 * sig6 * scale
            local_max6 = tl.max(gated6)
            acc6 = tl.maximum(acc6, local_max6)

        if ROWS >= 8:
            r7 = row_start + 7
            ptr7 = input_ptr + r7 * N_COLS + cols
            vals7 = tl.load(ptr7, mask=mask, other=-1e30)
            vals7 = tl.cast(vals7, tl.float32)
            sig7 = 1.0 / (1.0 + tl.exp(-1.702 * vals7))
            gated7 = vals7 * sig7 * scale
            local_max7 = tl.max(gated7)
            acc7 = tl.maximum(acc7, local_max7)

    # Store results back to global memory for rows that exist (outputs are fp32)
    if ROWS >= 1:
        idx0 = row_start + 0
        if idx0 < M:
            tl.store(output_ptr + idx0, acc0)
    if ROWS >= 2:
        idx1 = row_start + 1
        if idx1 < M:
            tl.store(output_ptr + idx1, acc1)
    if ROWS >= 3:
        idx2 = row_start + 2
        if idx2 < M:
            tl.store(output_ptr + idx2, acc2)
    if ROWS >= 4:
        idx3 = row_start + 3
        if idx3 < M:
            tl.store(output_ptr + idx3, acc3)
    if ROWS >= 5:
        idx4 = row_start + 4
        if idx4 < M:
            tl.store(output_ptr + idx4, acc4)
    if ROWS >= 6:
        idx5 = row_start + 5
        if idx5 < M:
            tl.store(output_ptr + idx5, acc5)
    if ROWS >= 7:
        idx6 = row_start + 6
        if idx6 < M:
            tl.store(output_ptr + idx6, acc6)
    if ROWS >= 8:
        idx7 = row_start + 7
        if idx7 < M:
            tl.store(output_ptr + idx7, acc7)


def triton_gelu_scaled_row_max(x: torch.Tensor, scale: float):
    """
    Wrapper launching the autotuned Triton kernel that computes per-row:
      max_j( GELU(x[row, j]) * scale )
    Uses mixed precision: cast the input to fp16 to reduce bandwidth, while the kernel
    casts tiles back to fp32 for computation/accumulation and writes fp32 outputs.
    """
    assert x.is_cuda, "Input must be on CUDA"
    x = x.contiguous()
    M, N = x.shape
    # output in fp32 to preserve numeric stability
    out = torch.empty((M,), dtype=torch.float32, device=x.device)

    # Cast input to fp16 to reduce memory traffic for the kernel
    x_half = x.half()

    # Grid depends on the autotuned ROWS meta parameter
    grid = lambda meta: ((M + meta["ROWS"] - 1) // meta["ROWS"],)

    # Launch kernel; autotuner will provide BLOCK and ROWS through meta
    _gelu_scaled_row_max_kernel[grid](x_half, out, N, float(scale), M)
    return out


class ModelNew(nn.Module):
    """
    Optimized Model:
      - Fuses the original Linear + AvgPool (over contiguous groups of outputs) into a
        smaller Linear by averaging groups of output weights/biases. This reduces the
        linear output dimension by pool_kernel_size.
      - Applies a highly-tuned Triton kernel to compute GELU (approx) + scale + row-wise max
        in a single fused pass with multi-row processing and autotuned block sizes.
    """
    def __init__(self, in_features, out_features, pool_kernel_size, scale_factor):
        super(ModelNew, self).__init__()
        assert out_features % pool_kernel_size == 0, "out_features must be divisible by pool_kernel_size"
        pooled_out = out_features // pool_kernel_size
        self.pool_kernel_size = pool_kernel_size
        self.scale_factor = float(scale_factor)

        # Initialize a temporary linear to obtain sensible defaults, then pool weights
        tmp = nn.Linear(in_features, out_features, bias=True)
        with torch.no_grad():
            # tmp.weight: (out_features, in_features)
            # Reshape to (pooled_out, pool_kernel_size, in_features) and average over the group dim
            w = tmp.weight.view(pooled_out, pool_kernel_size, in_features).mean(dim=1).contiguous()
            b = tmp.bias.view(pooled_out, pool_kernel_size).mean(dim=1).contiguous()

        # Register pooled parameters in fp16 to enable fp16 GEMM (Tensor Cores) and reduce bandwidth
        self.weight = nn.Parameter(w.half())   # shape: (pooled_out, in_features) in fp16
        self.bias = nn.Parameter(b.half())     # shape: (pooled_out,) in fp16

        # cleanup
        del tmp

    def forward(self, x: torch.Tensor):
        """
        x: (batch_size, in_features)
        returns: (batch_size,) -> max across pooled positions after GELU and scaling
        """
        # Ensure params on same device
        if x.device != self.weight.device:
            self.to(x.device)

        x = x.contiguous()
        # Compute linear with pooled weights in fp16 to leverage Tensor Cores -> (batch, pooled_out) in fp16
        y = torch.addmm(self.bias.unsqueeze(0).half(), x.half(), self.weight.t().half())
        # Use the Triton kernel for fused GELU (approx) + scale + row-wise max
        out = triton_gelu_scaled_row_max(y, self.scale_factor)
        return out