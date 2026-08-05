import torch
import torch.nn as nn
import triton
import triton.language as tl

# Autotune configurations tuned for A6000 / Ampere-like
GEMM_CONFIGS = [
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 32}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_M": 256, "BLOCK_N": 256, "BLOCK_K": 64}, num_warps=8, num_stages=3),
]

REDUCE_CONFIGS = [
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 256, "BLOCK_N": 128}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_M": 512, "BLOCK_N": 128}, num_warps=8, num_stages=3),
]


@triton.autotune(
    configs=GEMM_CONFIGS,
    key=['M', 'N', 'K'],
)
@triton.jit
def _fused_gemm_scale_kernel(
    A_ptr,        # pointer to A (M x K)
    W_ptr,        # pointer to Wt (K x N)  <-- weight transposed before kernel
    bias_ptr,     # pointer to bias (N,) or 0
    scale_ptr,    # pointer to scale (N,)
    C_ptr,        # pointer to output C (M x N)
    M, N, K,
    stride_am, stride_ak,   # A strides: stride along M (rows), stride along K (cols)
    stride_wk, stride_wn,   # Wt strides: stride along K (rows), stride along N (cols)
    stride_cm, stride_cn,   # C strides
    has_bias: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # program ids for tiling
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    row_start = pid_m * BLOCK_M
    col_start = pid_n * BLOCK_N

    # ranges
    row_offsets = row_start + tl.arange(0, BLOCK_M)
    col_offsets = col_start + tl.arange(0, BLOCK_N)
    k_offsets = tl.arange(0, BLOCK_K)

    # compute address bases
    # shapes:
    # A: (M, K) with strides stride_am, stride_ak
    # Wt: (K, N) with strides stride_wk, stride_wn
    # C: (M, N) with strides stride_cm, stride_cn

    # initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # iterate over K dimension tiles
    for k_start in range(0, K, BLOCK_K):
        k = k_start + k_offsets  # shape [BLOCK_K]
        # masks for bounds
        mask_a = (row_offsets[:, None] < M) & (k[None, :] < K)
        mask_w = (k[:, None] < K) & (col_offsets[None, :] < N)

        # compute addresses
        a_addr = A_ptr + (row_offsets[:, None] * stride_am) + (k[None, :] * stride_ak)
        w_addr = W_ptr + (k[:, None] * stride_wk) + (col_offsets[None, :] * stride_wn)

        a = tl.load(a_addr, mask=mask_a, other=0.0)
        w = tl.load(w_addr, mask=mask_w, other=0.0)  # shape (BLOCK_K, BLOCK_N)

        # a: (BLOCK_M, BLOCK_K), w: (BLOCK_K, BLOCK_N) -> dot -> (BLOCK_M, BLOCK_N)
        acc += tl.dot(a, w)

    # after accumulation, add bias and scale
    col_mask = col_offsets < N
    row_mask = row_offsets < M
    store_mask = row_mask[:, None] & col_mask[None, :]

    if has_bias:
        bias_addr = bias_ptr + col_offsets
        bias_vals = tl.load(bias_addr, mask=col_mask, other=0.0)  # (BLOCK_N,)
        acc = acc + bias_vals[None, :]

    # apply scale per output channel
    scale_addr = scale_ptr + col_offsets
    scale_vals = tl.load(scale_addr, mask=col_mask, other=1.0)
    acc = acc * scale_vals[None, :]

    # store
    c_addr = C_ptr + (row_offsets[:, None] * stride_cm) + (col_offsets[None, :] * stride_cn)
    tl.store(c_addr, acc, mask=store_mask)


@triton.autotune(
    configs=REDUCE_CONFIGS,
    key=['M', 'N'],
)
@triton.jit
def _reduce_mean_var_kernel(
    X_ptr,      # pointer to X (M x N)
    mean_ptr,   # pointer to mean (N,)
    sq_ptr,     # pointer to sum_of_squares (N,)
    M, N,
    stride_xm, stride_xn,
    stride_mean, stride_sq,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    # reduce over rows (M) for each column block (N)
    pid_n = tl.program_id(0)
    col_start = pid_n * BLOCK_N
    col_offsets = col_start + tl.arange(0, BLOCK_N)

    m_start = tl.program_id(1) * BLOCK_M
    m_offsets = m_start + tl.arange(0, BLOCK_M)

    # initialize accumulators
    sum_acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    sumsq_acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    # iterate over row tiles
    # each program handles a tile of rows (BLOCK_M) at given m_start
    # but we launch over multiple m tiles; we'll perform reduction across all m tiles in the grid
    # To do a full reduction, we will require grid to cover all m tiles; we will use atomic adds by launching
    # a reduction in two phases: 1) this kernel sums across its local rows and writes to temporary buffers
    # But Triton doesn't have native atomics for float reliably across SMs in all configs; instead,
    # we'll structure the kernel so that program_id(1) sweeps across M in steps of BLOCK_M and accumulates
    # the full column sums in this single program. For simplicity, we assume grid is (ceil(N/BLOCK_N),)
    # and loop over rows here.
    # However, to be robust, we will sum across all rows inside this kernel by looping over m in python style.

    # Instead of relying on multiple program_ids over m, we just loop across all m ranges here:
    # NOTE: This is potentially less parallel but simplifies correctness.
    for m in range(0, M, BLOCK_M):
        m_range = m + tl.arange(0, BLOCK_M)
        mask_m = m_range < M
        # compute addresses: X[m_range, col_offsets] -> shape (BLOCK_M, BLOCK_N)
        x_addr = X_ptr + (m_range[:, None] * stride_xm) + (col_offsets[None, :] * stride_xn)
        mask = mask_m[:, None] & (col_offsets[None, :] < N)
        x = tl.load(x_addr, mask=mask, other=0.0)  # shape (BLOCK_M, BLOCK_N)
        # sum across rows
        sum_acc += tl.sum(x, axis=0)
        sumsq_acc += tl.sum(x * x, axis=0)

    col_mask = col_offsets < N
    # write results to mean_ptr and sq_ptr (these store sum and sumsq, will be normalized outside)
    write_mean_addr = mean_ptr + col_offsets
    write_sq_addr = sq_ptr + col_offsets
    tl.store(write_mean_addr, sum_acc, mask=col_mask)
    tl.store(write_sq_addr, sumsq_acc, mask=col_mask)


@triton.jit
def _apply_batchnorm_kernel(
    X_ptr,
    mean_ptr,
    invstd_ptr,
    gamma_ptr,
    beta_ptr,
    M, N,
    stride_xm, stride_xn,
    stride_mean, stride_invstd,
    stride_gamma, stride_beta,
    stride_outm, stride_outn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    row_start = pid_m * BLOCK_M
    col_start = pid_n * BLOCK_N

    row_offsets = row_start + tl.arange(0, BLOCK_M)
    col_offsets = col_start + tl.arange(0, BLOCK_N)

    m_mask = row_offsets < M
    n_mask = col_offsets < N
    store_mask = m_mask[:, None] & n_mask[None, :]

    x_addr = X_ptr + (row_offsets[:, None] * stride_xm) + (col_offsets[None, :] * stride_xn)
    x = tl.load(x_addr, mask=store_mask, other=0.0)

    mean = tl.load(mean_ptr + col_offsets, mask=n_mask, other=0.0)  # (BLOCK_N,)
    invstd = tl.load(invstd_ptr + col_offsets, mask=n_mask, other=1.0)
    gamma = tl.load(gamma_ptr + col_offsets, mask=n_mask, other=1.0)
    beta = tl.load(beta_ptr + col_offsets, mask=n_mask, other=0.0)

    # broadcast and apply: (x - mean) * invstd * gamma + beta
    out = (x - mean[None, :]) * invstd[None, :] * gamma[None, :] + beta[None, :]

    out_addr = X_ptr + (row_offsets[:, None] * stride_outm) + (col_offsets[None, :] * stride_outn)
    tl.store(out_addr, out, mask=store_mask)


def triton_fused_gemm_scale(x: torch.Tensor, weight_t: torch.Tensor, bias: torch.Tensor, scale: torch.Tensor):
    """
    Performs C = (x @ weight_t) + bias  (weight_t is KxN), then C *= scale (per-column).
    x: (M, K)
    weight_t: (K, N)
    bias: (N,) or None
    scale: (N,)
    """
    assert x.is_cuda and weight_t.is_cuda and scale.is_cuda
    M, K = x.shape
    K_w, N = weight_t.shape
    assert K == K_w
    out = torch.empty((M, N), device=x.device, dtype=x.dtype)

    # ensure contiguous
    x_ = x.contiguous()
    w_ = weight_t.contiguous()
    scale_ = scale.contiguous()
    if bias is None:
        bias_ptr = torch.empty((0,), device=x.device, dtype=x.dtype)
        has_bias = 0
    else:
        bias_ = bias.contiguous()
        bias_ptr = bias_
        has_bias = 1

    # strides (elements)
    stride_am = x_.stride(0)  # step to next row in elements
    stride_ak = x_.stride(1)
    stride_wk = w_.stride(0)
    stride_wn = w_.stride(1)
    stride_cm = out.stride(0)
    stride_cn = out.stride(1)

    # Convert strides from bytes to elements
    # In PyTorch strides are in elements already.
    # Launch kernel
    grid = lambda meta: (
        (M + meta['BLOCK_M'] - 1) // meta['BLOCK_M'],
        (N + meta['BLOCK_N'] - 1) // meta['BLOCK_N'],
    )

    _fused_gemm_scale_kernel[grid](
        x_, w_, bias_ptr if has_bias else x_.new_empty(0), scale_, out,
        M, N, K,
        stride_am, stride_ak,
        stride_wk, stride_wn,
        stride_cm, stride_cn,
        has_bias,
    )
    return out


def triton_mean_var(x: torch.Tensor):
    """
    Compute per-column sum and sum of squares across rows using Triton kernel,
    then return mean and var (unbiased population var E[x^2] - mean^2).
    x: (M, N)
    returns mean (N,), var (N,)
    """
    M, N = x.shape
    # temporary tensors to store sums
    sum_tensor = torch.empty((N,), device=x.device, dtype=torch.float32)
    sumsq_tensor = torch.empty((N,), device=x.device, dtype=torch.float32)

    # strides
    stride_xm = x.stride(0)
    stride_xn = x.stride(1)
    stride_mean = sum_tensor.stride(0)
    stride_sq = sumsq_tensor.stride(0)

    # grid choose over columns only (kernel loops over rows)
    # grid for _reduce_mean_var_kernel: (ceil(N/BLOCK_N),)
    grid = lambda meta: ((N + meta['BLOCK_N'] - 1) // meta['BLOCK_N'], 1)

    _reduce_mean_var_kernel[grid](
        x, sum_tensor, sumsq_tensor,
        M, N,
        stride_xm, stride_xn,
        stride_mean, stride_sq,
    )

    # Convert sums to mean and variance (population)
    mean = sum_tensor / float(M)
    ex2 = sumsq_tensor / float(M)
    var = ex2 - mean * mean
    # clamp small negative numerical values to zero
    var = torch.clamp(var, min=0.0)
    return mean, var


def triton_apply_batchnorm(x: torch.Tensor, mean: torch.Tensor, invstd: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor):
    """
    In-place application of batchnorm on x:
    x = (x - mean) * invstd * gamma + beta
    """
    M, N = x.shape
    # ensure contiguity
    x_ = x.contiguous()
    mean_ = mean.contiguous()
    invstd_ = invstd.contiguous()
    gamma_ = gamma.contiguous()
    beta_ = beta.contiguous()

    stride_xm = x_.stride(0)
    stride_xn = x_.stride(1)
    stride_mean = mean_.stride(0)
    stride_invstd = invstd_.stride(0)
    stride_gamma = gamma_.stride(0)
    stride_beta = beta_.stride(0)
    stride_outm = stride_xm
    stride_outn = stride_xn

    # Choose blocks
    BLOCK_M = 128
    BLOCK_N = 128
    grid = ((M + BLOCK_M - 1) // BLOCK_M, (N + BLOCK_N - 1) // BLOCK_N)
    _apply_batchnorm_kernel[grid](
        x_, mean_, invstd_, gamma_, beta_,
        M, N,
        stride_xm, stride_xn,
        stride_mean, stride_invstd,
        stride_gamma, stride_beta,
        stride_outm, stride_outn,
        BLOCK_M, BLOCK_N
    )
    return x_


class ModelNew(nn.Module):
    """
    Optimized model that fuses GEMM + scale + batchnorm using Triton kernels.
    NOTE: This ModelNew performs the forward-pass using custom Triton kernels.
    It uses the stored BatchNorm running_mean and running_var for inference.
    If bn.training == True, it will compute batch statistics and update running stats
    in the standard PyTorch way (on the CPU/GPU tensors).
    """
    def __init__(self, in_features, out_features, scale_shape, eps=1e-5, momentum=0.1):
        super(ModelNew, self).__init__()
        # keep the same modules/parameters so state_dict is compatible
        self.gemm = nn.Linear(in_features, out_features)
        self.scale = nn.Parameter(torch.randn(scale_shape))
        self.bn = nn.BatchNorm1d(out_features, eps=eps, momentum=momentum)
        # ensure parameters are float32
        self.eps = eps
        self.momentum = momentum

    def forward(self, x):
        # x: (B, in_features)
        assert x.is_cuda, "Inputs must be on CUDA."
        # Prepare weight transposed for our kernel: weight_t shape (K, N) where K=in_features, N=out_features
        weight = self.gemm.weight  # shape (out_features, in_features)
        # transpose to (in_features, out_features)
        weight_t = weight.t().contiguous()

        bias = self.gemm.bias if self.gemm.bias is not None else None
        scale = self.scale

        x = x.contiguous().to(torch.float32)

        # Stage 1: fused GEMM + scale
        out = triton_fused_gemm_scale(x, weight_t, bias, scale)

        # Stage 2: compute mean and variance across batch (per-channel)
        if self.bn.training:
            # compute batch stats
            mean, var = triton_mean_var(out)
            # update running stats in-place (same behavior as PyTorch)
            with torch.no_grad():
                self.bn.running_mean = (1 - self.momentum) * self.bn.running_mean + self.momentum * mean
                self.bn.running_var = (1 - self.momentum) * self.bn.running_var + self.momentum * var
            used_mean = mean
            used_var = var
        else:
            # use stored running stats (inference)
            used_mean = self.bn.running_mean.to(out.device).to(out.dtype)
            used_var = self.bn.running_var.to(out.device).to(out.dtype)

        # compute invstd
        invstd = 1.0 / torch.sqrt(used_var + self.eps)

        # Stage 3: apply batchnorm affine
        gamma = self.bn.weight if self.bn.weight is not None else torch.ones_like(used_mean)
        beta = self.bn.bias if self.bn.bias is not None else torch.zeros_like(used_mean)
        # ensure on correct device/dtype
        gamma = gamma.to(out.device).to(out.dtype)
        beta = beta.to(out.device).to(out.dtype)
        used_mean = used_mean.to(out.device).to(out.dtype)
        invstd = invstd.to(out.device).to(out.dtype)

        out = triton_apply_batchnorm(out, used_mean, invstd, gamma, beta)
        return out

# Provide the same helper functions to generate inputs as original module expects
batch_size = 1024
in_features = 8192
out_features = 8192
scale_shape = (out_features,)

def get_inputs():
    return [torch.rand(batch_size, in_features).cuda()]

def get_init_inputs():
    return [in_features, out_features, scale_shape]