import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

# Autotune configs for the 2D tiled scaling kernel (broadcast-aware)
# Favor very wide contiguous BLOCK_N (2048/4096) and small BLOCK_M (16/32/64) for Ampere.
AUTOTUNE_CONFIGS_2D = [
    triton.Config({"BLOCK_M": 16, "BLOCK_N": 4096}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_M": 32, "BLOCK_N": 4096}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 2048}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_M": 32, "BLOCK_N": 2048}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_M": 16, "BLOCK_N": 2048}, num_warps=8, num_stages=2),
]


@triton.autotune(
    configs=AUTOTUNE_CONFIGS_2D,
    key=["M", "N"],
)
@triton.jit
def _scale_2d_kernel(x_ptr, out_ptr, scale_ptr, M, N, stride_row, stride_col,
                     BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    """
    2D tiled scaling kernel (per-row load/store):
      out = x * scale (scale is length-N, broadcasted over rows)
    Each program handles a tile of shape (BLOCK_M, BLOCK_N).
    This implementation:
      - Loads the column block of 'scale' once.
      - For each row in the tile computes a base pointer and performs a single vectorized load/store.
    This avoids building a full (BLOCK_M x BLOCK_N) offset matrix and reduces register/shared pressure.
    stride_row / stride_col are element strides (in elements, not bytes).
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    row_start = pid_m * BLOCK_M
    col_start = pid_n * BLOCK_N

    rows = row_start + tl.arange(0, BLOCK_M)        # shape (BLOCK_M,)
    cols = col_start + tl.arange(0, BLOCK_N)       # shape (BLOCK_N,)

    row_mask = rows < M                              # shape (BLOCK_M,)
    col_mask = cols < N                              # shape (BLOCK_N,)

    # Load column block of scale once (broadcast over rows)
    scale_vals = tl.load(scale_ptr + cols, mask=col_mask, other=1.0)  # shape (BLOCK_N,)

    # For each row in the tile, perform a contiguous vector load/store across cols.
    # Keep memory accesses coalesced when stride_col == 1.
    for i in range(BLOCK_M):
        row = rows[i]                              # scalar tl.int32
        # per-row mask for columns (broadcast scalar valid-row over columns)
        mask_cols = col_mask & (row < M)          # shape (BLOCK_N,)
        base = x_ptr + row * stride_row + cols * stride_col  # per-column pointers
        x_row = tl.load(base, mask=mask_cols, other=0.0)     # shape (BLOCK_N,)
        y_row = x_row * scale_vals
        tl.store(out_ptr + row * stride_row + cols * stride_col, y_row, mask=mask_cols)


def triton_scale_2d(x: torch.Tensor, scale: torch.Tensor):
    """
    Wrapper for the Triton 2D scaling kernel.
    x: (M, N) tensor (batch, features)
    scale: scalar or tensor broadcastable to length N (per-feature)
    Returns a new tensor with elementwise scaling applied.
    """
    assert x.is_cuda, "Input must be on CUDA."

    x_contig = x.contiguous()
    M, N = x_contig.shape
    out = torch.empty_like(x_contig)

    # Prepare scale vector on device and contiguous
    device = x_contig.device
    dtype = x_contig.dtype
    if scale.numel() == 1:
        # expand scalar to full feature vector to reuse same kernel path
        scale_vec = torch.full((N,), float(scale.item()), device=device, dtype=dtype)
    else:
        # move scale to device and make it a 1-D vector of length N (broadcast/reshape if necessary)
        scale_dev = scale.to(device=device, dtype=dtype)
        if scale_dev.dim() == 1 and scale_dev.numel() == N:
            scale_vec = scale_dev.contiguous()
        else:
            # try to broadcast into shape (N,)
            try:
                scale_vec = torch.broadcast_tensors(scale_dev, torch.empty(N, device=device, dtype=dtype))[0].view(-1).contiguous()
            except Exception:
                # fallback: attempt view/expand
                scale_vec = scale_dev.contiguous().view(-1).expand(N).contiguous()

    # element strides (in elements)
    stride_row, stride_col = x_contig.stride()

    grid = lambda meta: ((M + meta["BLOCK_M"] - 1) // meta["BLOCK_M"],
                         (N + meta["BLOCK_N"] - 1) // meta["BLOCK_N"])

    _scale_2d_kernel[grid](x_contig, out, scale_vec, M, N, stride_row, stride_col)
    return out


class ModelNew(nn.Module):
    """
    Optimized model:
      - Uses standard nn.Linear and BatchNorm.
      - In eval mode, folds BN + optional scale into the Linear weights and bias on-device and
        performs a single F.linear call (fast, cuBLAS-backed GEMM).
      - In training or unfusable cases, folds scale into BN.weight when possible or uses PyTorch
        elementwise multiplication (avoids launching tiny Triton kernels).
      - Uses torch.softmax for the final softmax.
    """
    def __init__(self, in_features, out_features, bn_eps=1e-5, bn_momentum=0.1, scale_shape=(1,)):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features, eps=bn_eps, momentum=bn_momentum)
        # scale is a learnable parameter shaped as given (default (1,))
        self.scale = nn.Parameter(torch.ones(scale_shape))

        # Cache for fused weights/bias in eval mode to avoid recomputing gigantic tensors every forward.
        # Keys: data_ptrs are used to detect parameter changes.
        self._fused_cache = {
            "W_ptr": None,
            "bn_weight_ptr": None,
            "running_var_ptr": None,
            "scale_ptr": None,
            "W_fused": None,
            "b_fused": None
        }

    def forward(self, x: torch.Tensor):
        """
        Forward pass:
          x: (batch_size, in_features) fp32 CUDA tensor
        Returns:
          (batch_size, out_features) fp32 CUDA tensor after linear -> BN -> scale -> softmax
        """
        # Ensure linear params/bn params live on the same device for fusion ops
        W = self.gemm.weight
        device = W.device
        dtype = W.dtype
        # Linear bias (may be None)
        b_lin = self.gemm.bias if self.gemm.bias is not None else torch.zeros(W.shape[0], device=device, dtype=dtype)

        # BN params (on device)
        bn_weight = self.bn.weight.to(device)
        bn_bias = self.bn.bias.to(device)
        running_mean = self.bn.running_mean.to(device)
        running_var = self.bn.running_var.to(device)
        eps = self.bn.eps
        momentum = self.bn.momentum

        scale = self.scale.to(device)

        # If we switched into training, clear any cached fused weights to avoid stale/memory issues.
        if self.training and getattr(self, "_fused_cache", None) and self._fused_cache.get("W_fused") is not None:
            self._fused_cache = {
                "W_ptr": None,
                "bn_weight_ptr": None,
                "running_var_ptr": None,
                "scale_ptr": None,
                "W_fused": None,
                "b_fused": None
            }

        # If in eval mode, fold BN and scale into the linear weight and bias and do a single F.linear.
        # Use a small cache so we don't recompute the huge W_fused every forward when params haven't changed.
        if not self.training:
            # pointers to detect changes
            W_ptr = W.data_ptr()
            bw_ptr = bn_weight.data_ptr()
            rv_ptr = running_var.data_ptr()
            sc_ptr = scale.data_ptr() if isinstance(scale, torch.Tensor) else None

            cache = getattr(self, "_fused_cache", None)
            if cache is None:
                self._fused_cache = {
                    "W_ptr": None,
                    "bn_weight_ptr": None,
                    "running_var_ptr": None,
                    "scale_ptr": None,
                    "W_fused": None,
                    "b_fused": None
                }
                cache = self._fused_cache

            need_recompute = (
                cache.get("W_fused") is None
                or cache.get("W_ptr") != W_ptr
                or cache.get("bn_weight_ptr") != bw_ptr
                or cache.get("running_var_ptr") != rv_ptr
                or cache.get("scale_ptr") != sc_ptr
            )

            if need_recompute:
                # inv = gamma / sqrt(running_var + eps)
                inv = bn_weight / torch.sqrt(running_var + eps)

                # Fold post-BN scale if present (scalar or per-channel)
                if scale.numel() == 1:
                    inv = inv * float(scale)
                elif scale.numel() == inv.numel():
                    inv = inv * scale.view_as(inv)

                # Compute fused weight and bias on-device without tracking autograd (eval uses no grad anyway)
                with torch.no_grad():
                    W_fused = W * inv.view(-1, 1)
                    b_fused = inv * (b_lin - running_mean) + bn_bias

                # update cache
                self._fused_cache.update({
                    "W_ptr": W_ptr,
                    "bn_weight_ptr": bw_ptr,
                    "running_var_ptr": rv_ptr,
                    "scale_ptr": sc_ptr,
                    "W_fused": W_fused,
                    "b_fused": b_fused
                })

            # Single fused GEMM (cuBLAS) + bias: fastest path using cached fused tensors
            x = F.linear(x, self._fused_cache["W_fused"], self._fused_cache["b_fused"])
            x = torch.softmax(x, dim=1)
            return x

        # Training or unfusable cases: keep training semantics.
        # Prefer folding scale into BN.weight when possible (no extra kernel launch).
        if scale.numel() == 1:
            scaled_weight = bn_weight * float(scale)
            x = F.batch_norm(x, running_mean, running_var, weight=scaled_weight, bias=bn_bias,
                             training=True, momentum=momentum, eps=eps)
        else:
            # Try to broadcast the scale into the 1-D per-channel bn_weight shape.
            scale_broadcast = None
            try:
                # move to bn's device/dtype for safe broadcasting
                scale_dev = scale.to(device=bn_weight.device, dtype=bn_weight.dtype)
                scale_broadcast = torch.broadcast_tensors(scale_dev, bn_weight)[0].view_as(bn_weight)
            except Exception:
                scale_broadcast = None

            if scale_broadcast is not None:
                scaled_weight = bn_weight * scale_broadcast
                x = F.batch_norm(x, running_mean, running_var, weight=scaled_weight, bias=bn_bias,
                                 training=True, momentum=momentum, eps=eps)
            elif scale.numel() == bn_weight.numel():
                # exact per-channel vector
                scaled_weight = bn_weight * scale.view_as(bn_weight)
                x = F.batch_norm(x, running_mean, running_var, weight=scaled_weight, bias=bn_bias,
                                 training=True, momentum=momentum, eps=eps)
            else:
                # General fallback: use module BN (handles shapes) then elementwise multiply on-device.
                x = self.bn(x)
                # Use Triton 2D scaling kernel for broadcasted scaling (better throughput on large (M,N))
                x = triton_scale_2d(x, self.scale.to(x.device))
                x = torch.softmax(x, dim=1)
                return x

        # Final softmax across feature dimension (dim=1)
        x = torch.softmax(x, dim=1)
        return x