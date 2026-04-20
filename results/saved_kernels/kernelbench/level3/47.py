# Copyright 2018 Antoine Miech All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Highly-optimized VLAD Model using Triton:
- Fuses the very tall-skinny matmul with a numerically-stable row-wise softmax
  and optional BatchNorm affine (inference) into a single Triton kernel.
- Uses mixed precision (FP16 tiles, FP32 accumulation) to leverage Tensor Cores
  on Ampere (A6000) and reduce global memory traffic for the large operands.
- Training path preserves exact PyTorch BatchNorm1d semantics by using cuBLAS-backed
  torch.matmul + PyTorch BatchNorm1d followed by a fast Triton row-wise softmax.
- The fused kernel is aggressively autotuned for large BLOCK_M and BLOCK_K to
  maximize throughput on the A6000.
- Remaining VLAD aggregation uses cuBLAS batched matmuls which are efficient
  on these sizes.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch as th

import triton
import triton.language as tl

# -----------------------
# Fused matmul (+ optional BN/bias) -> rowwise softmax Triton kernel
# Tuned for very tall M and small N (N ~ 32). Supports mixed precision (FP16 tiles).
# -----------------------
FUSED_AUTOTUNE_CONFIGS = [
    # Conservative, A6000-friendly configs to avoid excessive shared memory usage.
    triton.Config({"BLOCK_M": 8192,  "BLOCK_K": 256, "BLOCK_N": 32}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 4096,  "BLOCK_K": 256, "BLOCK_N": 32}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 2048,  "BLOCK_K": 128, "BLOCK_N": 32}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 1024,  "BLOCK_K": 64,  "BLOCK_N": 32}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 512,   "BLOCK_K": 64,  "BLOCK_N": 32}, num_warps=2, num_stages=2),
]


@triton.autotune(
    configs=FUSED_AUTOTUNE_CONFIGS,
    key=['M', 'K', 'N'],
)
@triton.jit
def _matmul_rowwise_softmax_kernel(
    A_ptr,          # A: M x K (row-major)
    B_ptr,          # B: K x N (row-major)
    Out_ptr,        # Out: M x N (row-major) -> softmax across N
    M, K, N,        # dimensions
    bias_ptr,       # bias vector length N (used when apply_bn == 0)
    gamma_ptr,      # bn weight length N
    beta_ptr,       # bn bias length N
    rm_ptr,         # running_mean length N
    rv_ptr,         # running_var length N
    eps,            # bn eps
    apply_bn,       # 1 -> apply BN using running stats inside kernel; 0 -> add bias_ptr
    use_fp16,       # 1 -> cast tiles to fp16 for dot to leverage tensor cores
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row_block = tl.program_id(0)
    row_ids = row_block * BLOCK_M + tl.arange(0, BLOCK_M)   # (BLOCK_M,)
    col_ids = tl.arange(0, BLOCK_N)                         # (BLOCK_N,)

    row_mask = row_ids < M
    col_mask = col_ids < N
    mask = row_mask[:, None] & col_mask[None, :]

    # accumulator in fp32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension in tiles
    for k_start in range(0, K, BLOCK_K):
        k_ids = k_start + tl.arange(0, BLOCK_K)              # (BLOCK_K,)
        k_mask = k_ids < K

        # load A tile: (BLOCK_M, BLOCK_K) from A_ptr laid out M x K
        offs_a = row_ids[:, None] * K + k_ids[None, :]
        a_tile = tl.load(A_ptr + offs_a, mask=(row_mask[:, None] & k_mask[None, :]), other=0.0)

        # load B tile: (BLOCK_K, BLOCK_N) from B_ptr laid out K x N
        offs_b = k_ids[:, None] * N + col_ids[None, :]
        b_tile = tl.load(B_ptr + offs_b, mask=(k_mask[:, None] & col_mask[None, :]), other=0.0)

        if use_fp16 != 0:
            # cast tiles to fp16 for dot product (tensor core friendly)
            a_f = tl.cast(a_tile, tl.float16)
            b_f = tl.cast(b_tile, tl.float16)
            acc += tl.dot(a_f, b_f)
        else:
            acc += tl.dot(a_tile, b_tile)

    # Apply BN affine using running stats (inference semantics) or add pre-folded bias
    if apply_bn != 0:
        gamma = tl.load(gamma_ptr + col_ids, mask=col_mask, other=1.0)
        beta = tl.load(beta_ptr + col_ids, mask=col_mask, other=0.0)
        rm = tl.load(rm_ptr + col_ids, mask=col_mask, other=0.0)
        rv = tl.load(rv_ptr + col_ids, mask=col_mask, other=1.0)
        invstd = 1.0 / tl.sqrt(rv + eps)
        acc = (acc - rm[None, :]) * invstd[None, :] * gamma[None, :] + beta[None, :]
    else:
        bias = tl.load(bias_ptr + col_ids, mask=col_mask, other=0.0)
        acc = acc + bias[None, :]

    # Compute numerically stable softmax across columns for each row
    neg_inf = -1e20
    acc_safe = tl.where(mask, acc, tl.full((BLOCK_M, BLOCK_N), neg_inf, dtype=tl.float32))
    max_val = tl.max(acc_safe, axis=1)              # (BLOCK_M,)
    vals_sub = acc_safe - max_val[:, None]
    exp_vals = tl.exp(vals_sub)
    sum_exp = tl.sum(exp_vals, axis=1)              # (BLOCK_M,)
    out = exp_vals / (sum_exp[:, None] + 1e-20)

    offs_out = row_ids[:, None] * N + col_ids[None, :]
    tl.store(Out_ptr + offs_out, out, mask=mask)


# -----------------------
# Row-wise softmax kernel (fallback for training path)
# -----------------------
SOFTMAX_AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK_ROWS": 256, "BLOCK_COLS": 32}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_ROWS": 128, "BLOCK_COLS": 32}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_ROWS": 64,  "BLOCK_COLS": 32}, num_warps=8, num_stages=2),
]


@triton.autotune(configs=SOFTMAX_AUTOTUNE_CONFIGS, key=['n_rows', 'n_cols'])
@triton.jit
def _softmax_rowwise_kernel(
    x_ptr,
    out_ptr,
    n_rows,
    n_cols,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
):
    row_block = tl.program_id(0)
    row_ids = row_block * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    col_ids = tl.arange(0, BLOCK_COLS)

    offs = row_ids[:, None] * n_cols + col_ids[None, :]
    mask_cols = col_ids[None, :] < n_cols
    mask_rows = row_ids[:, None] < n_rows
    mask = mask_cols & mask_rows

    neg_inf = -1e20
    vals = tl.load(x_ptr + offs, mask=mask, other=neg_inf)

    max_val = tl.max(vals, axis=1)
    vals_sub = vals - max_val[:, None]
    exp_vals = tl.exp(vals_sub)
    sum_exp = tl.sum(exp_vals, axis=1)
    out = exp_vals / (sum_exp[:, None] + 1e-20)
    tl.store(out_ptr + offs, out, mask=mask)


def triton_rowwise_softmax(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda, "Input must be on CUDA."
    assert x.dim() == 2, "triton_rowwise_softmax expects a 2D tensor."
    x_c = x.contiguous()
    rows, cols = x_c.shape
    out = torch.empty((rows, cols), device=x.device, dtype=torch.float32)
    grid = lambda meta: (triton.cdiv(rows, meta['BLOCK_ROWS']),)
    _softmax_rowwise_kernel[grid](x_c, out, rows, cols)
    return out


def triton_matmul_rowwise_softmax(A: torch.Tensor, B: torch.Tensor,
                                  bn_weight: torch.Tensor = None,
                                  bn_bias: torch.Tensor = None,
                                  running_mean: torch.Tensor = None,
                                  running_var: torch.Tensor = None,
                                  eps: float = 1e-5,
                                  apply_bn: bool = False,
                                  use_fp16: bool = False,
                                  bias: torch.Tensor = None) -> torch.Tensor:
    """
    Wrapper for fused matmul + (BN or bias) + rowwise softmax.
    - If apply_bn=True, bn_* and running_* must be provided (kernel applies BN using running stats).
    - If apply_bn=False, 'bias' must be provided (pre-folded bias).
    use_fp16 toggles casting of tiles to fp16 for faster matmuls.
    Returns float32 output (M x N).
    """
    assert A.is_cuda and B.is_cuda, "Inputs must be CUDA tensors."
    assert A.dim() == 2 and B.dim() == 2, "A and B must be 2D."
    M, K = A.shape
    Kb, N = B.shape
    assert K == Kb, "Inner dimensions must match."

    A_c = A.contiguous()
    B_c = B.contiguous()

    out = torch.empty((M, N), device=A.device, dtype=torch.float32)

    if apply_bn:
        assert bn_weight is not None and bn_bias is not None and running_mean is not None and running_var is not None
        gamma = bn_weight.contiguous().to(dtype=torch.float32, device=A.device)
        beta = bn_bias.contiguous().to(dtype=torch.float32, device=A.device)
        rm = running_mean.contiguous().to(dtype=torch.float32, device=A.device)
        rv = running_var.contiguous().to(dtype=torch.float32, device=A.device)
        bias_tensor = torch.zeros((N,), device=A.device, dtype=torch.float32)
    else:
        gamma = torch.ones((N,), device=A.device, dtype=torch.float32)
        beta = torch.zeros((N,), device=A.device, dtype=torch.float32)
        rm = torch.zeros((N,), device=A.device, dtype=torch.float32)
        rv = torch.ones((N,), device=A.device, dtype=torch.float32)
        if bias is not None:
            # ensure bias is float32 for stable addition inside kernel
            bias_tensor = bias.contiguous().to(dtype=torch.float32, device=A.device)
        else:
            bias_tensor = torch.zeros((N,), device=A.device, dtype=torch.float32)

    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']),)
    _matmul_rowwise_softmax_kernel[grid](
        A_c, B_c, out, M, K, N,
        bias_tensor, gamma, beta, rm, rv, float(eps), int(1 if apply_bn else 0), int(1 if use_fp16 else 0)
    )
    return out


# -----------------------
# Fused aggregation kernel: compute VLAD per-sample from assignment_probs and x
# -----------------------
@triton.jit
def _aggregate_vlad_kernel(
    assignment_ptr,  # N_rows x K  (row-major) -- assignments for a single sample
    x_ptr,           # N_rows x D  (row-major) -- features for a single sample
    out_ptr,         # K x D (row-major) output for this sample
    N_rows, K, D,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
    TILE_N: tl.constexpr,
    USE_FP16: tl.constexpr,
):
    """
    Compute out = assignment^T @ x for a single sample (assignment: N x K, x: N x D).
    Tiled implementation reduces over N using tl.dot on tiles:
      - Load assignment_tile already transposed to (BLOCK_K x TILE_N) to avoid calling .transpose()
        on Triton tensors (illegal) and to improve memory layout for the dot.
      - Load x_tile: (TILE_N x BLOCK_D)
      - Compute acc += (assignment_tile^T) @ x_tile via tl.dot, using fp16 tiles when USE_FP16 != 0.
    Accumulation is kept in fp32. Masks are used for tail tiles.
    """
    k_idx = tl.arange(0, BLOCK_K)   # (BLOCK_K,)
    d_idx = tl.arange(0, BLOCK_D)   # (BLOCK_D,)
    n_idx = tl.arange(0, TILE_N)    # (TILE_N,)

    # tile over K and D for output tiles
    for k_start in range(0, K, BLOCK_K):
        for d_start in range(0, D, BLOCK_D):
            acc = tl.zeros((BLOCK_K, BLOCK_D), dtype=tl.float32)

            # reduction over N in tiles
            for n_start in range(0, N_rows, TILE_N):
                # masks for current tiles
                k_mask = (k_start + k_idx) < K            # (BLOCK_K,)
                d_mask = (d_start + d_idx) < D            # (BLOCK_D,)
                n_mask = (n_start + n_idx) < N_rows       # (TILE_N,)

                # load assignment tile already transposed into (BLOCK_K, TILE_N):
                # For row-major assignment (N_rows x K), the element at (n, k) is at ptr + n*K + k.
                # To form a (BLOCK_K x TILE_N) tile where rows are k and cols are n:
                # offs = k_idx[:, None] + (n_start + n_idx)[None, :] * K
                a_offs = (k_start + k_idx)[:, None] + (n_start + n_idx)[None, :] * K
                a_tile_t = tl.load(assignment_ptr + a_offs, mask=(k_mask[:, None] & n_mask[None, :]), other=0.0)
                # a_tile_t has shape (BLOCK_K, TILE_N)

                # load feature tile: shape (TILE_N, BLOCK_D)
                x_offs = (n_start + n_idx)[:, None] * D + (d_start + d_idx)[None, :]
                x_tile = tl.load(x_ptr + x_offs, mask=(n_mask[:, None] & d_mask[None, :]), other=0.0)  # (TILE_N, BLOCK_D)

                # perform tiled dot: (BLOCK_K x TILE_N) @ (TILE_N x BLOCK_D) -> (BLOCK_K x BLOCK_D)
                if USE_FP16 != 0:
                    # Use fp16 tiles to leverage tensor cores; accumulation remains fp32
                    a_f = tl.cast(a_tile_t, tl.float16)
                    x_f = tl.cast(x_tile, tl.float16)
                    acc += tl.dot(a_f, x_f)
                else:
                    # use fp32 tiles for higher precision
                    acc += tl.dot(a_tile_t, x_tile)

            # store accumulated tile into output K x D
            offs = (k_start + k_idx)[:, None] * D + (d_start + d_idx)[None, :]
            k_mask_tail = (k_start + k_idx) < K
            d_mask_tail = (d_start + d_idx) < D
            tl.store(out_ptr + offs, acc, mask=(k_mask_tail[:, None] & d_mask_tail[None, :]))


# -----------------------
# ModelNew using fused kernels
# -----------------------
class ModelNew(nn.Module):
    def __init__(self, cluster_size, feature_size, ghost_clusters):
        super(ModelNew, self).__init__()

        self.feature_size = feature_size
        self.cluster_size = cluster_size
        self.ghost_clusters = ghost_clusters

        init_sc = (1 / math.sqrt(feature_size))
        clusters = cluster_size + ghost_clusters

        # clusters: D x (K+G)
        self.clusters = nn.Parameter(init_sc * th.randn(feature_size, clusters))
        self.batch_norm = nn.BatchNorm1d(clusters)
        # clusters2: 1 x D x K
        self.clusters2 = nn.Parameter(init_sc * th.randn(1, feature_size, cluster_size))
        self.out_dim = self.cluster_size * feature_size

    def forward(self, x, mask=None):
        """
        x: B x N x D
        returns: B x (D*K)
        """
        max_sample = x.size(1)
        # Flatten to (B*N) x D
        x_flat = x.view(-1, self.feature_size)  # (B*N) x D

        if x_flat.device != self.clusters.device:
            msg = f"x.device {x_flat.device} != cluster.device {self.clusters.device}"
            raise ValueError(msg)

        M = x_flat.shape[0]
        K_dim = self.clusters.shape[1]  # total clusters (K + ghosts)

        # Inference/eval path: use fused kernel and prefer BN-folding when no ghosts to reduce memory loads.
        if not self.batch_norm.training:
            device = self.clusters.device
            dtype = self.clusters.dtype

            gamma = (self.batch_norm.weight.detach() if self.batch_norm.weight is not None else torch.ones((K_dim,), device=device)).to(torch.float32)
            beta = (self.batch_norm.bias.detach() if self.batch_norm.bias is not None else torch.zeros((K_dim,), device=device)).to(torch.float32)
            rm = self.batch_norm.running_mean.detach().to(torch.float32)
            rv = self.batch_norm.running_var.detach().to(torch.float32)
            eps = float(self.batch_norm.eps)

            if self.ghost_clusters == 0:
                # Fold BN into clusters for faster inference: W' = W * (gamma / sqrt(running_var + eps))
                invstd = 1.0 / torch.sqrt(rv + eps)
                scale = (invstd * gamma).to(dtype)  # length K_dim
                folded_clusters = (self.clusters * scale[None, :]).detach().contiguous()  # D x K
                folded_bias = ((-rm * invstd * gamma) + beta).detach().contiguous()      # length K

                # Use mixed precision on the wire: cast large tensors to fp16 to reduce bandwidth.
                # Keep accumulation and softmax in fp32 (kernel returns fp32).
                A_in = x_flat.half().contiguous()
                B_in = folded_clusters.half().contiguous()

                # pass folded_bias as fp32 to kernel (kernel will add it in fp32)
                assignment_prob = triton_matmul_rowwise_softmax(
                    A_in, B_in,
                    eps=eps, apply_bn=False, use_fp16=True, bias=folded_bias.float().contiguous().to(device)
                )
            else:
                # There are ghost clusters: slice out the used columns to avoid computing ghosts
                clusters_sliced = self.clusters[:, :self.cluster_size].contiguous()
                gamma_s = gamma[:self.cluster_size].contiguous()
                beta_s = beta[:self.cluster_size].contiguous()
                rm_s = rm[:self.cluster_size].contiguous()
                rv_s = rv[:self.cluster_size].contiguous()

                A_in = x_flat.half().contiguous()
                B_in = clusters_sliced.half().contiguous()

                assignment_prob = triton_matmul_rowwise_softmax(
                    A_in, B_in,
                    bn_weight=gamma_s.to(device), bn_bias=beta_s.to(device),
                    running_mean=rm_s.to(device), running_var=rv_s.to(device),
                    eps=eps, apply_bn=True, use_fp16=True
                )
        else:
            # Training path: preserve PyTorch BatchNorm training semantics.
            assignment_logits = torch.matmul(x_flat.contiguous(), self.clusters.contiguous())  # (BN x (K+G))
            assignment_logits = self.batch_norm(assignment_logits)  # PyTorch BatchNorm (training semantics)
            assignment_prob = triton_rowwise_softmax(assignment_logits.contiguous())

        # Remove ghost assignments if any and reshape to B x N x K
        assignment_prob = assignment_prob[:, :self.cluster_size]
        assignment_prob = assignment_prob.view(-1, max_sample, self.cluster_size)  # B x N x K

        # a_sum = sum over N -> B x 1 x K
        a_sum = th.sum(assignment_prob, dim=1, keepdim=True)  # B x 1 x K

        # Use Triton kernel to aggregate per-sample to avoid cuBLAS batched matmul
        B = assignment_prob.shape[0]
        N = assignment_prob.shape[1]
        K = self.cluster_size
        D = self.feature_size

        # ensure contiguous for pointer arithmetic inside Triton
        assignment_contig = assignment_prob.contiguous()
        x_reshaped = x_flat.view(-1, max_sample, self.feature_size)
        x_contig = x_reshaped.contiguous()

        # output vlad: B x K x D (we will transpose to B x D x K later)
        vlad = torch.empty((B, K, D), device=x.device, dtype=torch.float32)

        # Launch the Triton aggregation kernel with one program per sample (grid=B)
        # Choose BLOCK_K = K (full K fits comfortably, K is small, e.g., 32)
        # and BLOCK_D = 32 to reduce per-block shared/register pressure (use smaller tiles).
        grid = (B,)
        # Pick a modest TILE_N to keep per-block scratch small; N≈100 works well with TILE_N=64.
        TILE_N = 64
        # Enable fp16 tiles by passing USE_FP16=1 as the last constexpr argument to the kernel.
        _aggregate_vlad_kernel[grid](assignment_contig, x_contig, vlad, N, K, D, K, 32, TILE_N, 1)

        # subtract a_sum * clusters2 in the (B x K x D) layout to avoid an extra big transpose.
        # clusters2: (1 x D x K) -> clusters2_t: (1 x K x D)
        clusters2_t = self.clusters2.permute(0, 2, 1).contiguous()
        # a_sum: B x 1 x K -> a_sum_t: B x K x 1
        a_sum_t = a_sum.permute(0, 2, 1)
        vlad = vlad - a_sum_t * clusters2_t
        # transpose to B x D x K for subsequent normalization
        vlad = vlad.transpose(1, 2)  # -> B x D x K

        # L2 intra norm (normalize across feature dimension D)
        vlad = F.normalize(vlad, p=2, dim=1)

        # flattening + final L2 norm
        vlad = vlad.reshape(-1, self.cluster_size * self.feature_size)  # -> B x DK
        vlad = F.normalize(vlad, p=2, dim=1)
        return vlad  # B x DK


# Keep helper variables for compatibility / testing harness
batch_size = 2048
num_features = 100
num_clusters = 32
feature_size = 512
ghost_clusters = 0

def get_inputs():
    # Return CUDA inputs for the GPU kernels
    return [torch.rand(batch_size, num_features, feature_size).cuda()]

def get_init_inputs():
    return [num_clusters, feature_size, ghost_clusters]