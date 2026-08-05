import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import torch as th

# Autotune configs for the Triton kernel. We autotune over BLOCK sizes.
AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK": 64},  num_warps=4, num_stages=2),
    triton.Config({"BLOCK": 128}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK": 256}, num_warps=4, num_stages=2),
]

@triton.autotune(configs=AUTOTUNE_CONFIGS, key=['R', 'D'])
@triton.jit
def _sub_intra_norm_kernel(
    v_ptr,      # pointer to vlad flattened as (R, D) where R = B*K
    a_ptr,      # pointer to a flattened as (R, D)
    out_ptr,    # pointer to output flattened as (R, D) but will be viewed as (B, K*D)
    R,          # number of rows (B*K)
    D,          # feature dim
    final_scale, # final scale factor (1/sqrt(K))
    BLOCK: tl.constexpr,
):
    """
    For each row r in [0..R):
      - compute sumsq = sum_d (v[r,d] - a[r,d])^2
      - inv_norm = 1 / sqrt(sumsq + eps)
      - write out[r, d] = (v[r,d] - a[r,d]) * inv_norm * final_scale

    The kernel processes one row per program id. The D dimension is traversed in blocks.
    """
    row = tl.program_id(0)
    # If out-of-range row, return early
    if row >= R:
        return

    offs = tl.arange(0, BLOCK)
    # first pass: compute sumsq
    # Initialize as a Python float (will be promoted when combined with Triton expressions).
    sumsq = 0.0
    d = 0
    while d < D:
        d_idx = d + offs
        mask = d_idx < D
        # load v and a
        v = tl.load(v_ptr + row * D + d_idx, mask=mask, other=0.0).to(tl.float32)
        a = tl.load(a_ptr + row * D + d_idx, mask=mask, other=0.0).to(tl.float32)
        diff = v - a
        # mask to float32
        mask_f = mask.to(tl.float32)
        sumsq = sumsq + tl.sum(diff * diff * mask_f)
        d += BLOCK

    inv_norm = 1.0 / tl.sqrt(sumsq + 1e-12)

    # second pass: write normalized result
    d = 0
    # final_scale is a Python float passed in; use it directly (no tl.float32(...) call).
    final_scale_tl = final_scale
    while d < D:
        d_idx = d + offs
        mask = d_idx < D
        v = tl.load(v_ptr + row * D + d_idx, mask=mask, other=0.0).to(tl.float32)
        a = tl.load(a_ptr + row * D + d_idx, mask=mask, other=0.0).to(tl.float32)
        out = (v - a) * inv_norm * final_scale_tl
        tl.store(out_ptr + row * D + d_idx, out, mask=mask)
        d += BLOCK


def triton_sub_intra_normalize(vlad: torch.Tensor, a: torch.Tensor, cluster_size: int):
    """
    vlad: (B, K, D) torch.cuda.FloatTensor
    a:    (B, K, D) torch.cuda.FloatTensor
    returns: out (B, K*D) torch.cuda.FloatTensor, where:
      - subtraction (vlad - a) is done,
      - each (B, K, D) vector is intra-normalized across D,
      - final flattened vector is scaled by 1/sqrt(K) (equivalent to final L2 normalization
        after intra-normalization).
    """
    assert vlad.is_cuda and a.is_cuda, "Tensors must be on CUDA."
    B, K, D = vlad.shape
    assert a.shape == (B, K, D)

    # Flatten to (R, D) where R = B*K to let each Triton program handle one (b,k) pair
    R = B * K
    v_flat = vlad.contiguous().view(R, D)
    a_flat = a.contiguous().view(R, D)

    out = torch.empty((B, K * D), device=vlad.device, dtype=vlad.dtype)

    # final_scale = 1 / sqrt(K) as explained: after intra-normalization each cluster has norm 1,
    # so flattening gives norm sqrt(K); dividing by sqrt(K) yields final L2-normalized vector.
    final_scale = 1.0 / math.sqrt(cluster_size)

    # grid: one Triton program per row (R = B * K)
    grid = lambda meta: (R,)
    _sub_intra_norm_kernel[grid](v_flat, a_flat, out.view(-1), R, D, final_scale)
    return out


def triton_vlad_bmm(A: torch.Tensor, X: torch.Tensor):
    """
    A: (B, K, N) torch.cuda.FloatTensor
    X: (B, N, D) torch.cuda.FloatTensor
    returns: out (B, K, D) torch.cuda.FloatTensor

    For the heavy batched GEMM we delegate to PyTorch/cuBLAS with TF32 enabled,
    which is typically very fast on Ampere GPUs. We ensure contiguity and enable TF32.
    """
    assert A.is_cuda and X.is_cuda, "Inputs must be CUDA tensors"
    assert A.dtype == torch.float32 and X.dtype == torch.float32
    B, K, N = A.shape
    _, N2, D = X.shape
    assert N == N2, "Incompatible N dimensions"

    A = A.contiguous()
    X = X.contiguous()

    # Enable TF32 on Ampere GPUs to allow cuBLAS to use Tensor Cores for fp32 matmuls.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    out = torch.bmm(A, X)  # (B, K, D)
    return out


class ModelNew(nn.Module):
    def __init__(self, cluster_size, feature_size, ghost_clusters):
        super(ModelNew, self).__init__()

        self.feature_size = feature_size
        self.cluster_size = cluster_size
        self.ghost_clusters = ghost_clusters

        init_sc = (1 / math.sqrt(feature_size))
        clusters = cluster_size + ghost_clusters

        # The `clusters` weights are the `(w,b)` in the paper
        self.clusters = nn.Parameter(init_sc * th.randn(feature_size, clusters))
        self.batch_norm = nn.BatchNorm1d(clusters)
        # The `clusters2` weights are the visual words `c_k` in the paper
        self.clusters2 = nn.Parameter(init_sc * th.randn(1, feature_size, cluster_size))
        self.out_dim = self.cluster_size * feature_size

    def forward(self, x, mask=None):
        """Aggregates feature maps into a fixed size representation.  In the following
        notation, B = batch_size, N = num_features, K = num_clusters, D = feature_size.

        Args:
            x (th.Tensor): B x N x D

        Returns:
            (th.Tensor): B x DK
        """
        max_sample = x.size(1)
        B = x.size(0)
        x_flat = x.view(-1, self.feature_size)  # (B*N) x D

        if x_flat.device != self.clusters.device:
            msg = f"x.device {x_flat.device} != cluster.device {self.clusters.device}"
            raise ValueError(msg)

        # compute assignment logits = x_flat @ clusters  -> (BN x (K+G))
        assignment = th.matmul(x_flat, self.clusters)  # (BN x (K+G))
        assignment = self.batch_norm(assignment)
        assignment = F.softmax(assignment, dim=1)  # BN x (K+G)

        # remove ghost assignments
        assignment = assignment[:, :self.cluster_size]
        assignment = assignment.view(-1, max_sample, self.cluster_size)  # B x N x K

        # a_sum: B x K (sum over N)
        a_sum = th.sum(assignment, dim=1)  # B x K

        # clusters2: (1, D, K) -> squeeze -> (D, K) -> transpose -> (K, D)
        clusters2_kd = self.clusters2.squeeze(0).transpose(0, 1)  # K x D

        # a: B x K x D
        a = a_sum.unsqueeze(2) * clusters2_kd.unsqueeze(0)  # B x K x D

        # assignment_t: B x K x N
        assignment_t = assignment.transpose(1, 2)  # B x K x N

        # Use fast cuBLAS batched GEMM for the heavy operation
        vlad = triton_vlad_bmm(assignment_t, x)  # B x K x D

        # vlad is B x K x D, a is B x K x D, now fuse subtraction + intra-L2-normalization +
        # final flattening and final normalization by 1/sqrt(K) in a Triton kernel.
        out = triton_sub_intra_normalize(vlad, a, self.cluster_size)  # B x (K*D)

        return out  # B x DK


# Input helpers with CUDA tensors for evaluation
batch_size = 2048
num_features = 100
num_clusters = 32
feature_size = 512
ghost_clusters = 16

def get_inputs():
    return [torch.rand(batch_size, num_features, feature_size).cuda()]

def get_init_inputs():
    return [num_clusters, feature_size, ghost_clusters]