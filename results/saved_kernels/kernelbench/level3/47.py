import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import torch as th

# Triton kernel: L2-normalize many vectors of length L (each vector normalized independently).
# This kernel processes ROWS_PER_PROG rows per program and tiles the vector length with
# a constexpr BLOCK so each row can be done in few (or one) chunk(s).
@triton.jit
def _l2_norm_kernel(
    x_ptr,           # pointer to input (flat)
    out_ptr,         # pointer to output (flat)
    num_rows,        # number of rows (runtime)
    L,               # length of each vector (runtime)
    eps,             # small epsilon to avoid div by zero
    ROWS_PER_PROG: tl.constexpr,  # how many rows each program handles
    BLOCK: tl.constexpr,  # tile size for processing chunks of the vector
):
    pid = tl.program_id(0)  # which program
    base_row = pid * ROWS_PER_PROG

    # rows this program handles
    rows = base_row + tl.arange(0, ROWS_PER_PROG)             # shape: (ROWS_PER_PROG,)
    row_mask = rows < num_rows                               # which rows are valid
    row_base = rows * L                                      # starting index for each row in flat buffer

    # accumulator per row
    sumsq = tl.zeros((ROWS_PER_PROG,), dtype=tl.float32)

    off = 0
    # loop over chunks of size BLOCK (BLOCK is constexpr)
    while off < L:
        offs = tl.arange(0, BLOCK)                           # shape: (BLOCK,)
        # compute 2D indices: (ROWS_PER_PROG, BLOCK)
        idxs = row_base[:, None] + off + offs[None, :]
        # mask columns inside vector bounds and invalidate whole rows beyond num_rows
        mask_cols = (off + offs) < L                         # shape: (BLOCK,)
        mask = row_mask[:, None] & mask_cols[None, :]        # shape: (ROWS_PER_PROG, BLOCK)
        vals = tl.load(x_ptr + idxs, mask=mask, other=0.0)   # shape: (ROWS_PER_PROG, BLOCK)
        # accumulate sum of squares per row (reduce across BLOCK axis)
        sumsq = sumsq + tl.sum(vals * vals, axis=1)
        off += BLOCK

    # compute inverse norm per row
    inv_norm = 1.0 / tl.sqrt(sumsq + eps)                   # shape: (ROWS_PER_PROG,)

    # Second pass: write normalized values for all valid rows handled by this program
    off = 0
    while off < L:
        offs = tl.arange(0, BLOCK)
        idxs = row_base[:, None] + off + offs[None, :]
        mask_cols = (off + offs) < L
        mask = row_mask[:, None] & mask_cols[None, :]
        vals = tl.load(x_ptr + idxs, mask=mask, other=0.0)
        out_vals = vals * inv_norm[:, None]
        tl.store(out_ptr + idxs, out_vals, mask=mask)
        off += BLOCK

@triton.jit
def _l2_norm_kernel_fused(
    x_ptr,           # pointer to input (flat)
    out_ptr,         # pointer to output (flat)
    num_rows,        # number of rows (runtime)
    L,               # length of each vector (runtime)
    eps,             # small epsilon to avoid div by zero
    sample_scale,    # runtime scalar to apply after intra-normalization (1.0 / sqrt(K))
    ROWS_PER_PROG: tl.constexpr,  # how many rows each program handles
    BLOCK: tl.constexpr,          # tile size for processing chunks of the vector
):
    """
    Fused kernel that:
      - computes per-row sumsq over length L
      - computes inv_norm = 1.0 / sqrt(sumsq + eps)
      - writes normalized row values multiplied by sample_scale

    This replaces the previous multi-kernel pipeline and eliminates the per-row sums buffer
    and the separate reduction + scaling kernels.
    """
    pid = tl.program_id(0)
    base_row = pid * ROWS_PER_PROG

    rows = base_row + tl.arange(0, ROWS_PER_PROG)       # (ROWS_PER_PROG,)
    row_mask = rows < num_rows                          # (ROWS_PER_PROG,)
    row_base = rows * L

    # accumulate sumsq per row
    sumsq = tl.zeros((ROWS_PER_PROG,), dtype=tl.float32)

    off = 0
    # loop over chunks; when BLOCK == L this executes a single iteration
    while off < L:
        offs = tl.arange(0, BLOCK)
        idxs = row_base[:, None] + off + offs[None, :]
        mask_cols = (off + offs) < L
        mask = row_mask[:, None] & mask_cols[None, :]
        vals = tl.load(x_ptr + idxs, mask=mask, other=0.0)   # (ROWS_PER_PROG, BLOCK)
        # keep as fp32; tl.load returns fp32 for fp32 tensors already
        sumsq = sumsq + tl.sum(vals * vals, axis=1)
        off += BLOCK

    # compute inverse norm per row (for normalization)
    inv_norm = 1.0 / tl.sqrt(sumsq + eps)  # (ROWS_PER_PROG,)

    # second pass: write normalized values and apply sample-level scaling
    off = 0
    while off < L:
        offs = tl.arange(0, BLOCK)
        idxs = row_base[:, None] + off + offs[None, :]
        mask_cols = (off + offs) < L
        mask = row_mask[:, None] & mask_cols[None, :]
        vals = tl.load(x_ptr + idxs, mask=mask, other=0.0)
        out_vals = vals * inv_norm[:, None] * sample_scale
        tl.store(out_ptr + idxs, out_vals, mask=mask)
        off += BLOCK

def triton_l2_normalize_rows(x: torch.Tensor, B: int = None, K: int = None, eps: float = 1e-12, BLOCK: int = 512, ROWS_PER_PROG: int = 8, out: torch.Tensor = None):
    """
    Normalize each row of x (2D tensor) to unit L2 norm, and apply a per-sample scalar scaling
    equal to 1.0 / sqrt(K) (this is equivalent to flattening and L2-normalizing across all rows
    when each intra-row vector has been L2-normalized).
    This single-kernel implementation computes per-row normalization and applies the sample-level
    scalar in the same pass, removing extra buffers and kernels.

    x: Tensor of shape (num_rows, L), on CUDA, dtype float32
    B: number of samples (unused in the kernel, provided for API compatibility)
    K: number of rows per sample (cluster_size). If provided sample_scale = 1.0 / sqrt(K), otherwise 1.0.
    """
    assert x.is_cuda, "Input must be on CUDA"
    assert x.dtype == torch.float32, "Only float32 supported"

    x_contig = x.contiguous()
    num_rows, L = x_contig.shape

    # If out is provided, validate; otherwise operate in-place.
    if out is None:
        out_contig = x_contig
    else:
        if not out.is_cuda:
            raise ValueError("out must be a CUDA tensor")
        if out.dtype != torch.float32:
            raise ValueError("out must be float32")
        if out.shape != x_contig.shape:
            raise ValueError("out must have same shape as x")
        out_contig = out.contiguous()

    # compute sample-level scaling scalar (1 / sqrt(K)) if K provided; otherwise default to 1.0
    sample_scale = 1.0
    if K is not None:
        sample_scale = 1.0 / math.sqrt(float(K))

    # grid: one program handles ROWS_PER_PROG rows
    grid = ((num_rows + ROWS_PER_PROG - 1) // ROWS_PER_PROG,)

    # launch fused kernel; pass ROWS_PER_PROG and BLOCK as constexpr
    _l2_norm_kernel_fused[grid](
        x_contig,               # x_ptr
        out_contig,             # out_ptr
        num_rows,               # num_rows (runtime)
        L,                      # L (runtime)
        eps,                    # eps
        sample_scale,           # runtime scalar applied after intra-normalization
        ROWS_PER_PROG=ROWS_PER_PROG,
        BLOCK=BLOCK
    )
    return out_contig

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
        # Store them directly as (K, D) so forward can use them without squeezing/transposing.
        self.clusters2 = nn.Parameter(init_sc * th.randn(cluster_size, feature_size))
        self.out_dim = self.cluster_size * feature_size

    def forward(self, x, mask=None):
        """
        Aggregates feature maps into a fixed size representation.  In the following
        notation, B = batch_size, N = num_features, K = num_clusters, D = feature_size.

        Args:
            x (th.Tensor): B x N x D

        Returns:
            (th.Tensor): B x DK
        """
        if x.dim() != 3:
            raise ValueError("Input must be 3D B x N x D")

        max_sample = x.size(1)
        B = x.size(0)
        N = x.size(1)
        D = self.feature_size
        K = self.cluster_size

        x_flat = x.view(-1, self.feature_size)  # (B*N) x D

        if x_flat.device != self.clusters.device:
            msg = f"x.device {x_flat.device} != cluster.device {self.clusters.device}"
            raise ValueError(msg)

        # Compute assignments via linear projection and batchnorm + softmax
        assignment = th.matmul(x_flat, self.clusters)  # (BN x D) x (D x (K+G)) -> BN x (K+G)
        assignment = self.batch_norm(assignment)
        assignment = F.softmax(assignment, dim=1)  # BN x (K+G)
        # remove ghost assignments
        assignment = assignment[:, :self.cluster_size]
        assignment = assignment.view(-1, max_sample, self.cluster_size)  # -> B x N x K
        a_sum = th.sum(assignment, dim=1, keepdim=True)  # B x 1 x K

        # Build `a` in B x K x D layout to match `vlad` (which will be B x K x D).
        clusters2_kd = self.clusters2  # K x D (pre-shaped in constructor)
        a = a_sum.squeeze(1).unsqueeze(2) * clusters2_kd.unsqueeze(0)  # B x K x D

        assignment = assignment.transpose(1, 2)  # B x K x N
        x_reshaped = x_flat.view(-1, max_sample, self.feature_size)  # B x N x D

        # vlad = assignment @ x  -> (B x K x N) x (B x N x D) -> B x K x D
        vlad = th.matmul(assignment, x_reshaped)  # B x K x D

        # subtract a (same layout)
        vlad = vlad - a  # B x K x D

        # L2 intra norm across the D dimension for each (B, K) vector.
        # Keep vlad in B x K x D, view as (B*K, D) and normalize rows using Triton multi-kernel pipeline.
        vlad_rows = vlad.contiguous().view(-1, self.feature_size)  # (B*K) x D

        # Use Triton multi-kernel pipeline to:
        #  - compute per-row sumssq and normalized rows,
        #  - reduce per-sample sums into per-sample inv_sqrt (K passed as constexpr),
        #  - scale rows by per-sample inv_sqrt.
        # Choose BLOCK and ROWS_PER_PROG tuned for Ampere and D=512.
        vlad_rows_norm = triton_l2_normalize_rows(vlad_rows, B=B, K=K, eps=1e-12, BLOCK=256, ROWS_PER_PROG=4)

        # reshape back to B x K x D (already fully normalized per-sample)
        vlad = vlad_rows_norm.view(B, self.cluster_size, self.feature_size)  # B x K x D

        # flatten to B x DK and return
        vlad_flat = vlad.reshape(B, self.cluster_size * self.feature_size)  # -> B x DK
        return vlad_flat  # B x DK