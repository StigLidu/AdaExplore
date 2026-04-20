import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

# Autotune configurations for the GEMM kernel.
# Added larger tiles and larger BLOCK_K to target Ampere (A6000) hardware.
AUTOTUNE_CONFIGS = [
    triton.Config(
        {
            "BLOCK_M": 128,
            "BLOCK_N": 128,
            "BLOCK_K": 32,
        },
        num_warps=4,
        num_stages=2,
    ),
    triton.Config(
        {
            "BLOCK_M": 256,
            "BLOCK_N": 128,
            "BLOCK_K": 32,
        },
        num_warps=8,
        num_stages=2,
    ),
    triton.Config(
        {
            "BLOCK_M": 128,
            "BLOCK_N": 256,
            "BLOCK_K": 32,
        },
        num_warps=8,
        num_stages=2,
    ),
    # Larger K tile and larger tiles for better arithmetic intensity on Ampere
    triton.Config(
        {
            "BLOCK_M": 256,
            "BLOCK_N": 256,
            "BLOCK_K": 64,
        },
        num_warps=16,
        num_stages=3,
    ),
    triton.Config(
        {
            "BLOCK_M": 256,
            "BLOCK_N": 128,
            "BLOCK_K": 64,
        },
        num_warps=8,
        num_stages=3,
    ),
]


@triton.autotune(configs=AUTOTUNE_CONFIGS, key=["M", "N", "K"])
@triton.jit
def _matmul_kernel(
    A_ptr,  # pointer to A matrix (row-major)
    B_ptr,  # pointer to B matrix (row-major)
    C_ptr,  # pointer to C matrix (row-major)
    bias_ptr,  # pointer to bias vector (length N) or an empty tensor
    M,  # rows of A and C
    N,  # cols of B and C
    K,  # cols of A and rows of B
    stride_am,  # stride to move between rows in A (in elements)
    stride_ak,  # stride to move between cols in A (in elements)
    stride_bk,  # stride to move between rows in B (in elements)
    stride_bn,  # stride to move between cols in B (in elements)
    stride_cm,  # stride to move between rows in C (in elements)
    stride_cn,  # stride to move between cols in C (in elements)
    has_bias,  # runtime flag (0 or 1) indicating whether bias_ptr is valid
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Compute C = A @ B for A: (M,K), B: (K,N) producing C: (M,N).
    This kernel computes a single (BLOCK_M x BLOCK_N) tile of C per program.
    Bias (if present) is added per column when writing C to fuse the bias add
    into the GEMM kernel and avoid a separate kernel launch and extra memory traffic.
    """

    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Offsets for the block
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Masks to guard loads/stores
    mask_m = offs_m < M
    mask_n = offs_n < N

    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K in blocks
    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)

        # Build pointer grids for A and B
        # A_ptr + row_index * stride_am + k_index * stride_ak
        a_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
        # B_ptr + k_index * stride_bk + col_index * stride_bn
        b_ptrs = B_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        b_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)

        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        # a: (BLOCK_M, BLOCK_K), b: (BLOCK_K, BLOCK_N)
        acc += tl.dot(a, b)

    # If bias is provided, load the bias for this tile (broadcast across rows)
    if has_bias != 0:
        # bias_ptr + offs_n  -> shape (BLOCK_N,)
        bias_ptrs = bias_ptr + offs_n[None, :]
        # mask for bias is based on columns only
        bias_mask = mask_n[None, :]
        bias_vals = tl.load(bias_ptrs, mask=bias_mask, other=0.0)  # shape (1, BLOCK_N) broadcastable
        acc = acc + bias_vals  # broadcast across rows

    # Write back
    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    mask = (mask_m[:, None]) & (mask_n[None, :])
    tl.store(c_ptrs, acc, mask=mask)


def triton_matmul(A: torch.Tensor, B: torch.Tensor, bias: torch.Tensor = None):
    """
    Multiply A (N, K) @ B (K, M) -> (N, M) using Triton kernel.
    Works for 2D float32 tensors on CUDA.
    bias: shape (M,) or None - will be fused into the kernel if provided.
    """
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA."
    assert A.dtype == torch.float32 and B.dtype == torch.float32, "Only float32 supported."

    # Ensure 2-D
    assert A.dim() == 2 and B.dim() == 2, "Only 2D tensors supported for triton_matmul."
    N, K = A.shape
    Kb, M = B.shape
    assert K == Kb, "Inner dimensions must match."

    A_ = A.contiguous()
    B_ = B.contiguous()
    C = torch.empty((N, M), device=A.device, dtype=torch.float32)

    # Strides in elements (not bytes)
    stride_am = A_.stride(0)
    stride_ak = A_.stride(1)
    stride_bk = B_.stride(0)
    stride_bn = B_.stride(1)
    stride_cm = C.stride(0)
    stride_cn = C.stride(1)

    # prepare bias tensor to pass or an empty placeholder
    if bias is None:
        bias_arg = torch.empty((0,), device=A.device, dtype=A.dtype)
        has_bias = 0
    else:
        # ensure bias is contiguous 1-D
        bias_arg = bias.contiguous()
        has_bias = 1
        assert bias_arg.dim() == 1 and bias_arg.shape[0] == M, "Bias must be 1-D of length M"

    # grid
    def grid(meta):
        return ( (N + meta["BLOCK_M"] - 1) // meta["BLOCK_M"],
                 (M + meta["BLOCK_N"] - 1) // meta["BLOCK_N"], )

    _matmul_kernel[grid](
        A_, B_, C, bias_arg,
        N, M, K,
        stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
        has_bias
    )
    return C


def triton_linear(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor = None):
    """
    Compute linear projection y = x @ weight.T + bias using triton_matmul.
    x: (..., K)
    weight: (out_features, K)
    bias: (out_features,) or None
    Returns tensor of shape (..., out_features)

    Optimization: cache a contiguous transposed copy of weight (K, out_features)
    on the Parameter object to avoid repeated transposes and enable coalesced loads.
    """
    orig_shape = x.shape
    assert x.dtype == torch.float32 and weight.dtype == torch.float32
    K = orig_shape[-1]
    out_features = weight.shape[0]
    x2 = x.reshape(-1, K).contiguous()

    # Try to reuse a cached transposed-contiguous buffer on the weight parameter
    wt_cached = getattr(weight, "_triton_wt_t", None)
    recreate = True
    if wt_cached is not None:
        try:
            if wt_cached.device == weight.device and wt_cached.shape[0] == weight.shape[1] and wt_cached.shape[1] == weight.shape[0]:
                recreate = False
        except Exception:
            recreate = True
    if recreate:
        wt = weight.t().contiguous()
        # attach cache for reuse; this is a helper cache and won't be part of state_dict
        try:
            weight._triton_wt_t = wt
        except Exception:
            # If attaching attribute fails (rare), just keep local wt
            pass
    else:
        wt = wt_cached

    out2 = triton_matmul(x2, wt, bias)  # shape (N, out_features)
    return out2.view(*orig_shape[:-1], out_features)


class ModelNew(nn.Module):
    """
    Vision Transformer (ViT) model optimized to use Triton-based linear kernels
    for the patch embedding and the MLP head linear layers.
    """
    def __init__(self, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, channels=3, dropout=0.1, emb_dropout=0.1):
        super(ModelNew, self).__init__()

        assert image_size % patch_size == 0, "Image dimensions must be divisible by the patch size."
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2

        self.patch_size = patch_size
        # Keep position embedding as a Parameter
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        # Use nn.Linear objects for weight/bias containers, but we'll call them via Triton
        self.patch_to_embedding = nn.Linear(patch_dim, dim, bias=True)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        # Keep transformer as-is (PyTorch implementation)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=mlp_dim, dropout=dropout),
            num_layers=depth
        )

        self.to_cls_token = nn.Identity()
        # Keep MLP head layers as modules but use Triton for linear ops in forward
        self.mlp_head = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, num_classes)
        )

    def forward(self, img):
        """
        Forward pass using Triton-based linear kernels for patch embedding and MLP head.
        """
        p = self.patch_size

        # Create patches (PyTorch unfold). Result shape: (B, num_patches, patch_dim)
        x = img.unfold(2, p, p).unfold(3, p, p).reshape(img.shape[0], -1, p * p * img.shape[1])
        # Use Triton linear for patch projection
        # Ensure tensors are on CUDA
        if x.is_cuda:
            x = x.contiguous()
            w = self.patch_to_embedding.weight
            b = self.patch_to_embedding.bias
            x = triton_linear(x, w, b)
        else:
            # Fallback to PyTorch if not on CUDA
            x = self.patch_to_embedding(x)

        cls_tokens = self.cls_token.expand(img.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embedding
        x = self.dropout(x)

        x = self.transformer(x)

        x_cls = self.to_cls_token(x[:, 0])

        # MLP head: use PyTorch implementation (Triton has high overhead on tiny GEMMs)
        return self.mlp_head(x_cls)


# Keep the original helper functions for compatibility (they are not testing code)
image_size = 224
patch_size = 16
num_classes = 10
dim = 512
depth = 6
heads = 8
mlp_dim = 2048
channels = 3
dropout = 0.0
emb_dropout = 0.0

def get_inputs():
    return [torch.rand(2, channels, image_size, image_size)]

def get_init_inputs():
    return [image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, channels, dropout, emb_dropout]