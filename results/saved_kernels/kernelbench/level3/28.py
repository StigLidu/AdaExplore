import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def _matmul_kernel(
    A_ptr, B_ptr, C_ptr, bias_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    """
    Blocked GEMM kernel computing C[M,N] += A[M,K] @ B[K,N] with fused bias add.
    A_ptr: pointer to A (M, K)
    B_ptr: pointer to weight in native layout (N, K). The kernel indexes it as W[n, k]
    C_ptr: pointer to output C (M, N)
    bias_ptr: pointer to bias (N,)
    Strides are in elements (not bytes).
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m_start = pid_m * BLOCK_M
    n_start = pid_n * BLOCK_N

    m_range = m_start + tl.arange(0, BLOCK_M)
    n_range = n_start + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # iterate over K in blocks
    k = 0
    while k < K:
        kk = k + tl.arange(0, BLOCK_K)

        # compute addresses
        a_ptrs = A_ptr + (m_range[:, None] * stride_am) + (kk[None, :] * stride_ak)
        # B_ptr points to weight in (N, K) layout, we want logical b[k, n] = W[n, k]
        # so address for b[k, n] is B_ptr + (n * stride_bn) + (k * stride_bk)
        b_ptrs = B_ptr + (kk[:, None] * stride_bk) + (n_range[None, :] * stride_bn)

        # masks
        mask_a = (m_range[:, None] < M) & (kk[None, :] < K)
        mask_b = (kk[:, None] < K) & (n_range[None, :] < N)

        a = tl.load(a_ptrs, mask=mask_a, other=0.0)
        b = tl.load(b_ptrs, mask=mask_b, other=0.0)

        # accumulation: a is (BLOCK_M, BLOCK_K), b is (BLOCK_K, BLOCK_N)
        acc += tl.dot(a, b)

        k += BLOCK_K

    # Fuse bias add: load bias vector for the block of N and broadcast across rows
    n_mask = n_range < N
    bias_vals = tl.load(bias_ptr + n_range, mask=n_mask, other=0.0)  # (BLOCK_N,)
    acc = acc + bias_vals[None, :]

    c_ptrs = C_ptr + (m_range[:, None] * stride_cm) + (n_range[None, :] * stride_cn)
    mask_c = (m_range[:, None] < M) & (n_range[None, :] < N)
    tl.store(c_ptrs, acc, mask=mask_c)


def triton_linear(input_2d: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor = None):
    """
    Compute input_2d @ weight.T + bias using the Triton matmul kernel.
    input_2d: (M, K)
    weight: (out_features, K)  # native layout (N, K), no per-forward transpose
    returns: (M, out_features)
    """
    assert input_2d.is_cuda and weight.is_cuda, "Tensors must be on CUDA."
    assert input_2d.dtype == torch.float32 and weight.dtype == torch.float32
    # Make contiguous 2D tensors for A; weight we keep in its native layout to avoid copies
    A = input_2d.contiguous()
    W = weight  # (N, K) in native layout
    M, K = A.shape
    N, Kb = W.shape
    assert K == Kb, "Incompatible shapes for matmul"

    C = torch.empty((M, N), device=A.device, dtype=A.dtype)

    # get strides in elements (not bytes)
    stride_am = A.stride(0)
    stride_ak = A.stride(1)
    # Since W is (N, K) and kernel indexes it as W[n, k], use these strides:
    stride_bk = W.stride(1)  # stride along K dimension (fastest varying in typical row-major)
    stride_bn = W.stride(0)  # stride along N dimension
    stride_cm = C.stride(0)
    stride_cn = C.stride(1)

    # Prepare bias tensor (avoid calling extra kernels later). If no bias provided, pass zeros.
    if bias is not None:
        bias_tensor = bias.contiguous()
    else:
        bias_tensor = torch.zeros((N,), device=A.device, dtype=A.dtype)

    # Choose block sizes tuned for Ampere / this problem size
    BLOCK_M = 64
    BLOCK_N = 128
    BLOCK_K = 32

    grid = ( (M + BLOCK_M - 1) // BLOCK_M, (N + BLOCK_N - 1) // BLOCK_N )

    _matmul_kernel[grid](
        A, W, C, bias_tensor,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K
    )

    return C


class ModelNew(nn.Module):
    def __init__(self, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, channels=3, dropout=0.1, emb_dropout=0.1):
        super(ModelNew, self).__init__()

        assert image_size % patch_size == 0, "Image dimensions must be divisible by the patch size."
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2

        self.patch_size = patch_size
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        # Keep a linear layer object to hold weights/bias for easy parameter management,
        # but we'll call it using the Triton kernel in the forward pass.
        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=mlp_dim, dropout=dropout),
            num_layers=depth
        )

        self.to_cls_token = nn.Identity()
        self.mlp_head = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, num_classes)
        )

    def forward(self, img):
        """
        Forward pass with a Triton-accelerated linear projection for patch embeddings.
        img: (batch_size, channels, image_size, image_size)
        """
        p = self.patch_size

        # Extract patches (same as original)
        # x shape: (batch, num_patches, patch_dim)
        x = img.unfold(2, p, p).unfold(3, p, p).reshape(img.shape[0], -1, p*p*img.shape[1])

        B, P, K = x.shape  # K == patch_dim
        # Merge batch and patches to form 2D input for Triton linear: (M, K) where M = B * P
        M = B * P
        x_flat = x.contiguous().view(M, K)

        # Use Triton-backed linear: x_flat @ W.T + b
        W = self.patch_to_embedding.weight  # shape (dim, K)
        b = self.patch_to_embedding.bias
        embedded_flat = triton_linear(x_flat, W, b)  # (M, dim)

        # reshape back to (batch, num_patches, dim)
        embedded = embedded_flat.view(B, P, -1)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, embedded), dim=1)
        x = x + self.pos_embedding
        x = self.dropout(x)

        # Transformer forward (keeps original behavior)
        x = self.transformer(x)

        x = self.to_cls_token(x[:, 0])
        return self.mlp_head(x)