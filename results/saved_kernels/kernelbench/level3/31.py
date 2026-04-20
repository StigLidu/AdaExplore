import torch
import torch.nn as nn
import triton
import triton.language as tl

# Autotune configurations tuned for NVIDIA A6000 (Ampere).
# Larger num_warps and stages help throughput for full-row kernels.
AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK": 128}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK": 128}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK": 64},  num_warps=4, num_stages=2),
    triton.Config({"BLOCK": 64},  num_warps=2, num_stages=2),
    triton.Config({"BLOCK": 32},  num_warps=2, num_stages=2),
]

@triton.jit
def _fused_add_ln_128_kernel(
    a_ptr, b_ptr, out_ptr, weight_ptr, bias_ptr,
    ROWS, D, eps,
    BLOCK: tl.constexpr, ROWS_PER_PROG: tl.constexpr
):
    """
    Specialized fast path for embed_dim == 128.
    Each program handles ROWS_PER_PROG full rows of length 128.
    Inputs a/b are flattened from (S, B, E) => contiguous rows of length E.
    The kernel writes the normalized result into out_ptr (same layout as inputs: flattened rows).
    """
    row0 = tl.program_id(0) * ROWS_PER_PROG
    offs = tl.arange(0, BLOCK)  # BLOCK expected to be 128

    # Hoist weight/bias loads to encourage vectorized reads (full-width, unmasked)
    w = tl.load(weight_ptr + offs)
    bias = tl.load(bias_ptr + offs)

    # Handle multiple rows per program to amortize launch overhead and improve locality
    for r in range(ROWS_PER_PROG):
        row = row0 + r
        if row < ROWS:
            base = row * D
            idxs = base + offs

            a_vals = tl.load(a_ptr + idxs)
            b_vals = tl.load(b_ptr + idxs)
            vals = a_vals + b_vals

            s = tl.sum(vals, 0)
            s2 = tl.sum(vals * vals, 0)

            mean = s / D
            invstd = 1.0 / tl.sqrt(s2 / D - mean * mean + eps)

            normalized = (vals - mean) * invstd
            out = normalized * w + bias

            tl.store(out_ptr + idxs, out)


@triton.autotune(
    configs=AUTOTUNE_CONFIGS,
    key=['D', 'ROWS']
)
@triton.jit
def _fused_add_ln_kernel(
    a_ptr, b_ptr, out_ptr, weight_ptr, bias_ptr,
    ROWS, D, eps,
    BLOCK: tl.constexpr
):
    """
    Generic fused (a + b) -> LayerNorm over last dimension (D).
    Each program handles one row but processes the row in tiles of size BLOCK.
    """
    row = tl.program_id(0)
    if row >= ROWS:
        return

    row_offset = row * D
    offs = tl.arange(0, BLOCK)

    # First pass: accumulate sum and sum of squares
    s = 0.0
    s2 = 0.0
    for off in range(0, D, BLOCK):
        idxs = row_offset + off + offs
        mask = offs < (D - off)
        a_vals = tl.load(a_ptr + idxs, mask=mask, other=0.0)
        b_vals = tl.load(b_ptr + idxs, mask=mask, other=0.0)
        vals = a_vals + b_vals
        s += tl.sum(vals, 0)
        s2 += tl.sum(vals * vals, 0)

    mean = s / D
    invstd = 1.0 / tl.sqrt(s2 / D - mean * mean + eps)

    # Second pass: normalize, apply affine transform, store
    for off in range(0, D, BLOCK):
        idxs = row_offset + off + offs
        mask = offs < (D - off)
        a_vals = tl.load(a_ptr + idxs, mask=mask, other=0.0)
        b_vals = tl.load(b_ptr + idxs, mask=mask, other=0.0)
        vals = a_vals + b_vals
        normalized = (vals - mean) * invstd
        w = tl.load(weight_ptr + off + offs, mask=mask, other=1.0)
        bias = tl.load(bias_ptr + off + offs, mask=mask, other=0.0)
        out = normalized * w + bias
        tl.store(out_ptr + idxs, out, mask=mask)


def triton_fused_add_layernorm(a: torch.Tensor, b: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float):
    """
    Fused elementwise add (a + b) followed by LayerNorm over the last dimension.
    a, b: (seq_len, batch, embed_dim), float32 CUDA tensors (contiguous or not)
    weight, bias: (embed_dim,) float32 tensors or None
    Returns: tensor of same shape as a (contiguous)
    """
    assert a.device.type == 'cuda' and b.device.type == 'cuda', "Triton fused kernel requires CUDA tensors"
    assert a.dtype == torch.float32 and b.dtype == torch.float32, "Only float32 supported"

    seq_len, batch, embed_dim = a.shape
    N = seq_len * batch
    device = a.device
    dtype = a.dtype

    if weight is None:
        weight = torch.ones(embed_dim, device=device, dtype=dtype)
    if bias is None:
        bias = torch.zeros(embed_dim, device=device, dtype=dtype)
    weight = weight.contiguous()
    bias = bias.contiguous()

    # Ensure inputs are contiguous flattened rows: shape (N, D) flattened -> 1D
    a_flat = a.contiguous().view(-1)
    b_flat = b.contiguous().view(-1)
    out_flat = torch.empty_like(a_flat)

    # Fast path specialized for D == 128 (common in this benchmark)
    if embed_dim == 128:
        BLOCK = 128
        ROWS_PER_PROG = 4  # each program will process 4 rows -> fewer, larger programs
        # ensure weight/bias contiguous for vectorized loads (backend can emit wide ops)
        weight = weight.contiguous()
        bias = bias.contiguous()
        # coarser grid with fewer programs (each program handles ROWS_PER_PROG rows)
        grid = ((N + ROWS_PER_PROG - 1) // ROWS_PER_PROG,)
        _fused_add_ln_128_kernel[grid](
            a_flat, b_flat, out_flat, weight, bias,
            N, embed_dim, float(eps),
            BLOCK=BLOCK, ROWS_PER_PROG=ROWS_PER_PROG
        )
        out = out_flat.view(seq_len, batch, embed_dim)
        return out

    # Generic autotuned kernel
    grid = (N,)
    _fused_add_ln_kernel[grid](
        a_flat, b_flat, out_flat, weight, bias,
        N, embed_dim, float(eps)
    )
    out = out_flat.view(seq_len, batch, embed_dim)
    return out


class ModelNew(nn.Module):
    """
    MultiheadAttention block with a Triton-fused residual add + LayerNorm kernel.
    Uses a specialized fast path for embed_dim == 128 and autotuned configs for other sizes.
    """
    def __init__(self, embed_dim, num_heads):
        super(ModelNew, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        """
        x: (B, C, H, W)
        returns: (B, C, H, W)
        """
        B, C, H, W = x.shape
        seq_len = H * W

        # Reshape to (S, B, E)
        x_perm = x.view(B, C, seq_len).permute(2, 0, 1)
        if not x_perm.is_contiguous():
            x_perm = x_perm.contiguous()

        attn_output, _ = self.attn(x_perm, x_perm, x_perm)  # (S, B, E)

        # If not on CUDA, fallback to standard LayerNorm
        if attn_output.device.type != 'cuda':
            out = self.norm(attn_output + x_perm)
            out = out.permute(1, 2, 0).contiguous().view(B, C, H, W)
            return out

        # Use Triton fused kernel for residual add + layernorm
        attn_contig = attn_output if attn_output.is_contiguous() else attn_output.contiguous()
        res = triton_fused_add_layernorm(attn_contig, x_perm, self.norm.weight, self.norm.bias, self.norm.eps)

        out = res.permute(1, 2, 0).contiguous().view(B, C, H, W)
        return out


# Model hyperparameters and helpers kept for compatibility
embed_dim = 128
num_heads = 4
batch_size = 2
num_channels = embed_dim
image_height = 128
image_width = 128

def get_inputs():
    if torch.cuda.is_available():
        return [torch.rand(batch_size, num_channels, image_height, image_width, device='cuda', dtype=torch.float32)]
    else:
        return [torch.rand(batch_size, num_channels, image_height, image_width, dtype=torch.float32)]

def get_init_inputs():
    return [embed_dim, num_heads]