import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

# Autotune configs favoring the Ampere sweet spot for E=128.
# Keep a single, preferred config to avoid chunked fallback for the common E=128 case.
AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK": 128}, num_warps=4, num_stages=1),  # preferred for E=128 on Ampere
]


@triton.autotune(configs=AUTOTUNE_CONFIGS, key=['rows', 'N'])
@triton.jit
def _layernorm_fused_kernel(
    attn_ptr,        # pointer to attn input (rows x N) flattened
    x_ptr,           # pointer to x input (rows x N) flattened
    gamma_ptr,       # pointer to gamma (N,)
    beta_ptr,        # pointer to beta (N,)
    out_ptr,         # pointer to output (rows x N) flattened
    rows,            # number of rows
    N,               # number of columns (embedding dim)
    eps,             # epsilon for numerical stability
    attn_row_stride, # stride between attn rows (in elements)
    x_row_stride,    # stride between x rows (in elements)
    BLOCK: tl.constexpr
):
    """
    Fused kernel that computes v = attn + x, then LayerNorm across last dim.
    Inputs may be stored as fp16 to save bandwidth; the kernel casts loads to fp32
    for accumulation and computation, preserving numerical stability.
    """
    row = tl.program_id(0)
    if row >= rows:
        return

    # Precompute row bases for coalesced loads/stores
    attn_row_base = attn_ptr + row * attn_row_stride
    x_row_base = x_ptr + row * x_row_stride
    out_row_base = out_ptr + row * N

    # Fast path: embedding dimension fits into a single BLOCK -> single-pass algorithm
    if N <= BLOCK:
        offs = tl.arange(0, BLOCK)
        col_idx = offs
        mask = col_idx < N

        # Load and cast to fp32 for stable accumulation (works whether inputs are fp16 or fp32)
        a = tl.cast(tl.load(attn_row_base + col_idx, mask=mask, other=0.0), tl.float32)
        b = tl.cast(tl.load(x_row_base + col_idx, mask=mask, other=0.0), tl.float32)
        v = a + b  # stored in registers

        # single conversion of mask to float and reuse
        mask_f = tl.cast(mask, tl.float32)
        n_valid = tl.sum(mask_f, 0)  # scalar float
        # sum and sumsq across the vector
        sum_v = tl.sum(v * mask_f, 0)
        sumsq_v = tl.sum(v * v * mask_f, 0)

        mean = sum_v / n_valid
        var = sumsq_v / n_valid - mean * mean
        rstd = 1.0 / tl.sqrt(var + eps)

        gamma = tl.cast(tl.load(gamma_ptr + col_idx, mask=mask, other=1.0), tl.float32)
        beta = tl.cast(tl.load(beta_ptr + col_idx, mask=mask, other=0.0), tl.float32)

        y = (v - mean) * rstd * gamma + beta
        tl.store(out_row_base + col_idx, y, mask=mask)
        return

    # Fallback path: chunked two-pass approach for N > BLOCK
    offs = tl.arange(0, BLOCK)

    # First pass: accumulate mean and m2 (Welford-style combination) across chunks
    n_total = 0.0
    mean = 0.0
    m2 = 0.0

    for start in range(0, N, BLOCK):
        col_idx = start + offs
        mask = col_idx < N

        a = tl.cast(tl.load(attn_row_base + col_idx, mask=mask, other=0.0), tl.float32)
        b = tl.cast(tl.load(x_row_base + col_idx, mask=mask, other=0.0), tl.float32)
        v = a + b

        # compute mask_f once per chunk and reuse for sums
        mask_f = tl.cast(mask, tl.float32)
        n_chunk = tl.sum(mask_f, axis=0)
        sum_chunk = tl.sum(v * mask_f, axis=0)
        sumsq_chunk = tl.sum(v * v * mask_f, axis=0)

        if n_chunk > 0.0:
            mean_chunk = sum_chunk / n_chunk
            m2_chunk = sumsq_chunk - n_chunk * mean_chunk * mean_chunk

            if n_total == 0.0:
                mean = mean_chunk
                m2 = m2_chunk
                n_total = n_chunk
            else:
                delta = mean_chunk - mean
                n_comb = n_total + n_chunk
                mean = (n_total * mean + n_chunk * mean_chunk) / n_comb
                m2 = m2 + m2_chunk + delta * delta * (n_total * n_chunk) / n_comb
                n_total = n_comb

    # finalize rstd
    inv_cnt = 1.0 / n_total
    rstd = 1.0 / tl.sqrt(m2 * inv_cnt + eps)

    # Second pass: recompute v per chunk, normalize and write out
    for start in range(0, N, BLOCK):
        col_idx = start + offs
        mask = col_idx < N

        a = tl.cast(tl.load(attn_row_base + col_idx, mask=mask, other=0.0), tl.float32)
        b = tl.cast(tl.load(x_row_base + col_idx, mask=mask, other=0.0), tl.float32)
        v = a + b

        gamma = tl.cast(tl.load(gamma_ptr + col_idx, mask=mask, other=1.0), tl.float32)
        beta = tl.cast(tl.load(beta_ptr + col_idx, mask=mask, other=0.0), tl.float32)

        y = (v - mean) * rstd * gamma + beta
        tl.store(out_row_base + col_idx, y, mask=mask)


def triton_layernorm(attn: torch.Tensor, x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float = 1e-5):
    """
    Fused Triton LayerNorm that computes v = attn + x inside GPU and normalizes across the last dim.
    Expects attn and x shaped (L, N_batch, E) or (rows, E). Returns a tensor shaped (rows, E).
    The kernel accepts inputs stored as fp16 or fp32; it will cast loads to fp32 internally.
    The output tensor is allocated as fp32 so the public interface remains fp32.
    """
    if not attn.is_cuda:
        res = attn + x
        return F.layer_norm(res.view(-1, res.shape[-1]), (res.shape[-1],), weight, bias, eps)

    # Assume weight and bias are created as fp32 and are moved with the module.
    # Avoid per-forward .to(...) copies; users should move the module to the target device once.

    # Make sure inputs are contiguous in memory in the expected layout to enable coalesced loads
    attn = attn.contiguous()
    x = x.contiguous()

    # Flatten inputs to (rows, E)
    if attn.dim() == 3:
        L, N_batch, E = attn.shape
        rows = L * N_batch
        attn_flat = attn.reshape(rows, E)
    else:
        rows, E = attn.shape
        attn_flat = attn.reshape(rows, E)

    if x.dim() == 3:
        x_flat = x.reshape(rows, E)
    else:
        x_flat = x.reshape(rows, E)

    # Allocate output as fp32 to preserve model interface; kernel will write fp32 values.
    out = torch.empty((rows, E), device=attn.device, dtype=torch.float32)

    attn_row_stride = attn_flat.stride(0)
    x_row_stride = x_flat.stride(0)

    grid = (rows,)

    # Launch fused kernel. Autotune will pick BLOCK that usually equals E for best performance.
    _layernorm_fused_kernel[grid](
        attn_flat,              # attn_ptr
        x_flat,                 # x_ptr
        weight,                 # gamma_ptr
        bias,                   # beta_ptr
        out,                    # out_ptr
        rows,                   # rows
        E,                      # N (embedding dim)
        float(eps),             # eps
        attn_row_stride,        # attn_row_stride
        x_row_stride,           # x_row_stride
    )
    return out


class LayerNormTriton(nn.Module):
    """
    nn.Module replacement for LayerNorm that fuses (attn + x) + LayerNorm into a single Triton kernel.
    The forward expects attn and x of identical shape (L, N_batch, E) and returns (rows, E) flattened output.
    """
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(self.normalized_shape, dtype=torch.float32))
        self.bias = nn.Parameter(torch.zeros(self.normalized_shape, dtype=torch.float32))

    def forward(self, attn: torch.Tensor, x: torch.Tensor):
        E = attn.shape[-1]
        weight = self.weight
        bias = self.bias
        # Assume parameters are on the correct device/dtype (fp32). Avoid per-forward copying.
        out_flat = triton_layernorm(attn, x, weight, bias, self.eps)
        return out_flat


class ModelNew(nn.Module):
    def __init__(self, embed_dim, num_heads):
        """
        Attention Block using Multihead Self-Attention with Triton-optimized fused LayerNorm.
        :param embed_dim: Embedding dimension (the number of channels)
        :param num_heads: Number of attention heads
        """
        super(ModelNew, self).__init__()
        # Keep PyTorch's MultiheadAttention (highly optimized) and fuse the residual add + layernorm.
        self.attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.norm = LayerNormTriton(embed_dim)

    def forward(self, x):
        """
        Forward pass of the AttentionBlock.
        :param x: Input tensor of shape (B, C, H, W)
        :return: Output tensor of the same shape (B, C, H, W)
        """
        B, C, H, W = x.shape
        seq_len = H * W

        # reshape to (seq_len, batch_size, embed_dim) as required by MultiheadAttention
        x0 = x.view(B, C, seq_len).permute(2, 0, 1)  # (L, N, E)

        attn_output, _ = self.attn(x0, x0, x0)  # (L, N, E)

        # Make tensors contiguous for coalesced Triton loads
        attn_output = attn_output.contiguous()
        x0 = x0.contiguous()

        # Call Triton LayerNorm kernel directly with fp32 inputs (kernel casts loads internally).
        L, N_batch, E = attn_output.shape

        ln_flat = self.norm(attn_output, x0)  # (rows, E) kernel writes fp32 out
        ln = ln_flat.view(L, N_batch, E)

        out = ln.permute(1, 2, 0).view(B, C, H, W)
        return out


# Defaults and helpers to match the harness expectations
embed_dim = 128
num_heads = 4
batch_size = 2
num_channels = embed_dim
image_height = 128
image_width = 128

def get_inputs():
    return [torch.rand(batch_size, num_channels, image_height, image_width).cuda()]

def get_init_inputs():
    return [embed_dim, num_heads]