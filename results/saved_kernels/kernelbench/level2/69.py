import torch
import torch.nn as nn
import triton
import triton.language as tl

# Autotune configurations tuned for Ampere (NVIDIA A6000).
# Expanded search space with larger blocks and more warps to better utilize SMs.
AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK_M": 64,  "BLOCK_N": 128,  "BLOCK_K": 32},  num_warps=4,  num_stages=2),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128,  "BLOCK_K": 32},  num_warps=4,  num_stages=2),
    triton.Config({"BLOCK_M": 64,  "BLOCK_N": 256,  "BLOCK_K": 32},  num_warps=8,  num_stages=2),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 256,  "BLOCK_K": 32},  num_warps=8,  num_stages=3),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 512,  "BLOCK_K": 64},  num_warps=8,  num_stages=3),
    triton.Config({"BLOCK_M": 256, "BLOCK_N": 512,  "BLOCK_K": 64},  num_warps=16, num_stages=4),
    triton.Config({"BLOCK_M": 256, "BLOCK_N": 1024, "BLOCK_K": 64},  num_warps=16, num_stages=4),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 1024, "BLOCK_K": 128}, num_warps=16, num_stages=4),
]


@triton.autotune(configs=AUTOTUNE_CONFIGS, key=['M', 'Ncol', 'K'])
@triton.jit
def _conv_fused_kernel(
    A_ptr,           # (M, K) weights, row-major
    x_ptr,           # flattened input images: (batch, image_elems) row-major (per-batch contiguous)
    base_k_ptr,      # (K,) int32 base offsets for each im2col element within a single image
    out_ptr,         # (batch, M, Ncol) output
    bias_ptr,        # (M,)
    M, Ncol, K,      # dims: M = out_channels, Ncol = H_out * W_out, K = C*ks*ks
    H, W, W_out,     # image dims and output width
    image_elems,     # C*H*W (elements per image)
    lda, ldc,        # leading dims for A (K) and out (Ncol)
    x_batch_stride, out_batch_stride, batch_count,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    """
    Triton kernel that computes convolution by reading patches from the original input
    (avoiding an explicit im2col buffer), and fusing bias + hardswish + relu.
    Grid: (ceil(M/BLOCK_M), ceil(Ncol/BLOCK_N), batch_count)
    Each program computes a BLOCK_M x BLOCK_N tile of the output for one batch image.
    """
    row_block = tl.program_id(0)  # output-channel block
    col_block = tl.program_id(1)  # output-spatial block
    batch = tl.program_id(2)      # batch index

    # Offsets for this program
    row_offsets = row_block * BLOCK_M + tl.arange(0, BLOCK_M)           # (BLOCK_M,)
    col_offsets = col_block * BLOCK_N + tl.arange(0, BLOCK_N)           # (BLOCK_N,)

    # Masks for bounds
    row_mask = row_offsets < M                                            # (BLOCK_M,)
    col_mask = col_offsets < Ncol                                         # (BLOCK_N,)

    # Initialize accumulator in fp32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Per-batch base offsets (in elements)
    x_batch_offset = batch * x_batch_stride
    out_batch_offset = batch * out_batch_stride

    # Compute per-column contribution: for each output column index (linear), compute input row/col add
    # col_offsets -> oh = col // W_out ; ow = col % W_out; inp_add = oh * W + ow
    # Use integer arithmetic
    col_offsets_clamped = col_offsets  # masking later
    oh = col_offsets_clamped // W_out   # (BLOCK_N,)
    ow = col_offsets_clamped - oh * W_out  # (BLOCK_N,)
    inp_add = oh * W + ow                # (BLOCK_N,)

    # Loop over K in chunks of BLOCK_K
    for k_start in range(0, K, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)                        # (BLOCK_K,)
        k_mask = k_offsets < K

        # Load tile of A (weights): shape (BLOCK_M, BLOCK_K)
        a_ptrs = A_ptr + (row_offsets[:, None] * lda + k_offsets[None, :])
        a_mask = row_mask[:, None] & k_mask[None, :]
        a_tile = tl.load(a_ptrs, mask=a_mask, other=0.0)
        # Cast input tiles to fp32 for accumulation (allow launching with fp16 operands)
        a_tile = tl.cast(a_tile, tl.float32)

        # Load base offsets for this k-block: (BLOCK_K,) integers
        base_offsets = tl.load(base_k_ptr + k_offsets, mask=k_mask, other=0)  # int offsets

        # Expand base_offsets and inp_add to form pointers into flattened input image
        # b_ptrs shape: (BLOCK_K, BLOCK_N)
        b_ptrs = x_ptr + x_batch_offset + (base_offsets[:, None] + inp_add[None, :])

        # Masks for loading B tile
        b_mask = k_mask[:, None] & col_mask[None, :]
        # Load B tile (input patches)
        b_tile = tl.load(b_ptrs, mask=b_mask, other=0.0)
        b_tile = tl.cast(b_tile, tl.float32)

        # Accumulate using dot in fp32 (a_tile: BLOCK_M x BLOCK_K, b_tile: BLOCK_K x BLOCK_N)
        acc += tl.dot(a_tile, b_tile)

    # Add bias: load bias for rows and broadcast
    bias_vals = tl.load(bias_ptr + row_offsets, mask=row_mask, other=0.0)  # (BLOCK_M,)
    acc = acc + bias_vals[:, None]

    # Fused activation: HardSwish followed by ReLU
    # Since final ReLU zeros negatives, we can short-circuit negative inputs before HardSwish
    acc_pos = tl.where(acc > 0.0, acc, 0.0)
    tmp = acc_pos + 3.0
    tmp = tl.minimum(tl.maximum(tmp, 0.0), 6.0)
    hs = acc_pos * (tmp / 6.0)  # (BLOCK_M, BLOCK_N)
    out_block = hs  # already non-negative due to acc_pos

    # Store result
    store_mask = row_mask[:, None] & col_mask[None, :]
    c_ptrs = out_ptr + out_batch_offset + (row_offsets[:, None] * ldc + col_offsets[None, :])
    tl.store(c_ptrs, out_block, mask=store_mask)


def triton_fused_conv_direct(A: torch.Tensor, x_flat: torch.Tensor, base_k: torch.Tensor, bias: torch.Tensor, out_all: torch.Tensor, H: int, W: int, W_out: int):
    """
    Wrapper to launch the Triton convolution kernel:
    - A: (M, K) weight matrix (contiguous)
    - x_flat: (batch, C*H*W) flattened input images (contiguous)
    - base_k: (K,) int32 base offsets for each im2col element within a single image
    - bias: (M,)
    - out_all: (batch, M, Ncol) preallocated contiguous output
    - H, W, W_out: image dimensions (integers)
    """
    assert A.is_cuda and x_flat.is_cuda and base_k.is_cuda and out_all.is_cuda and bias.is_cuda
    # Allow weights and inputs to be float16 (for mixed precision) or float32.
    assert A.dtype in (torch.float16, torch.float32) and x_flat.dtype in (torch.float16, torch.float32)
    # Output buffer and bias must be float32 because kernel accumulates and stores fp32.
    assert out_all.dtype == torch.float32 and bias.dtype == torch.float32
    assert base_k.dtype == torch.int32

    batch, image_elems = x_flat.shape
    M, K = A.shape
    _, M_check, Ncol = out_all.shape
    assert M_check == M

    # Make sure tensors are contiguous
    A = A.contiguous()
    x_flat = x_flat.contiguous()
    base_k = base_k.contiguous()
    bias = bias.contiguous()
    out_all = out_all.contiguous()

    lda = K
    ldc = Ncol

    x_batch_stride = image_elems
    out_batch_stride = M * ldc

    def grid(meta):
        return (
            (M + meta["BLOCK_M"] - 1) // meta["BLOCK_M"],
            (Ncol + meta["BLOCK_N"] - 1) // meta["BLOCK_N"],
            batch
        )

    _conv_fused_kernel[grid](
        A, x_flat, base_k, out_all, bias,
        M, Ncol, K,
        H, W, W_out,
        image_elems,
        lda, ldc,
        x_batch_stride, out_batch_stride, batch
    )


class ModelNew(nn.Module):
    """
    ModelNew implements a direct Triton convolution kernel that:
      - avoids an explicit im2col buffer by reading input patches on-the-fly,
      - fuses bias + hardswish + relu,
      - reduces memory bandwidth and launch overhead by batching images in the kernel grid.
    """
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ModelNew, self).__init__()
        conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        # store weight and bias as parameters (same layout as Conv2d)
        self.weight = nn.Parameter(conv.weight.detach().clone())  # (out_channels, in_channels, k, k)
        if conv.bias is not None:
            self.bias = nn.Parameter(conv.bias.detach().clone())
        else:
            self.bias = nn.Parameter(torch.zeros(out_channels, dtype=torch.float32))

        # Cache for base_k offsets. We'll initialize lazily on first forward
        # to avoid repeated CPU allocations for different devices / input sizes.
        self.register_buffer("base_k", torch.empty(0, dtype=torch.int32))

    def forward(self, x):
        """
        x: (batch_size, in_channels, H, W)  (must be on CUDA, float32)
        returns: (batch_size, out_channels, H_out, W_out)
        """
        assert x.dtype == torch.float32 and x.is_cuda, "Input must be float32 CUDA tensor."

        N, C, H, W = x.shape
        ks = self.kernel_size
        assert C == self.in_channels
        H_out = H - ks + 1
        W_out = W - ks + 1
        Ncol = H_out * W_out
        M = self.out_channels
        K = C * ks * ks

        # Prepare flattened weight A: (M, K)
        A = self.weight.view(M, -1).contiguous()

        # Prepare flattened input images: (N, C*H*W)
        x_contig = x.contiguous()
        x_flat = x_contig.view(N, -1).contiguous()  # (N, image_elems)
        image_elems = C * H * W

        # Prepare base_k offsets (int32) for each k in [0..K)
        # base_k[k] = c*H*W + kh*W + kw  (offset within a single image)
        # Build lazily and cache as a buffer to avoid repeated allocations on CPU/GPU.
        if self.base_k.numel() != K or self.base_k.device != x.device:
            base_offsets = []
            for c in range(C):
                base_c = c * (H * W)
                for kh in range(ks):
                    base_kh = kh * W
                    for kw in range(ks):
                        base_offsets.append(base_c + base_kh + kw)
            base_k = torch.tensor(base_offsets, dtype=torch.int32, device=x.device).contiguous()
            # store as buffer on the module so subsequent forwards reuse it
            self.base_k = base_k
        else:
            base_k = self.base_k

        # Output buffer (N, M, Ncol)
        out_all = x.new_empty((N, M, Ncol))

        # Ensure bias tensor
        bias = self.bias
        if bias is None:
            bias = x.new_zeros((M,))
        else:
            bias = bias.contiguous()

        # Launch Triton kernel
        triton_fused_conv_direct(A, x_flat, base_k, bias, out_all, H, W, W_out)

        # Reshape output back to (N, M, H_out, W_out)
        out = out_all.view(N, M, H_out, W_out)
        return out