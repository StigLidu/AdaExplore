import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

# Autotune configs tuned to prefer warp-friendly tile sizes (multiples of 32 for M/N) and tensor-core-friendly K (32).
AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32},  num_warps=8,  num_stages=2),
    triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 32},  num_warps=16, num_stages=3),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 64,  "BLOCK_K": 32},  num_warps=8,  num_stages=2),
    triton.Config({"BLOCK_M": 64,  "BLOCK_N": 128, "BLOCK_K": 32},  num_warps=8,  num_stages=2),
    triton.Config({"BLOCK_M": 256, "BLOCK_N": 256, "BLOCK_K": 32},  num_warps=16, num_stages=3),
    triton.Config({"BLOCK_M": 64,  "BLOCK_N": 64,  "BLOCK_K": 32},  num_warps=8,  num_stages=2),
]

@triton.autotune(configs=AUTOTUNE_CONFIGS, key=['M', 'N', 'K'])
@triton.jit
def _linear_kernel_packed(
    A_ptr,               # pointer to A (M, K)
    B_ptr,               # pointer to packed B (K, N)  <-- NOTE: packed as transpose of original weight (may be padded)
    C_ptr,               # pointer to output C (M, N_true)  <-- output buffer sized to original N
    bias_ptr,            # pointer to bias (N_padded,) or integer 0 when no bias
    M, N, K,             # dimensions: N here is the padded N used for internal tiling
    stride_am, stride_ak,# strides for A: (stride over M, stride over K)
    stride_bk, stride_bn,# strides for B (K, N): stride when moving along K and N
    stride_cm, stride_cn,# strides for C
    RELU: tl.constexpr,   # whether to apply ReLU
    HAS_BIAS: tl.constexpr,# whether bias_ptr is valid
    N_true,              # original (true) N before padding (used for masking writes & bias loads)
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    """
    Tiled GEMM where B is provided in packed (possibly padded) form (K, N).
    Computes: C = A @ B + bias  where A: (M, K), B: (K, N_padded)
    Fuses optional bias addition and ReLU.
    Uses k-major blocking loading B tiles of shape (BLOCK_K, BLOCK_N) and A tiles (BLOCK_M, BLOCK_K),
    then performs a small inner K-loop to drive FMAs (avoids materializing a large broadcasted temporary).
    Loads are cast to float32 for accumulation (mixed precision safe).
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m_start = pid_m * BLOCK_M
    n_start = pid_n * BLOCK_N

    offs_m = m_start + tl.arange(0, BLOCK_M)   # (BLOCK_M,)
    offs_n = n_start + tl.arange(0, BLOCK_N)   # (BLOCK_N,)

    # accumulator for the tile (in FP32)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # iterate over K dimension in tiles
    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)  # (BLOCK_K,)

        # Load A tile: shape (BLOCK_M, BLOCK_K)
        a_ptrs = A_ptr + (offs_m[:, None] * stride_am) + (offs_k[None, :] * stride_ak)
        mask_a = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        a_tile_raw = tl.load(a_ptrs, mask=mask_a, other=0.0)  # (BLOCK_M, BLOCK_K)
        # cast to FP32 for accumulation (safe whether input is fp16 or fp32)
        a_tile = tl.cast(a_tile_raw, tl.float32)

        # Load B tile from packed B (K, N_padded): shape (BLOCK_K, BLOCK_N)
        b_ptrs = B_ptr + (offs_k[:, None] * stride_bk) + (offs_n[None, :] * stride_bn)
        mask_b = (offs_k[:, None] < K) & (offs_n[None, :] < N)
        b_tile_raw = tl.load(b_ptrs, mask=mask_b, other=0.0)  # (BLOCK_K, BLOCK_N)
        b_tile = tl.cast(b_tile_raw, tl.float32)

        # Vectorized block matmul over K to update the tile at once (avoids Python loop)
        # Use Triton's dot for (BLOCK_M, BLOCK_K) x (BLOCK_K, BLOCK_N) -> (BLOCK_M, BLOCK_N)
        acc += tl.dot(a_tile, b_tile)

    # bias handling (load only the true original bias positions)
    offs_n_mask_true = offs_n < N_true
    if HAS_BIAS:
        bias_vals_raw = tl.load(bias_ptr + offs_n, mask=offs_n_mask_true, other=0.0)  # (BLOCK_N,)
        bias_vals = tl.cast(bias_vals_raw, tl.float32)
    else:
        bias_vals = tl.zeros((BLOCK_N,), dtype=tl.float32)
    out = acc + bias_vals[None, :]

    # fused ReLU
    if RELU:
        zero = tl.zeros((), dtype=tl.float32)
        out = tl.maximum(out, zero)

    # store result with mask (only write the true N columns)
    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm) + (offs_n[None, :] * stride_cn)
    mask_c = (offs_m[:, None] < M) & (offs_n[None, :] < N_true)
    tl.store(c_ptrs, out, mask=mask_c)


def triton_linear_packed(input_: torch.Tensor, B_packed: torch.Tensor, bias: torch.Tensor = None, relu: bool = False):
    """
    Wrapper around the Triton GEMM kernel that expects B_packed to be shape (K, N_padded).
    Computes: output = input_ @ B_packed[:, :N_true] + bias  (input_: M x K, B_packed: K x N_padded)
    Supports fp32 and fp16 packed weights/inputs; kernel will cast loaded values to fp32 for accumulation.
    The wrapper will pad the bias if needed so kernel loads are safe and will pass the true N (N_true)
    so stores only write actual columns.
    """
    assert input_.is_cuda and B_packed.is_cuda, "Tensors must be on CUDA."
    if bias is not None:
        assert bias.is_cuda, "Bias must be on CUDA."

    assert input_.dtype in (torch.float32, torch.float16)
    assert B_packed.dtype in (torch.float32, torch.float16)

    A = input_.contiguous()
    B = B_packed.contiguous()
    bias_t = bias.contiguous() if bias is not None else None

    M, K = A.shape
    # B is packed as (K, N_padded)
    Kb, N_padded = B.shape
    assert Kb == K, f"Packed weight K dim {Kb} does not match input K {K}"

    # infer the true/original N from bias if provided; otherwise assume no padding
    N_true = bias_t.shape[0] if bias_t is not None else N_padded
    # output buffer is sized to the true N (we will only write the true columns)
    out = A.new_empty((M, N_true))

    # strides in elements
    stride_am = A.stride(0)
    stride_ak = A.stride(1)
    stride_bk = B.stride(0)  # stride moving along K in packed (K,N) layout
    stride_bn = B.stride(1)  # stride moving along N in packed layout (likely 1)
    stride_cm = out.stride(0)
    stride_cn = out.stride(1)

    # compute grid using conservative small block sizes (autotune will override meta parameters)
    BLOCK_M_DEFAULT = 64
    BLOCK_N_DEFAULT = 64
    blocks_m = (M + BLOCK_M_DEFAULT - 1) // BLOCK_M_DEFAULT
    blocks_n = (N_padded + BLOCK_N_DEFAULT - 1) // BLOCK_N_DEFAULT
    grid = (blocks_m, blocks_n)

    # prepare bias argument: if bias exists but is smaller than N_padded, pad it so kernel can safely load
    if bias_t is None:
        bias_arg = 0
        has_bias_flag = 0
    else:
        if bias_t.shape[0] != N_padded:
            bias_padded = bias_t.new_zeros((N_padded,))
            bias_padded[:N_true] = bias_t
            bias_arg = bias_padded
        else:
            bias_arg = bias_t
        has_bias_flag = 1

    # Launch kernel: pass N_padded as N and also pass N_true so kernel masks stores/loads properly.
    _linear_kernel_packed[grid](
        A, B, out, bias_arg,
        M, N_padded, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        int(relu),
        int(has_bias_flag),
        N_true
    )
    return out


class ModelNew(nn.Module):
    """
    LeNet-5 optimized for large-batch inference on A6000.

    Optimizations applied:
      - Keep convolutional layers on PyTorch/cuDNN (these are small layers and well-optimized).
      - Replace fully-connected (dense) layers with a Triton-accelerated GEMM kernel that:
          * Expects weights packed as (K, N) (transpose + contiguous) to improve memory access/coalescing.
          * Fuses bias addition and ReLU where applicable.
          * Is autotuned across a set of tile sizes for best performance on the device.
      - Pack (transpose+contiguous) each Linear.weight once and cache it as a buffer; repack only if the weight storage changes.
    """
    def __init__(self, num_classes):
        super(ModelNew, self).__init__()
        # Convolutional layers (leave to PyTorch/cuDNN)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1)

        # Fully connected layers: keep nn.Linear modules for parameter management
        self.fc1 = nn.Linear(in_features=16*5*5, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(in_features=84, out_features=num_classes)

        # Reserve buffers for packed weights (packed as weight.t().contiguous() -> shape (K, N))
        # Initialize to None; they will be populated on first forward (or when weight storage changes).
        self.register_buffer('_packed_fc1', None)
        self.register_buffer('_packed_fc2', None)
        self.register_buffer('_packed_fc3', None)
        # Track original storage pointer to detect parameter updates (e.g., during training)
        self._packed_ptr_fc1 = None
        self._packed_ptr_fc2 = None
        self._packed_ptr_fc3 = None

    def _maybe_pack(self, linear_module: nn.Linear, buf_name: str):
        """
        Ensure that a packed (transposed, contiguous) copy of linear_module.weight exists in the named buffer.
        When packing:
          - transpose and make contiguous
          - cast to fp16 (to enable mixed-precision/tensor-core paths)
          - pad the N dimension (columns) to a small multiple (8) for better memory alignment/coalescing
        If the underlying weight storage changed (e.g., due to reinitialization or training), repack.
        Returns the packed tensor (K, N_padded) on the correct device.
        """
        w = linear_module.weight
        ptr = w.data_ptr()
        PAD_N = 8  # pad N to a small multiple friendly to tensor cores

        # Determine which buffer and ptr to use
        if buf_name == 'fc1':
            if self._packed_ptr_fc1 != ptr or getattr(self, '_packed_fc1') is None:
                packed = w.t().contiguous()
                # cast to fp16 for mixed precision (kernel will cast to fp32 for accumulation)
                packed = packed.to(dtype=torch.float16)
                K, N = packed.shape
                Npad = ((N + PAD_N - 1) // PAD_N) * PAD_N
                if Npad != N:
                    packed_padded = torch.zeros((K, Npad), dtype=packed.dtype, device=packed.device)
                    packed_padded[:, :N] = packed
                    packed = packed_padded
                # store packed tensor as buffer
                object.__setattr__(self, '_packed_fc1', packed)
                self._packed_ptr_fc1 = ptr
            return getattr(self, '_packed_fc1')
        elif buf_name == 'fc2':
            if self._packed_ptr_fc2 != ptr or getattr(self, '_packed_fc2') is None:
                packed = w.t().contiguous()
                packed = packed.to(dtype=torch.float16)
                K, N = packed.shape
                Npad = ((N + PAD_N - 1) // PAD_N) * PAD_N
                if Npad != N:
                    packed_padded = torch.zeros((K, Npad), dtype=packed.dtype, device=packed.device)
                    packed_padded[:, :N] = packed
                    packed = packed_padded
                object.__setattr__(self, '_packed_fc2', packed)
                self._packed_ptr_fc2 = ptr
            return getattr(self, '_packed_fc2')
        elif buf_name == 'fc3':
            if self._packed_ptr_fc3 != ptr or getattr(self, '_packed_fc3') is None:
                packed = w.t().contiguous()
                packed = packed.to(dtype=torch.float16)
                K, N = packed.shape
                Npad = ((N + PAD_N - 1) // PAD_N) * PAD_N
                if Npad != N:
                    packed_padded = torch.zeros((K, Npad), dtype=packed.dtype, device=packed.device)
                    packed_padded[:, :N] = packed
                    packed = packed_padded
                object.__setattr__(self, '_packed_fc3', packed)
                self._packed_ptr_fc3 = ptr
            return getattr(self, '_packed_fc3')
        else:
            # generic fallback (should not happen)
            packed = w.t().contiguous()
            packed = packed.to(dtype=torch.float16)
            K, N = packed.shape
            Npad = ((N + PAD_N - 1) // PAD_N) * PAD_N
            if Npad != N:
                packed_padded = torch.zeros((K, Npad), dtype=packed.dtype, device=packed.device)
                packed_padded[:, :N] = packed
                packed = packed_padded
            return packed

    def forward(self, x):
        # Convolutional feature extraction (use PyTorch/cuDNN)
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        # Flatten to (batch_size, 16*5*5)
        x = x.view(-1, 16*5*5)  # shape (batch_size, 400)

        # fc1: use Triton GEMM with packed weight, fuse ReLU
        packed_w1 = self._maybe_pack(self.fc1, 'fc1')
        x = triton_linear_packed(x, packed_w1, self.fc1.bias, relu=True)

        # fc2: pack if needed and run
        packed_w2 = self._maybe_pack(self.fc2, 'fc2')
        x = triton_linear_packed(x, packed_w2, self.fc2.bias, relu=True)

        # fc3: final linear (no ReLU)
        packed_w3 = self._maybe_pack(self.fc3, 'fc3')
        x = triton_linear_packed(x, packed_w3, self.fc3.bias, relu=False)

        return x