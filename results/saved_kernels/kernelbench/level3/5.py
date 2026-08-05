import torch
import torch.nn as nn
import triton
import triton.language as tl

# Autotune configurations tuned for NVIDIA A6000 / Ampere
AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 32}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 256, "BLOCK_N": 256, "BLOCK_K": 32}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64}, num_warps=8, num_stages=3),
]


@triton.autotune(
    configs=AUTOTUNE_CONFIGS,
    key=["M", "N", "K", "apply_relu", "bias_present"],
)
@triton.jit
def _gemm_bias_relu_kernel(
    A_ptr,          # pointer to A (row-major) shape (M, K)
    B_ptr,          # pointer to B (row-major) shape (K, N) -- B is expected pre-transposed when used via wrapper
    C_ptr,          # pointer to output C (row-major) shape (M, N) (fp32)
    bias_ptr,       # pointer to bias (N,) or empty tensor
    M, N, K,
    stride_am, stride_ak,  # strides for A
    stride_bk, stride_bn,  # strides for B
    stride_cm, stride_cn,  # strides for C
    apply_relu,     # int flag 0/1
    bias_present,   # int flag 0/1
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m_start = pid_m * BLOCK_M
    n_start = pid_n * BLOCK_N

    offs_m = m_start + tl.arange(0, BLOCK_M)   # [BLOCK_M]
    offs_n = n_start + tl.arange(0, BLOCK_N)   # [BLOCK_N]

    # accumulator in fp32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension
    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)  # [BLOCK_K]

        # A block: A[offs_m[:,None], offs_k[None,:]]
        A_idx = offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        # B block: B[offs_k[:,None], offs_n[None,:]]
        B_idx = offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

        mask_a = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        mask_b = (offs_k[:, None] < K) & (offs_n[None, :] < N)

        a = tl.load(A_ptr + A_idx, mask=mask_a, other=0.0)
        b = tl.load(B_ptr + B_idx, mask=mask_b, other=0.0)

        # Use dot to leverage Tensor Cores where possible; inputs may be fp16 or fp32
        acc += tl.dot(a, b)

    # Add bias if present
    if bias_present != 0:
        bvals = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
        acc = acc + bvals[None, :]

    # Apply ReLU in fp32 accumulator if requested
    if apply_relu == 1:
        acc = tl.maximum(acc, 0.0)

    # Write back to C (fp32)
    C_idx = offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask_c = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(C_ptr + C_idx, acc, mask=mask_c)


def triton_linear_fast(x: torch.Tensor, weight_t: torch.Tensor, bias: torch.Tensor = None, apply_relu: bool = False):
    """
    x: (M, K) activations
    weight_t: pre-transposed weight tensor with shape (K, N). Ideally fp16 contiguous.
    bias: (N,) tensor or None
    Returns: output tensor of dtype float32 (accumulated in fp32)
    """
    assert x.is_cuda and weight_t.is_cuda, "Tensors must be on CUDA."

    x = x.contiguous()
    M, K = x.shape
    K_b, N = weight_t.shape
    assert K == K_b, f"Incompatible shapes for matmul: x K={K}, weight_t rows={K_b}"

    # Choose input dtype to favor Tensor Cores when possible
    if weight_t.dtype == torch.float16:
        x_in = x.half()
    else:
        x_in = x

    # Prepare output as fp32 for numeric stability and API compatibility
    out = torch.empty((M, N), device=x.device, dtype=torch.float32)

    # Bias handling: always pass a tensor pointer; convert to the same dtype as A for kernel loads
    if bias is None:
        bias_ptr = torch.empty((0,), device=x.device, dtype=x_in.dtype)
        bias_present = 0
    else:
        bias_ptr = bias.contiguous().to(dtype=x_in.dtype)
        bias_present = 1

    # Strides (row-major contiguous)
    stride_am = x_in.stride(0)
    stride_ak = x_in.stride(1)
    stride_bk = weight_t.stride(0)
    stride_bn = weight_t.stride(1)
    stride_cm = out.stride(0)
    stride_cn = out.stride(1)

    # Launch grid determined by autotune parameters
    def grid(meta):
        bm = meta["BLOCK_M"]
        bn = meta["BLOCK_N"]
        return ( (M + bm - 1) // bm, (N + bn - 1) // bn )

    # Call the autotuned Triton kernel
    _gemm_bias_relu_kernel[grid](
        x_in, weight_t, out, bias_ptr,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        1 if apply_relu else 0,
        bias_present,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        """
        Optimized Model using Triton-accelerated fully-connected layers.
        Convolutional backbone left to PyTorch (highly optimized).
        """
        super(ModelNew, self).__init__()

        # Convolutional backbone (same as original)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=2)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, padding=2)
        self.relu2 = nn.ReLU(inplace=True)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv3 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU(inplace=True)

        self.conv4 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1)
        self.relu4 = nn.ReLU(inplace=True)

        self.conv5 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1)
        self.relu5 = nn.ReLU(inplace=True)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2)

        # FC layers kept for parameter storage
        self.fc1 = nn.Linear(in_features=256 * 6 * 6, out_features=4096)
        self.relu6 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout(p=0.0)

        self.fc2 = nn.Linear(in_features=4096, out_features=4096)
        self.relu7 = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout(p=0.0)

        self.fc3 = nn.Linear(in_features=4096, out_features=num_classes)

        # Lazy cached pre-transposed weights in fp16 (for Triton)
        self.register_buffer("_fc1_w_t", torch.empty(0), persistent=False)
        self.register_buffer("_fc2_w_t", torch.empty(0), persistent=False)
        self.register_buffer("_fc3_w_t", torch.empty(0), persistent=False)

    def _ensure_weights_cached(self, device):
        # Create cached pre-transposed weights (fp16 contiguous) lazily and place them on the right device
        if self._fc1_w_t.numel() == 0 or self._fc1_w_t.device != device:
            # weight is (out_features, in_features); we want (in_features, out_features)
            self._fc1_w_t = self.fc1.weight.t().contiguous().half().to(device)
        if self._fc2_w_t.numel() == 0 or self._fc2_w_t.device != device:
            self._fc2_w_t = self.fc2.weight.t().contiguous().half().to(device)
        if self._fc3_w_t.numel() == 0 or self._fc3_w_t.device != device:
            self._fc3_w_t = self.fc3.weight.t().contiguous().half().to(device)

    def forward(self, x):
        # Convolutional backbone using PyTorch
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)

        x = self.conv3(x)
        x = self.relu3(x)

        x = self.conv4(x)
        x = self.relu4(x)

        x = self.conv5(x)
        x = self.relu5(x)
        x = self.maxpool3(x)

        x = torch.flatten(x, 1)

        # If inputs are on CUDA, use Triton-accelerated FCs
        if x.is_cuda:
            device = x.device
            # Lazy cache pre-transposed fp16 weights on the correct device
            self._ensure_weights_cached(device)

            # fc1: x @ W1.T + b1  -> ReLU
            out1 = triton_linear_fast(x, self._fc1_w_t, self.fc1.bias, apply_relu=True)

            # fc2: out1 @ W2.T + b2 -> ReLU
            out2 = triton_linear_fast(out1, self._fc2_w_t, self.fc2.bias, apply_relu=True)

            # fc3: final linear
            out3 = triton_linear_fast(out2, self._fc3_w_t, self.fc3.bias, apply_relu=False)

            # Return fp32 tensor (out3 already fp32)
            x = out3
        else:
            # CPU fallback using PyTorch layers
            x = self.fc1(x)
            x = self.relu6(x)
            x = self.dropout1(x)

            x = self.fc2(x)
            x = self.relu7(x)
            x = self.dropout2(x)

            x = self.fc3(x)

        return x