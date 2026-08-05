import torch
import torch.nn as nn
import triton
import triton.language as tl

# Autotune configurations for different block sizes / warps
AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK_SIZE": 256}, num_warps=2, num_stages=2),
    triton.Config({"BLOCK_SIZE": 512}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_SIZE": 1024}, num_warps=8, num_stages=3),
]

@triton.autotune(configs=AUTOTUNE_CONFIGS, key=['numel', 'C', 'S_channel'])
@triton.jit
def fused_activations_kernel(
    x_ptr,            # pointer to input tensor (flattened)
    bias_ptr,         # pointer to bias of shape (C,)
    out_ptr,          # pointer to output tensor (flattened)
    numel,            # total number of elements
    C,                # number of channels
    S_channel,        # stride (length) per channel block = D*H*W
    NEG_SLOPE: tl.constexpr,  # leaky relu negative slope (constexpr)
    BLOCK_SIZE: tl.constexpr,
):
    # each program handles a contiguous block of BLOCK_SIZE elements
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offs = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offs < numel

    # Load input values (fp32)
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)

    # Compute channel index for each element:
    # block_index = offs // S_channel  (which enumerates across N*C blocks)
    # channel = block_index % C
    # Use integer arithmetic
    block_idx = offs // S_channel
    channel = block_idx % C

    # Load corresponding bias per channel
    bias = tl.load(bias_ptr + channel, mask=mask, other=0.0)

    # 1) ReLU
    zero = tl.zeros((), dtype=tl.float32)
    x_relu = tl.where(x > zero, x, zero)

    # 2) LeakyReLU with negative slope (after ReLU this is a no-op for negatives,
    #    but we keep it to preserve semantics if inputs are changed)
    x_leaky = tl.where(x_relu > zero, x_relu, x_relu * NEG_SLOPE)

    # 3) GELU approximation: use x * sigmoid(1.702 * x) approximation
    #    sigmoid implemented as 1 / (1 + exp(-z))
    a = 1.702
    z = x_leaky * a
    exp_neg = tl.exp(-z)
    sigmoid_approx = 1.0 / (1.0 + exp_neg)
    gelu_approx = x_leaky * sigmoid_approx

    # 4) Sigmoid on the GELU output
    exp_neg_g = tl.exp(-gelu_approx)
    sigmoid_out = 1.0 / (1.0 + exp_neg_g)

    # 5) Add bias (broadcasted per-channel)
    out = sigmoid_out + bias

    # Store result
    tl.store(out_ptr + offs, out, mask=mask)

def fused_activations(x: torch.Tensor, bias: torch.Tensor, neg_slope: float = 0.01):
    """
    Applies fused: ReLU -> LeakyReLU -> approx-GELU -> Sigmoid -> bias-add
    x: tensor of shape (N, C, D, H, W) contiguous on CUDA
    bias: tensor of shape (C, 1, 1, 1) or (C,) on CUDA
    """
    assert x.is_cuda and bias.is_cuda, "Inputs must be on CUDA"
    # Ensure contiguous
    x = x.contiguous()
    # Flatten input for kernel (view as 1D)
    x_flat = x.view(-1)
    numel = x_flat.numel()

    # Prepare bias as (C,)
    if bias.dim() != 1:
        bias_flat = bias.view(bias.shape[0]).contiguous()
    else:
        bias_flat = bias.contiguous()

    C = x.shape[1]
    # compute S_channel = D*H*W (number of elements per channel per batch)
    D = x.shape[2]
    H = x.shape[3]
    W = x.shape[4]
    S_channel = D * H * W

    out = torch.empty_like(x_flat)

    # grid and launch
    def grid(meta):
        return ((numel + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)

    fused_activations_kernel[grid](
        x_flat,
        bias_flat,
        out,
        numel,
        C,
        S_channel,
        0.01,  # NEG_SLOPE constexpr
    )

    return out.view_as(x)

class ModelNew(nn.Module):
    """
    Optimized Model that uses the original Conv3d but fuses the sequence of
    activations and the bias add into a single Triton kernel for improved throughput.
    """
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super(ModelNew, self).__init__()
        # Keep the original conv (leveraging highly-optimized cuDNN)
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        # bias kept as parameter with original shape; fusion kernel will view it as (C,)
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        # Run conv (kept in PyTorch / cuDNN)
        x = self.conv(x)
        # Apply fused activations + bias via Triton kernel
        return fused_activations(x, self.bias)