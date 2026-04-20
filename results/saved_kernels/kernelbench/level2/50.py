import torch
import torch.nn as nn
import triton
import triton.language as tl

# Autotune configs for the fused kernel
AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK": 256}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK": 512}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK": 1024}, num_warps=8, num_stages=3),
]

@triton.autotune(configs=AUTOTUNE_CONFIGS, key=['n_elements_out'])
@triton.jit
def fused_pool_bias_kernel(
    inp_ptr,        # pointer to input tensor (conv_transpose output), shape [N, C, D_in, H_in, W_in]
    bias_ptr,       # pointer to bias tensor, shape [C]
    out_ptr,        # pointer to output tensor, shape [N, C, D_out, H_out, W_out]
    N, C, D_in, H_in, W_in, D_out, H_out, W_out,
    scale_combined, # float: scale1 * scale2 / 8.0
    bias_scale,     # float: scale2
    n_elements_out, # total number of output elements = N*C*D_out*H_out*W_out
    BLOCK: tl.constexpr
):
    """
    Each program handles a contiguous block of up to BLOCK output elements.
    For each output element, performs 3D avg-pool over kernel 2x2x2, fused with:
      out = (avg_pool(input_region) * scale1 + bias[c]) * scale2
    This is computed as:
      out = sum(input_region) * (scale1*scale2/8) + bias[c] * scale2
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK
    offs = block_start + tl.arange(0, BLOCK)
    mask = offs < n_elements_out

    # Compute multi-dimensional indices from flattened output index:
    # layout: (((n * C + c) * D_out + d) * H_out + h) * W_out + w
    idx = offs

    w = idx % W_out
    idx = idx // W_out
    h = idx % H_out
    idx = idx // H_out
    d = idx % D_out
    idx = idx // D_out
    c = idx % C
    n = idx // C

    # convert to input coordinates (stride=2, kernel=2)
    d0 = d * 2
    h0 = h * 2
    w0 = w * 2

    # Compute flattened base input index: (((n*C + c)*D_in + d0)*H_in + h0)*W_in + w0
    nC = n * C
    nc_plus_c = nC + c
    base = (((nc_plus_c * D_in + d0) * H_in + h0) * W_in + w0)

    # precompute deltas for the 8 kernel points (kd,kh,kw in {0,1})
    HW = H_in * W_in
    delta0 = 0
    delta1 = 1
    delta2 = W_in
    delta3 = W_in + 1
    delta4 = HW
    delta5 = HW + 1
    delta6 = HW + W_in
    delta7 = HW + W_in + 1

    # Load 8 values and accumulate. Use the same mask for all loads; when an output slot is out-of-range
    # the mask disables the load and 'other' provides safe value.
    a0 = tl.load(inp_ptr + base + delta0, mask=mask, other=0.0)
    a1 = tl.load(inp_ptr + base + delta1, mask=mask, other=0.0)
    a2 = tl.load(inp_ptr + base + delta2, mask=mask, other=0.0)
    a3 = tl.load(inp_ptr + base + delta3, mask=mask, other=0.0)
    a4 = tl.load(inp_ptr + base + delta4, mask=mask, other=0.0)
    a5 = tl.load(inp_ptr + base + delta5, mask=mask, other=0.0)
    a6 = tl.load(inp_ptr + base + delta6, mask=mask, other=0.0)
    a7 = tl.load(inp_ptr + base + delta7, mask=mask, other=0.0)

    s = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7

    # Load bias for each channel (broadcast across the BLOCK entries)
    b = tl.load(bias_ptr + c, mask=mask, other=0.0)

    # Compute final output: (sum * (scale1/8) + bias) * scale2
    out = s * scale_combined + b * bias_scale

    # Store results
    tl.store(out_ptr + offs, out, mask=mask)


def fused_pool_bias(inp: torch.Tensor, bias: torch.Tensor, scale1: float, scale2: float):
    """
    Wrapper that prepares tensors and launches the Triton kernel.
    inp: [N, C, D_in, H_in, W_in] (torch.cuda.FloatTensor)
    bias: [C] or [C,1,1,1] (torch.cuda.FloatTensor)
    Returns: output tensor [N, C, D_out, H_out, W_out]
    """
    assert inp.is_cuda and bias.is_cuda, "Inputs must be CUDA tensors"
    assert inp.dtype == torch.float32 and bias.dtype == torch.float32

    # Ensure contiguous layouts
    inp = inp.contiguous()
    # flatten bias to [C]
    bias_flat = bias.view(-1).contiguous()

    N, C, D_in, H_in, W_in = inp.shape

    # pooling params
    kernel = 2
    stride = 2
    D_out = (D_in - kernel) // stride + 1
    H_out = (H_in - kernel) // stride + 1
    W_out = (W_in - kernel) // stride + 1

    out = torch.empty((N, C, D_out, H_out, W_out), device=inp.device, dtype=inp.dtype)

    n_elements_out = N * C * D_out * H_out * W_out

    # Prepare scalar fused factors
    # final expression: out = s * (scale1*scale2/8) + bias * scale2
    scale_combined = float(scale1 * scale2 * (1.0 / 8.0))
    bias_scale = float(scale2)

    # Launch kernel
    # Choose BLOCK as autotune parameter
    grid = lambda meta: ((n_elements_out + meta['BLOCK'] - 1) // meta['BLOCK'],)

    fused_pool_bias_kernel[grid](
        inp,
        bias_flat,
        out,
        N, C, D_in, H_in, W_in, D_out, H_out, W_out,
        scale_combined, bias_scale, n_elements_out
    )
    return out


class ModelNew(nn.Module):
    """
    Optimized model: keep ConvTranspose3d as PyTorch op (leverages high-performance cuDNN),
    fuse the subsequent scaling, 3D average pooling (kernel=2,stride=2), bias addition, and final scaling
    into a Triton kernel for improved memory locality and fewer kernel launches.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, scale1, scale2, bias_shape):
        super(ModelNew, self).__init__()
        # Keep the convolution transpose in PyTorch to use optimized backend
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        # Keep parameters as tensors for storage; note fused kernel uses their values (not autograd-tracked through Triton)
        self.scale1 = nn.Parameter(torch.tensor(scale1, dtype=torch.float32))
        self.scale2 = nn.Parameter(torch.tensor(scale2, dtype=torch.float32))
        # bias stored as same shape but flattened for kernel
        self.bias = nn.Parameter(torch.randn(bias_shape, dtype=torch.float32))

    def forward(self, x):
        # x: [N, in_channels, D, H, W]
        x = self.conv_transpose(x)  # use optimized cuDNN/cuBLAS implementation
        # Use fused Triton kernel for pooling, bias and scaling
        # Note: pass Python float values for scales; this will not track gradients through the Triton kernel.
        out = fused_pool_bias(x, self.bias, float(self.scale1.item()), float(self.scale2.item()))
        return out