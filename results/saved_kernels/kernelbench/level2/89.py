import torch
import torch.nn as nn
import triton
import triton.language as tl

# Triton fused kernel:
# For each spatial position (batch * D * H * W), process the vector across channels:
#   1) softmax across channels (numerically stable)
#   2) subtract per-channel bias
#   3) swish activation: x * sigmoid(x)
#   4) reduce (max) across channels -> single scalar per spatial position
@triton.jit
def _fused_softmax_sub_swish_max_kernel(
    inp_ptr,        # pointer to input tensor (flattened N*C*S layout)
    sub_ptr,        # pointer to subtraction vector of length C
    out_ptr,        # pointer to output tensor (flattened N*S layout)
    Npos,           # total number of positions = N * S
    S,              # spatial size per sample = D * H * W
    BLOCK_POS: tl.constexpr,  # number of positions per program
    BLOCK_C: tl.constexpr,    # number of channels (constexpr)
):
    pid = tl.program_id(0)
    # positions this program will process
    offs = pid * BLOCK_POS + tl.arange(0, BLOCK_POS)
    mask_pos = offs < Npos  # shape (BLOCK_POS,)

    # channel indices (constexpr length)
    c_idx = tl.arange(0, BLOCK_C)  # shape (BLOCK_C,)

    # compute indices into flattened input:
    # index for channel c and position p is: c * S + p
    # shape (BLOCK_C, BLOCK_POS)
    idxs = c_idx[:, None] * S + offs[None, :]

    # load values for all channels for each position
    vals = tl.load(inp_ptr + idxs, mask=mask_pos[None, :], other=-1e30)  # (C, BLOCK_POS)

    # compute softmax across channels in a numerically stable way
    m = tl.max(vals, 0)                         # (BLOCK_POS,)
    exps = tl.exp(vals - m[None, :])           # (C, BLOCK_POS)
    sumexp = tl.sum(exps, 0)                   # (BLOCK_POS,)
    soft = exps / sumexp[None, :]              # (C, BLOCK_POS)

    # load subtract vector (length C)
    sub_vec = tl.load(sub_ptr + c_idx)         # (C,)

    # subtract (broadcast) and apply swish: y * sigmoid(y)
    y = soft - sub_vec[:, None]                # (C, BLOCK_POS)
    sig = 1.0 / (1.0 + tl.exp(-y))
    z = y * sig                                # (C, BLOCK_POS)

    # max across channels -> output per position
    out_vals = tl.max(z, 0)                    # (BLOCK_POS,)

    # store results
    tl.store(out_ptr + offs, out_vals, mask=mask_pos)


def triton_fused_channelwise_process(x: torch.Tensor, subtract: torch.Tensor):
    """
    x: tensor of shape (N, C, D, H, W), contiguous, float32 on CUDA
    subtract: tensor of shape (C,), contiguous, float32 on CUDA
    returns: tensor of shape (N, D, H, W)
    """
    assert x.is_cuda and subtract.is_cuda, "Tensors must be on CUDA."
    assert x.dtype == torch.float32 and subtract.dtype == torch.float32

    x = x.contiguous()
    subtract = subtract.contiguous()

    N, C, D, H, W = x.shape
    S = D * H * W
    Npos = N * S

    # flatten tensors to 1D contiguous memory for simple pointer arithmetic in Triton
    inp_flat = x.view(-1)
    out = torch.empty((N, D, H, W), device=x.device, dtype=x.dtype)
    out_flat = out.view(-1)

    # choose block sizes; BLOCK_C must match channels (constexpr)
    BLOCK_POS = 256
    BLOCK_C = C  # must be a small constexpr, here C=16

    grid = ( (Npos + BLOCK_POS - 1) // BLOCK_POS, )

    # launch kernel; note we pass BLOCK_POS and BLOCK_C as constexpr kwargs
    _fused_softmax_sub_swish_max_kernel[grid](
        inp_flat, subtract, out_flat, Npos, S,
        BLOCK_POS=BLOCK_POS, BLOCK_C=BLOCK_C
    )

    return out


class ModelNew(nn.Module):
    """
    Optimized model: uses PyTorch ConvTranspose3d and MaxPool3d,
    then applies a fused Triton kernel that performs:
      softmax (across channels) -> subtract (per-channel) -> swish -> channel-wise max
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, pool_kernel_size, pool_stride, pool_padding):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.max_pool = nn.MaxPool3d(kernel_size=pool_kernel_size, stride=pool_stride, padding=pool_padding)
        # subtraction parameter across channels
        self.subtract = nn.Parameter(torch.randn(out_channels, dtype=torch.float32))

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.max_pool(x)
        # apply fused Triton kernel
        # ensure subtract is on same device
        subtract = self.subtract
        if subtract.device != x.device:
            subtract = subtract.to(x.device)
        out = triton_fused_channelwise_process(x, subtract)
        return out