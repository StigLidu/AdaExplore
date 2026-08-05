import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _bn_double_avgpool_kernel(
    x_ptr,           # input tensor pointer (N, C, D_in, H_in, W_in)
    out_ptr,         # output tensor pointer (N, C, D_out, H_out, W_out)
    N, C,
    D_in, H_in, W_in,
    D_out, H_out, W_out,
    scale_ptr,       # per-channel scale (C,)
    shift_ptr,       # per-channel shift (C,)
    total_elems,     # total number of output elements
    BLOCK: tl.constexpr
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total_elems
    # compute multi-d indices from flattened output index
    idx = offs
    ow = idx % W_out
    idx = idx // W_out
    oh = idx % H_out
    idx = idx // H_out
    od = idx % D_out
    idx = idx // D_out
    c = idx % C
    n = idx // C

    # base input indices (each final output corresponds to 4x4x4 block in input)
    base_z = od * 4
    base_y = oh * 4
    base_x = ow * 4

    # initialize accumulator
    acc = tl.zeros([BLOCK], dtype=tl.float32)

    # Loop over 4x4x4 block
    for dz in range(4):
        iz = base_z + dz
        mask_z = iz < D_in
        for dy in range(4):
            iy = base_y + dy
            mask_y = iy < H_in
            for dx in range(4):
                ix = base_x + dx
                mask_x = ix < W_in
                # compute input flat index: (((n*C + c)*D_in + iz)*H_in + iy)*W_in + ix
                inp_index = (((n * C + c) * D_in + iz) * H_in + iy) * W_in + ix
                # load with mask
                valid = mask & mask_z & mask_y & mask_x
                vals = tl.load(x_ptr + inp_index, mask=valid, other=0.0)
                acc += vals

    # divide by 64 (4*4*4)
    out_vals = acc / 64.0

    # load per-channel scale and shift
    scale = tl.load(scale_ptr + c, mask=mask, other=1.0)
    shift = tl.load(shift_ptr + c, mask=mask, other=0.0)

    out = out_vals * scale + shift

    # compute output flat index to store
    out_index = offs  # already the flattened index into output
    tl.store(out_ptr + out_index, out, mask=mask)


def triton_bn_double_avgpool(x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor):
    """
    x: (N, C, D_in, H_in, W_in) CUDA float32 tensor
    scale, shift: (C,) CUDA float32 tensors
    Returns:
      out: (N, C, D_out, H_out, W_out) CUDA float32 tensor
    The function applies BatchNorm affine (using provided scale/shift) and two consecutive AvgPool3d(kernel_size=2) fused.
    """
    assert x.is_cuda and scale.is_cuda and shift.is_cuda, "Inputs must be CUDA tensors"
    assert x.dtype == torch.float32 and scale.dtype == torch.float32 and shift.dtype == torch.float32

    N, C, D_in, H_in, W_in = x.shape

    # compute sizes after two AvgPool3d(kernel_size=2, stride=2)
    D1 = (D_in - 2) // 2 + 1
    H1 = (H_in - 2) // 2 + 1
    W1 = (W_in - 2) // 2 + 1

    D_out = (D1 - 2) // 2 + 1
    H_out = (H1 - 2) // 2 + 1
    W_out = (W1 - 2) // 2 + 1

    out = torch.empty((N, C, D_out, H_out, W_out), device=x.device, dtype=x.dtype)

    total_elems = out.numel()
    if total_elems == 0:
        return out

    BLOCK = 256
    grid = ( (total_elems + BLOCK - 1) // BLOCK, )

    _bn_double_avgpool_kernel[grid](
        x, out,
        N, C,
        D_in, H_in, W_in,
        D_out, H_out, W_out,
        scale, shift,
        total_elems,
        BLOCK=BLOCK
    )
    return out


class ModelNew(nn.Module):
    """
    Optimized model: keep ConvTranspose3d, but fuse BatchNorm (eval-style affine using running stats)
    with two AvgPool3d(kernel_size=2) operations into a single Triton kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias_shape):
        super(ModelNew, self).__init__()
        # keep conv transpose to leverage highly-optimized cuDNN implementation
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        # keep BatchNorm3d module to hold parameters/state
        self.batch_norm = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        # perform conv transpose using PyTorch (cuDNN)
        x = self.conv_transpose(x)  # shape (N, C, D_in, H_in, W_in)

        # prepare per-channel scale and shift using appropriate stats
        # - In eval mode, use running_mean/running_var (as affine is deterministic)
        # - In training mode, use batch statistics computed from the current activation x
        bn = self.batch_norm
        device = x.device
        dtype = x.dtype
        C = x.shape[1]

        if bn.affine:
            weight = bn.weight.detach().to(device=device, dtype=dtype)
            bias = bn.bias.detach().to(device=device, dtype=dtype)
        else:
            weight = torch.ones(C, device=device, dtype=dtype)
            bias = torch.zeros(C, device=device, dtype=dtype)

        eps = bn.eps

        if bn.training:
            # compute batch statistics on the activation produced by conv_transpose
            # BatchNorm3d computes mean/var over (N, D, H, W) for each channel
            batch_mean = x.mean(dim=(0, 2, 3, 4))
            batch_var = x.var(dim=(0, 2, 3, 4), unbiased=False)
            stats_mean = batch_mean.to(device=device, dtype=dtype)
            stats_var = batch_var.to(device=device, dtype=dtype)
        else:
            # eval mode: use running stats
            stats_mean = bn.running_mean.detach().to(device=device, dtype=dtype)
            stats_var = bn.running_var.detach().to(device=device, dtype=dtype)

        # scale and shift such that y = scale * x + shift
        scale = weight / torch.sqrt(stats_var + eps)
        shift = bias - stats_mean * scale

        # call fused Triton kernel (batch-norm affine + two avg pools)
        out = triton_bn_double_avgpool(x, scale.contiguous(), shift.contiguous())
        return out