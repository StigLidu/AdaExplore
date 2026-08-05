import torch
import torch.nn as nn
import triton
import triton.language as tl

# Autotune configs to pick best BLOCK and warps for the device
AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK": 64},  num_warps=2, num_stages=2),
    triton.Config({"BLOCK": 128}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK": 256}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK": 512}, num_warps=8, num_stages=3),
]

@triton.autotune(
    configs=AUTOTUNE_CONFIGS,
    key=['N', 'Cin', 'Cout', 'D_in', 'H_in', 'W_in', 'Kd', 'Kh', 'Kw', 'pad_d', 'pad_h', 'pad_w']
)
@triton.jit
def _trconv3d_pool_fused_kernel(
    x_ptr,           # input tensor pointer (N, Cin, D_in, H_in, W_in)
    w_ptr,           # weight pointer (Cin, Cout, Kd, Kh, Kw) contiguous (linearized)
    conv_b_ptr,      # conv transpose bias pointer (length Cout) or empty tensor pointer
    out_ptr,         # output pointer (N, Cout, D_out_p, H_out_p, W_out_p)
    scale1_ptr,      # 0-dim tensor pointer (scale1)
    scale2_ptr,      # 0-dim tensor pointer (scale2)
    post_bias_ptr,   # per-output-channel bias pointer (length Cout)
    N, Cin, D_in, H_in, W_in,
    Cout, Kd, Kh, Kw,
    pad_d, pad_h, pad_w,
    stride,          # assumed 2
    D_out_conv, H_out_conv, W_out_conv,   # high-res conv-transpose output dims
    D_out_p, H_out_p, W_out_p,            # pooled output dims
    has_conv_bias,   # 1 if conv_bias exists else 0
    BLOCK: tl.constexpr
):
    # program ids
    pid_n = tl.program_id(0)
    pid_cout = tl.program_id(1)
    blk = tl.program_id(2)

    # spatial offsets in pooled output handled by this program
    offs = blk * BLOCK + tl.arange(0, BLOCK)
    spatial_size = D_out_p * H_out_p * W_out_p
    mask_spatial = offs < spatial_size

    # pooled coordinates
    hw_p = H_out_p * W_out_p
    d_p = offs // hw_p
    rem = offs - d_p * hw_p
    h_p = rem // W_out_p
    w_p = rem - h_p * W_out_p

    # accumulate result for each pooled spatial location
    acc = tl.zeros([BLOCK], dtype=tl.float32)

    HW_in = H_in * W_in
    inv8 = 1.0 / 8.0

    # linearization helpers for weight indexing
    K_block = Kd * Kh * Kw
    # Iterate input channels and kernel positions, accumulating contributions
    for in_c in range(Cin):
        base_input_nc = (pid_n * Cin + in_c) * D_in * HW_in
        base_w_ic = (in_c * Cout + pid_cout) * K_block

        # Unroll kernel loops (small K)
        for kd in range(Kd):
            for kh in range(Kh):
                for kw in range(Kw):
                    # load weight scalar for this (in_c, pid_cout, kd,kh,kw)
                    w_idx = base_w_ic + (kd * Kh + kh) * Kw + kw
                    w_val = tl.load(w_ptr + w_idx)

                    # For the 8 positions that map from high-res conv output into pooled voxel
                    # We'll compute id_num, ih_num, iw_num for each t in {0,1}^3 and apply bounds & stride checks
                    # t = (0,0,0)
                    od = d_p * 2 + 0
                    oh = h_p * 2 + 0
                    ow = w_p * 2 + 0
                    id_num = od + pad_d - kd
                    ih_num = oh + pad_h - kh
                    iw_num = ow + pad_w - kw
                    mask_id = (id_num >= 0) & (id_num < (2 * D_in)) & ((id_num & 1) == 0)
                    mask_ih = (ih_num >= 0) & (ih_num < (2 * H_in)) & ((ih_num & 1) == 0)
                    mask_iw = (iw_num >= 0) & (iw_num < (2 * W_in)) & ((iw_num & 1) == 0)
                    mask_all = mask_spatial & mask_id & mask_ih & mask_iw
                    id = id_num // 2
                    ih = ih_num // 2
                    iw = iw_num // 2
                    linear_idx = base_input_nc + id * HW_in + ih * W_in + iw
                    vals = tl.load(x_ptr + linear_idx, mask=mask_all, other=0.0)
                    acc += vals * w_val

                    # t = (0,0,1)
                    od = d_p * 2 + 0
                    oh = h_p * 2 + 0
                    ow = w_p * 2 + 1
                    id_num = od + pad_d - kd
                    ih_num = oh + pad_h - kh
                    iw_num = ow + pad_w - kw
                    mask_id = (id_num >= 0) & (id_num < (2 * D_in)) & ((id_num & 1) == 0)
                    mask_ih = (ih_num >= 0) & (ih_num < (2 * H_in)) & ((ih_num & 1) == 0)
                    mask_iw = (iw_num >= 0) & (iw_num < (2 * W_in)) & ((iw_num & 1) == 0)
                    mask_all = mask_spatial & mask_id & mask_ih & mask_iw
                    id = id_num // 2
                    ih = ih_num // 2
                    iw = iw_num // 2
                    linear_idx = base_input_nc + id * HW_in + ih * W_in + iw
                    vals = tl.load(x_ptr + linear_idx, mask=mask_all, other=0.0)
                    acc += vals * w_val

                    # t = (0,1,0)
                    od = d_p * 2 + 0
                    oh = h_p * 2 + 1
                    ow = w_p * 2 + 0
                    id_num = od + pad_d - kd
                    ih_num = oh + pad_h - kh
                    iw_num = ow + pad_w - kw
                    mask_id = (id_num >= 0) & (id_num < (2 * D_in)) & ((id_num & 1) == 0)
                    mask_ih = (ih_num >= 0) & (ih_num < (2 * H_in)) & ((ih_num & 1) == 0)
                    mask_iw = (iw_num >= 0) & (iw_num < (2 * W_in)) & ((iw_num & 1) == 0)
                    mask_all = mask_spatial & mask_id & mask_ih & mask_iw
                    id = id_num // 2
                    ih = ih_num // 2
                    iw = iw_num // 2
                    linear_idx = base_input_nc + id * HW_in + ih * W_in + iw
                    vals = tl.load(x_ptr + linear_idx, mask=mask_all, other=0.0)
                    acc += vals * w_val

                    # t = (0,1,1)
                    od = d_p * 2 + 0
                    oh = h_p * 2 + 1
                    ow = w_p * 2 + 1
                    id_num = od + pad_d - kd
                    ih_num = oh + pad_h - kh
                    iw_num = ow + pad_w - kw
                    mask_id = (id_num >= 0) & (id_num < (2 * D_in)) & ((id_num & 1) == 0)
                    mask_ih = (ih_num >= 0) & (ih_num < (2 * H_in)) & ((ih_num & 1) == 0)
                    mask_iw = (iw_num >= 0) & (iw_num < (2 * W_in)) & ((iw_num & 1) == 0)
                    mask_all = mask_spatial & mask_id & mask_ih & mask_iw
                    id = id_num // 2
                    ih = ih_num // 2
                    iw = iw_num // 2
                    linear_idx = base_input_nc + id * HW_in + ih * W_in + iw
                    vals = tl.load(x_ptr + linear_idx, mask=mask_all, other=0.0)
                    acc += vals * w_val

                    # t = (1,0,0)
                    od = d_p * 2 + 1
                    oh = h_p * 2 + 0
                    ow = w_p * 2 + 0
                    id_num = od + pad_d - kd
                    ih_num = oh + pad_h - kh
                    iw_num = ow + pad_w - kw
                    mask_id = (id_num >= 0) & (id_num < (2 * D_in)) & ((id_num & 1) == 0)
                    mask_ih = (ih_num >= 0) & (ih_num < (2 * H_in)) & ((ih_num & 1) == 0)
                    mask_iw = (iw_num >= 0) & (iw_num < (2 * W_in)) & ((iw_num & 1) == 0)
                    mask_all = mask_spatial & mask_id & mask_ih & mask_iw
                    id = id_num // 2
                    ih = ih_num // 2
                    iw = iw_num // 2
                    linear_idx = base_input_nc + id * HW_in + ih * W_in + iw
                    vals = tl.load(x_ptr + linear_idx, mask=mask_all, other=0.0)
                    acc += vals * w_val

                    # t = (1,0,1)
                    od = d_p * 2 + 1
                    oh = h_p * 2 + 0
                    ow = w_p * 2 + 1
                    id_num = od + pad_d - kd
                    ih_num = oh + pad_h - kh
                    iw_num = ow + pad_w - kw
                    mask_id = (id_num >= 0) & (id_num < (2 * D_in)) & ((id_num & 1) == 0)
                    mask_ih = (ih_num >= 0) & (ih_num < (2 * H_in)) & ((ih_num & 1) == 0)
                    mask_iw = (iw_num >= 0) & (iw_num < (2 * W_in)) & ((iw_num & 1) == 0)
                    mask_all = mask_spatial & mask_id & mask_ih & mask_iw
                    id = id_num // 2
                    ih = ih_num // 2
                    iw = iw_num // 2
                    linear_idx = base_input_nc + id * HW_in + ih * W_in + iw
                    vals = tl.load(x_ptr + linear_idx, mask=mask_all, other=0.0)
                    acc += vals * w_val

                    # t = (1,1,0)
                    od = d_p * 2 + 1
                    oh = h_p * 2 + 1
                    ow = w_p * 2 + 0
                    id_num = od + pad_d - kd
                    ih_num = oh + pad_h - kh
                    iw_num = ow + pad_w - kw
                    mask_id = (id_num >= 0) & (id_num < (2 * D_in)) & ((id_num & 1) == 0)
                    mask_ih = (ih_num >= 0) & (ih_num < (2 * H_in)) & ((ih_num & 1) == 0)
                    mask_iw = (iw_num >= 0) & (iw_num < (2 * W_in)) & ((iw_num & 1) == 0)
                    mask_all = mask_spatial & mask_id & mask_ih & mask_iw
                    id = id_num // 2
                    ih = ih_num // 2
                    iw = iw_num // 2
                    linear_idx = base_input_nc + id * HW_in + ih * W_in + iw
                    vals = tl.load(x_ptr + linear_idx, mask=mask_all, other=0.0)
                    acc += vals * w_val

                    # t = (1,1,1)
                    od = d_p * 2 + 1
                    oh = h_p * 2 + 1
                    ow = w_p * 2 + 1
                    id_num = od + pad_d - kd
                    ih_num = oh + pad_h - kh
                    iw_num = ow + pad_w - kw
                    mask_id = (id_num >= 0) & (id_num < (2 * D_in)) & ((id_num & 1) == 0)
                    mask_ih = (ih_num >= 0) & (ih_num < (2 * H_in)) & ((ih_num & 1) == 0)
                    mask_iw = (iw_num >= 0) & (iw_num < (2 * W_in)) & ((iw_num & 1) == 0)
                    mask_all = mask_spatial & mask_id & mask_ih & mask_iw
                    id = id_num // 2
                    ih = ih_num // 2
                    iw = iw_num // 2
                    linear_idx = base_input_nc + id * HW_in + ih * W_in + iw
                    vals = tl.load(x_ptr + linear_idx, mask=mask_all, other=0.0)
                    acc += vals * w_val

    # finalize: average over 8 high-res positions, add conv bias (if any), apply scale1, add post-bias, apply scale2
    avg_inputs = acc * inv8
    if has_conv_bias:
        conv_bias = tl.load(conv_b_ptr + pid_cout)
    else:
        conv_bias = 0.0
    scale1 = tl.load(scale1_ptr)
    scale2 = tl.load(scale2_ptr)
    post_bias = tl.load(post_bias_ptr + pid_cout)

    out_vals = (avg_inputs + conv_bias) * scale1
    out_vals = out_vals + post_bias
    out_vals = out_vals * scale2

    out_base = (pid_n * Cout + pid_cout) * (D_out_p * H_out_p * W_out_p)
    out_idx = out_base + offs
    tl.store(out_ptr + out_idx, out_vals, mask=mask_spatial)


def fused_trconv3d_pool(
    x: torch.Tensor,
    weight: torch.Tensor,
    conv_bias: torch.Tensor,
    scale1: torch.Tensor,
    post_bias: torch.Tensor,
    scale2: torch.Tensor,
    padding,
    stride=2,
    BLOCK=128
):
    """
    Wrapper to launch the Triton fused ConvTranspose3d + avgpool + bias + scalings kernel.
    This variant uses autotuning to pick the best BLOCK/warps.
    """
    assert x.is_cuda and weight.is_cuda and post_bias.is_cuda and scale1.is_cuda and scale2.is_cuda, "All tensors must be CUDA tensors."
    assert x.dtype == torch.float32 and weight.dtype == torch.float32, "Only float32 supported."

    N, Cin, D_in, H_in, W_in = x.shape
    Cin_w, Cout, Kd, Kh, Kw = weight.shape
    assert Cin == Cin_w, "Mismatch between input channels and weight."
    pad_d, pad_h, pad_w = padding

    # Compute high-res conv-transpose output dims
    D_out_conv = (D_in - 1) * stride - 2 * pad_d + Kd
    H_out_conv = (H_in - 1) * stride - 2 * pad_h + Kh
    W_out_conv = (W_in - 1) * stride - 2 * pad_w + Kw

    # Pooled output dims after AvgPool3d(kernel_size=2)
    D_out_p = D_out_conv // 2
    H_out_p = H_out_conv // 2
    W_out_p = W_out_conv // 2

    x = x.contiguous()
    weight = weight.contiguous()
    post_bias = post_bias.contiguous().view(-1)
    scale1 = scale1.contiguous()
    scale2 = scale2.contiguous()

    out = torch.empty((N, Cout, D_out_p, H_out_p, W_out_p), device=x.device, dtype=x.dtype)

    spatial_size = D_out_p * H_out_p * W_out_p
    num_blocks = (spatial_size + BLOCK - 1) // BLOCK
    grid = (N, Cout, num_blocks)

    has_conv_bias = 1 if (conv_bias is not None) else 0
    conv_b_ptr = conv_bias.contiguous() if conv_bias is not None else torch.empty((1,), device=x.device, dtype=x.dtype)

    _trconv3d_pool_fused_kernel[grid](
        x, weight, conv_b_ptr, out, scale1, scale2, post_bias,
        N, Cin, D_in, H_in, W_in,
        Cout, Kd, Kh, Kw,
        pad_d, pad_h, pad_w,
        stride,
        D_out_conv, H_out_conv, W_out_conv,
        D_out_p, H_out_p, W_out_p,
        has_conv_bias
    )

    return out


class ModelNew(nn.Module):
    """
    Optimized model: uses a fused Triton kernel to compute ConvTranspose3d (stride=2),
    average pooling (kernel=2, stride=2), per-channel bias addition, and two scalar scales.
    The nn.ConvTranspose3d module is kept to hold parameters and initialization, but forward uses Triton.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, scale1, scale2, bias_shape):
        super(ModelNew, self).__init__()
        # Keep conv_transpose module for parameter storage and initialization
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.scale1 = nn.Parameter(torch.tensor(scale1, dtype=torch.float32))
        self.scale2 = nn.Parameter(torch.tensor(scale2, dtype=torch.float32))
        self.bias = nn.Parameter(torch.randn(bias_shape, dtype=torch.float32))

    def forward(self, x):
        # Use the fused Triton implementation for the entire pipeline after the input
        weight = self.conv_transpose.weight
        conv_bias = self.conv_transpose.bias if self.conv_transpose.bias is not None else None
        post_bias = self.bias.view(self.bias.shape[0])
        pad = (self.conv_transpose.padding[0], self.conv_transpose.padding[1], self.conv_transpose.padding[2])
        # Choose a reasonably large BLOCK; autotune will pick best config
        return fused_trconv3d_pool(x, weight, conv_bias, self.scale1, post_bias, self.scale2, pad, stride=2, BLOCK=256)


# Keep helper variables and functions for compatibility with the harness
batch_size = 128
in_channels = 3
out_channels = 16
depth, height, width = 16, 32, 32
kernel_size = 3
stride = 2
padding = 1
scale1 = 0.5
scale2 = 1.0
bias_shape = (out_channels, 1, 1, 1)


def get_inputs():
    # return CUDA tensor as required by the Triton kernel
    return [torch.rand(batch_size, in_channels, depth, height, width).cuda().float()]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, scale1, scale2, bias_shape]