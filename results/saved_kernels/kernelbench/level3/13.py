import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

# Triton kernel: fused ReLU + 2x2 AveragePool (stride 2)
@triton.jit
def relu_avgpool2d_kernel(
    x_ptr,                 # input tensor pointer
    out_ptr,               # output tensor pointer
    N, C, H, W,            # input dims
    H_out, W_out,          # output spatial dims
    stride_n, stride_c, stride_h, stride_w,         # input strides (in elements)
    out_stride_n, out_stride_c, out_stride_h, out_stride_w,  # output strides
    BLOCK_W: tl.constexpr
):
    # pid0 indexes across (N * C * H_out)
    pid0 = tl.program_id(0)
    # pid1 indexes width blocks
    pid1 = tl.program_id(1)

    # decode pid0 -> n, c, h_out
    n_c_h = pid0
    h_out = n_c_h % H_out
    tmp = n_c_h // H_out
    c = tmp % C
    n = tmp // C

    # compute a range of w_out positions this program will handle
    offs_w = pid1 * BLOCK_W + tl.arange(0, BLOCK_W)
    mask_w = offs_w < W_out

    # input spatial coordinates (top-left corner)
    h_in0 = 2 * h_out
    h_in1 = h_in0 + 1

    # corresponding input w positions (for 2x2 window)
    w_in0 = 2 * offs_w
    w_in1 = w_in0 + 1

    # compute base offsets (in elements) for the four positions
    base00 = n * stride_n + c * stride_c + h_in0 * stride_h
    base10 = n * stride_n + c * stride_c + h_in1 * stride_h

    addr00 = base00 + w_in0 * stride_w
    addr01 = base00 + w_in1 * stride_w
    addr10 = base10 + w_in0 * stride_w
    addr11 = base10 + w_in1 * stride_w

    # load values (use other=0.0 for masked loads)
    v00 = tl.load(x_ptr + addr00, mask=mask_w, other=0.0)
    v01 = tl.load(x_ptr + addr01, mask=mask_w, other=0.0)
    v10 = tl.load(x_ptr + addr10, mask=mask_w, other=0.0)
    v11 = tl.load(x_ptr + addr11, mask=mask_w, other=0.0)

    # apply ReLU to each input element (ensure avg(ReLU(x)) semantics)
    v00 = tl.where(v00 > 0.0, v00, 0.0)
    v01 = tl.where(v01 > 0.0, v01, 0.0)
    v10 = tl.where(v10 > 0.0, v10, 0.0)
    v11 = tl.where(v11 > 0.0, v11, 0.0)

    # compute average over 2x2 window
    s = v00 + v01 + v10 + v11
    avg = s * 0.25
    outvals = avg

    # store into output
    out_base = n * out_stride_n + c * out_stride_c + h_out * out_stride_h
    out_addr = out_base + offs_w * out_stride_w
    tl.store(out_ptr + out_addr, outvals, mask=mask_w)


def triton_relu_avgpool2d(x: torch.Tensor, BLOCK_W: int = 64):
    """
    Performs ReLU followed by 2x2 average pooling (stride 2) using Triton.
    Input: x (N, C, H, W) contiguous CUDA tensor, H and W must be divisible by 2.
    Returns: (N, C, H//2, W//2) tensor.

    Notes:
    - Increased default BLOCK_W to 64 to increase per-program work and enable wider vectorized loads/stores.
    - Enforce contiguous CUDA fp32 input for best performance and correct masking behavior inside the kernel.
    """
    assert x.is_cuda, "Input tensor must be on CUDA."
    assert x.dtype == torch.float32, "Input tensor must be float32 for the Triton kernel."
    x = x.contiguous()
    N, C, H, W = x.shape
    assert H % 2 == 0 and W % 2 == 0, "H and W must be divisible by 2 for 2x2 avg pool."
    H_out = H // 2
    W_out = W // 2
    out = torch.empty((N, C, H_out, W_out), device=x.device, dtype=x.dtype)

    # get strides in number of elements
    stride_n, stride_c, stride_h, stride_w = x.stride()
    out_stride_n, out_stride_c, out_stride_h, out_stride_w = out.stride()

    # grid: pid0 over N*C*H_out, pid1 over blocks of W_out
    grid_pid0 = N * C * H_out
    grid_pid1 = (W_out + BLOCK_W - 1) // BLOCK_W
    grid = (grid_pid0, grid_pid1)

    relu_avgpool2d_kernel[grid](
        x, out,
        N, C, H, W,
        H_out, W_out,
        stride_n, stride_c, stride_h, stride_w,
        out_stride_n, out_stride_c, out_stride_h, out_stride_w,
        BLOCK_W=BLOCK_W
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, num_input_features: int, num_output_features: int):
        """
        Reimplements the original Model, but uses a fused Triton kernel for ReLU+2x2-AvgPool.
        The forward pass is reordered to run: BatchNorm -> fused(ReLU+AvgPool) -> Conv(1x1),
        which is mathematically equivalent to the original order but reduces the conv spatial cost.

        A helper `fold_bn_into_conv` is provided to fold BatchNorm into the Conv weights/bias
        for inference-time optimization (call it once before switching to eval).
        """
        super(ModelNew, self).__init__()
        # Keep BatchNorm and Conv2d as in original architecture
        self.bn = nn.BatchNorm2d(num_input_features)
        # The original conv has bias=False
        self.conv = nn.Conv2d(num_input_features, num_output_features, kernel_size=1, bias=False)

    def fold_bn_into_conv(self):
        """
        Fold BatchNorm parameters into the 1x1 conv weights and bias.
        This is typically used for inference/eval to remove the separate BN op.

        After folding, self.bn is replaced with nn.Identity(). Call this method once when
        preparing the model for deployment (and then set the module to eval()).
        """
        if not isinstance(self.bn, nn.BatchNorm2d):
            return  # nothing to do

        bn = self.bn
        conv = self.conv

        # Ensure tensors are on the same device/dtype
        w = conv.weight.data
        device = w.device
        dtype = w.dtype

        # Create conv.bias if not present
        if conv.bias is None:
            conv.bias = nn.Parameter(torch.zeros(conv.out_channels, device=device, dtype=dtype))

        # Compute BN folding parameters
        # scale shape: (out_channels, 1, 1, 1)
        running_var = bn.running_var.to(device=device, dtype=dtype)
        running_mean = bn.running_mean.to(device=device, dtype=dtype)
        if bn.weight is None:
            bn_weight = torch.ones_like(running_var)
        else:
            bn_weight = bn.weight.data.to(device=device, dtype=dtype)
        if bn.bias is None:
            bn_bias = torch.zeros_like(running_var)
        else:
            bn_bias = bn.bias.data.to(device=device, dtype=dtype)

        denom = torch.sqrt(running_var + bn.eps)
        scale = (bn_weight / denom).reshape(-1, 1, 1, 1)

        # Fold weights and biases
        new_w = w * scale
        new_b = bn_bias - (bn_weight * running_mean) / denom
        new_b = new_b.to(device=device, dtype=dtype) + conv.bias.data

        conv.weight.data = new_w
        conv.bias.data = new_b

        # Replace bn with identity to avoid extra computation
        self.bn = nn.Identity()

    def forward(self, x):
        # Apply BatchNorm
        x = self.bn(x)
        # Apply fused ReLU + 2x2 AvgPool via Triton (runs on full-res input)
        # triton_relu_avgpool2d asserts x is CUDA and contiguous and returns a CUDA tensor
        x = triton_relu_avgpool2d(x)
        # Apply 1x1 convolution on reduced H/2 x W/2 spatial grid
        x = self.conv(x)
        return x


# Helper functions to match the original get_inputs/get_init_inputs in the task
batch_size = 128
num_input_features = 32
num_output_features = 64
height, width = 256, 256

def get_inputs():
    return [torch.rand(batch_size, num_input_features, height, width).cuda()]

def get_init_inputs():
    return [num_input_features, num_output_features]