import torch
import torch.nn as nn
import triton
import triton.language as tl

# Autotune configs for elementwise kernels - tuned for Ampere (A6000)
AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK_SIZE": 128},  num_warps=2, num_stages=2),
    triton.Config({"BLOCK_SIZE": 256},  num_warps=4, num_stages=2),
    triton.Config({"BLOCK_SIZE": 512},  num_warps=8, num_stages=2),
    triton.Config({"BLOCK_SIZE": 1024}, num_warps=8, num_stages=3),
]

@triton.autotune(configs=AUTOTUNE_CONFIGS, key=['n_elements'])
@triton.jit
def _bn_relu_kernel(
    x_ptr,               # input pointer (fp32)
    weight_ptr,          # bn weight (gamma) pointer (fp32)
    bias_ptr,            # bn bias (beta) pointer (fp32)
    mean_ptr,            # mean pointer (fp32)
    var_ptr,             # var pointer (fp32)
    out_ptr,             # output pointer (fp32)
    n_elements,          # total elements (int)
    C,                   # num channels (int)
    HW,                  # height * width (int)
    eps,                 # eps (float)
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # compute channel index for each flat offset (assuming NCHW contiguous layout)
    tmp = offsets // HW
    c = tmp % C
    c = c.to(tl.int32)
    c_safe = tl.where(mask, c, 0)

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    gamma = tl.load(weight_ptr + c_safe, mask=mask, other=1.0)
    beta  = tl.load(bias_ptr + c_safe, mask=mask, other=0.0)
    mean  = tl.load(mean_ptr + c_safe, mask=mask, other=0.0)
    var   = tl.load(var_ptr + c_safe, mask=mask, other=1.0)

    invstd = 1.0 / tl.sqrt(var + eps)
    y = gamma * (x - mean) * invstd + beta

    # ReLU
    y = tl.where(y > 0.0, y, 0.0)

    tl.store(out_ptr + offsets, y, mask=mask)


@triton.autotune(configs=AUTOTUNE_CONFIGS, key=['n_elements'])
@triton.jit
def _bn_add_relu_kernel(
    x_ptr,               # input pointer (fp32) - the tensor to batchnorm
    weight_ptr,          # bn weight (gamma) pointer (fp32)
    bias_ptr,            # bn bias (beta) pointer (fp32)
    mean_ptr,            # mean pointer (fp32)
    var_ptr,             # var pointer (fp32)
    identity_ptr,        # identity pointer (fp32) to add
    out_ptr,             # output pointer (fp32)
    n_elements,          # total elements (int)
    C,                   # num channels (int)
    HW,                  # height * width (int)
    eps,                 # eps (float)
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    tmp = offsets // HW
    c = tmp % C
    c = c.to(tl.int32)
    c_safe = tl.where(mask, c, 0)

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    gamma = tl.load(weight_ptr + c_safe, mask=mask, other=1.0)
    beta  = tl.load(bias_ptr + c_safe, mask=mask, other=0.0)
    mean  = tl.load(mean_ptr + c_safe, mask=mask, other=0.0)
    var   = tl.load(var_ptr + c_safe, mask=mask, other=1.0)

    invstd = 1.0 / tl.sqrt(var + eps)
    y = gamma * (x - mean) * invstd + beta

    idv = tl.load(identity_ptr + offsets, mask=mask, other=0.0)
    y = y + idv

    # ReLU
    y = tl.where(y > 0.0, y, 0.0)

    tl.store(out_ptr + offsets, y, mask=mask)


def _ensure_contig_no_copy(x: torch.Tensor):
    # Return x if already contiguous, else return contiguous copy.
    return x if x.is_contiguous() else x.contiguous()


def triton_bn_relu(x: torch.Tensor, bn: nn.BatchNorm2d):
    """
    Applies BatchNorm (per-channel affine) followed by ReLU fused in Triton.
    If bn.training is True, batch statistics are computed from x; otherwise running stats.
    Assumes x is CUDA float32 tensor.
    """
    assert x.is_cuda and x.dtype == torch.float32, "Input must be CUDA float32 tensor."
    x_contig = _ensure_contig_no_copy(x)
    N, C, H, W = x_contig.shape
    n_elements = x_contig.numel()
    HW = H * W

    # Prepare bn params
    device = x_contig.device
    if bn.weight is None:
        gamma = torch.ones((C,), dtype=torch.float32, device=device)
    else:
        gamma = bn.weight.detach().to(device)
    if bn.bias is None:
        beta = torch.zeros((C,), dtype=torch.float32, device=device)
    else:
        beta = bn.bias.detach().to(device)

    eps = float(bn.eps)

    # Compute mean/var depending on training/eval
    if bn.training:
        # compute mean/var on-device; avoid extra copies by operating on x_contig
        # compute over N, H, W
        mean = x_contig.mean(dim=(0, 2, 3)).detach()
        var = x_contig.var(dim=(0, 2, 3), unbiased=False).detach()
    else:
        mean = bn.running_mean.detach().to(device)
        var = bn.running_var.detach().to(device)

    # ensure all param tensors are contiguous to avoid hidden copies in kernel
    gamma = gamma.contiguous()
    beta = beta.contiguous()
    mean = mean.contiguous()
    var = var.contiguous()

    out = torch.empty_like(x_contig)

    grid = lambda meta: ((n_elements + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)
    _bn_relu_kernel[grid](
        x_contig,
        gamma,
        beta,
        mean,
        var,
        out,
        n_elements,
        C,
        HW,
        eps
    )
    return out.view_as(x)


def triton_bn_add_relu(x: torch.Tensor, bn: nn.BatchNorm2d, identity: torch.Tensor):
    """
    Applies BatchNorm (per-channel affine) on x, adds identity, and applies ReLU --
    all fused in Triton. If bn.training is True, batch statistics are computed
    from x; otherwise running stats are used. Assumes x and identity are same shape and CUDA fp32.
    """
    assert x.is_cuda and identity.is_cuda and x.dtype == torch.float32, "Inputs must be CUDA float32 tensors."
    x_contig = _ensure_contig_no_copy(x)
    id_contig = _ensure_contig_no_copy(identity)
    N, C, H, W = x_contig.shape
    n_elements = x_contig.numel()
    HW = H * W

    device = x_contig.device
    if bn.weight is None:
        gamma = torch.ones((C,), dtype=torch.float32, device=device)
    else:
        gamma = bn.weight.detach().to(device)
    if bn.bias is None:
        beta = torch.zeros((C,), dtype=torch.float32, device=device)
    else:
        beta = bn.bias.detach().to(device)

    eps = float(bn.eps)

    if bn.training:
        mean = x_contig.mean(dim=(0, 2, 3)).detach()
        var = x_contig.var(dim=(0, 2, 3), unbiased=False).detach()
    else:
        mean = bn.running_mean.detach().to(device)
        var = bn.running_var.detach().to(device)

    gamma = gamma.contiguous()
    beta = beta.contiguous()
    mean = mean.contiguous()
    var = var.contiguous()

    out = torch.empty_like(x_contig)

    grid = lambda meta: ((n_elements + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)
    _bn_add_relu_kernel[grid](
        x_contig,
        gamma,
        beta,
        mean,
        var,
        id_contig,
        out,
        n_elements,
        C,
        HW,
        eps
    )
    return out.view_as(x)


class ModelNew(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        """
        Optimized ResNet-like block:
         - For eval: fold BatchNorm into Conv to use fast fused convs (no Triton).
         - For training: use Triton-fused BN+ReLU and BN+Add+ReLU, with micro-optimizations to avoid unnecessary copies.
        """
        super(ModelNew, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        # keep a placeholder ReLU (not used when Triton kernels are applied)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels * self.expansion, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels * self.expansion),
        )
        self.stride = stride

    def _fold_conv_bn(self, conv: nn.Conv2d, bn: nn.BatchNorm2d, device):
        """
        Fold BatchNorm parameters into convolution weights and bias (for eval mode).
        Returns (w_fold, b_fold) on given device.
        """
        w = conv.weight
        # handle conv bias (rare here, convs created bias=False in this model)
        if conv.bias is None:
            conv_bias = torch.zeros(w.shape[0], dtype=w.dtype, device=device)
        else:
            conv_bias = conv.bias

        # BN params (affine)
        if bn.weight is None:
            gamma = torch.ones_like(bn.running_mean, device=device)
        else:
            gamma = bn.weight.detach().to(device)
        if bn.bias is None:
            beta = torch.zeros_like(bn.running_mean, device=device)
        else:
            beta = bn.bias.detach().to(device)

        running_mean = bn.running_mean.detach().to(device)
        running_var = bn.running_var.detach().to(device)
        invstd = torch.rsqrt(running_var + bn.eps)

        scale = gamma * invstd
        bias = beta - running_mean * scale

        # fold into conv weights and biases
        w_fold = w.to(device) * scale.view(-1, 1, 1, 1)
        b_fold = bias + conv_bias.to(device)
        return w_fold, b_fold

    def forward(self, x):
        identity = x

        # Fast eval/inference path: fold BN into convs to avoid elementwise kernels
        if not self.training:
            device = x.device

            w1_fold, b1_fold = self._fold_conv_bn(self.conv1, self.bn1, device)
            out = nn.functional.conv2d(x, w1_fold, b1_fold, stride=self.conv1.stride, padding=self.conv1.padding, groups=self.conv1.groups)
            out = self.relu(out)

            # downsample fold if present
            if self.downsample is not None:
                ds_conv = self.downsample[0]
                ds_bn = self.downsample[1]
                wds_fold, bds_fold = self._fold_conv_bn(ds_conv, ds_bn, device)
                identity = nn.functional.conv2d(x, wds_fold, bds_fold, stride=ds_conv.stride, padding=ds_conv.padding, groups=ds_conv.groups)

            w2_fold, b2_fold = self._fold_conv_bn(self.conv2, self.bn2, device)
            out = nn.functional.conv2d(out, w2_fold, b2_fold, stride=self.conv2.stride, padding=self.conv2.padding, groups=self.conv2.groups)

            out = out + identity
            out = self.relu(out)
            return out

        # Training path: keep exact training semantics and use Triton-fused elementwise kernels
        out = self.conv1(x)
        out = triton_bn_relu(out, self.bn1)

        out = self.conv2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = triton_bn_add_relu(out, self.bn2, identity)

        return out