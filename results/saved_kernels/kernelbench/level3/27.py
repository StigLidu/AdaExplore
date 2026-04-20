import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

# Autotune a range of BLOCK and BLOCK_C combinations to find best performing config on A6000.
AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK": 256, "BLOCK_C": 4},  num_warps=4, num_stages=2),
    triton.Config({"BLOCK": 512, "BLOCK_C": 4},  num_warps=4, num_stages=2),
    triton.Config({"BLOCK": 1024, "BLOCK_C": 8}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK": 2048, "BLOCK_C": 8}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK": 2048, "BLOCK_C": 16}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK": 4096, "BLOCK_C": 16}, num_warps=8, num_stages=4),
]

@triton.autotune(configs=AUTOTUNE_CONFIGS, key=['HxW', 'C'])
@triton.jit
def _bn_relu_global_avg_kernel(
    x_ptr,            # ptr to input tensor (N*C*H*W) flattened (expected fp16)
    out_ptr,          # ptr to output tensor (N*C) flattened (fp32)
    scale_ptr,        # ptr to folded scale per-channel: gamma * invstd  (C,)
    bias_ptr,         # ptr to folded bias per-channel: beta - mean * scale (C,)
    HxW,              # number of spatial elements per channel (H*W)
    stride_c,         # stride to next channel (HxW)
    stride_n,         # stride to next batch (C*HxW)
    C,                # number of channels
    BLOCK: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    """
    Each program handles BLOCK_C channels for a single batch element and reduces across
    the spatial dimension. This implementation uses channel blocking and processes
    spatial blocks of size BLOCK. Inputs are stored as fp16 to decrease bandwidth;
    computations and accumulations happen in fp32 for precision.
    """
    pid = tl.program_id(0)  # flattened over N * num_cblocks
    num_cblocks = (C + BLOCK_C - 1) // BLOCK_C

    n = pid // num_cblocks
    cb = pid % num_cblocks

    c_base = cb * BLOCK_C
    c_offsets = c_base + tl.arange(0, BLOCK_C)          # shape (BLOCK_C,)
    mask_c = c_offsets < C                              # channel-valid mask

    # Load per-channel folded parameters (fp32) into small vectors (BLOCK_C,)
    scale = tl.load(scale_ptr + c_offsets, mask=mask_c, other=0.0)
    bias = tl.load(bias_ptr + c_offsets, mask=mask_c, other=0.0)

    # Per-channel accumulators (fp32)
    acc = tl.zeros((BLOCK_C,), dtype=tl.float32)

    # number of spatial blocks (ceil)
    num_blocks = (HxW + BLOCK - 1) // BLOCK
    for b in range(num_blocks):
        start = b * BLOCK
        offs = start + tl.arange(0, BLOCK)              # shape (BLOCK,)
        mask_sp = offs < HxW                             # shape (BLOCK,)

        # base addresses for each channel lane
        bases = n * stride_n + c_offsets * stride_c     # shape (BLOCK_C,)

        # create (BLOCK_C x BLOCK) grid of pointers and matching mask
        ptrs = bases[:, None] + offs[None, :]            # shape (BLOCK_C, BLOCK)
        mask = mask_c[:, None] & mask_sp[None, :]        # shape (BLOCK_C, BLOCK)

        # Load activations (fp16 -> cast to fp32), apply folded BN and ReLU,
        # then sum across spatial axis to get per-channel partial sums.
        x = tl.load(x_ptr + ptrs, mask=mask, other=0.0)  # fp16 values
        x = tl.cast(x, tl.float32)
        y = x * scale[:, None] + bias[:, None]
        # ReLU
        y = tl.where(y > 0.0, y, 0.0)
        # sum over spatial dimension (axis=1) to produce (BLOCK_C,) partial sums
        s = tl.sum(y, 1)
        acc = acc + s

    # Normalize by number of spatial elements and store results for valid channels only
    out_ptrs = n * C + c_offsets                         # shape (BLOCK_C,)
    res = acc / HxW                                      # shape (BLOCK_C,)
    tl.store(out_ptr + out_ptrs, res, mask=mask_c)


def _triton_bn_relu_global_avg(x: torch.Tensor, running_mean: torch.Tensor, running_var: torch.Tensor,
                               weight: torch.Tensor, bias: torch.Tensor, eps: float):
    """
    Wrapper to run the Triton fused BN+ReLU+GlobalAvg kernel.
    Expects x shaped (N, C, H, W) on CUDA.
    Returns (N, C) in fp32.
    This wrapper casts activations to fp16 to reduce memory bandwidth and uses
    folded BN parameters (scale and bias) in fp32.
    """
    assert x.is_cuda, "Input must be on CUDA for Triton kernel."
    N, C, H, W = x.shape
    HxW = H * W

    # Cast activations to fp16 to reduce bandwidth (Ampere benefits from half precision loads)
    x_h = x.contiguous().half()
    out = torch.empty((N * C,), device=x.device, dtype=torch.float32)

    stride_c = HxW
    stride_n = C * HxW

    # Pre-fold BN: scale = gamma * invstd ; bias_fold = beta - mean * scale
    invstd = (running_var + eps).rsqrt()
    # keep folded params in fp32 for stable accumulation
    scale = (weight * invstd).to(device=x.device, dtype=torch.float32)
    bias_fold = (bias - running_mean * scale).to(device=x.device, dtype=torch.float32)

    # grid: one program handles BLOCK_C channels for a single batch element
    grid = lambda meta: (N * ((C + meta['BLOCK_C'] - 1) // meta['BLOCK_C']),)

    _bn_relu_global_avg_kernel[grid](
        x_h, out, scale, bias_fold,
        HxW, stride_c, stride_n, C
    )
    out = out.view(N, C)
    return out


class ConvBnReLU(nn.Module):
    """
    Conv2d -> BatchNorm2d -> ReLU in training.
    In eval, folds BatchNorm parameters into Conv weights/bias and runs a single conv + ReLU
    to reduce memory traffic and kernel launches.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1, bias=True, eps=1e-5, momentum=0.1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                              padding=padding, stride=stride, dilation=dilation,
                              groups=groups, bias=bias)
        # Keep a standard BatchNorm for training (to update running stats)
        self.bn = nn.BatchNorm2d(out_channels, eps=eps, momentum=momentum)

    def forward(self, x):
        if self.training:
            x = self.conv(x)
            x = self.bn(x)
            return F.relu(x)
        else:
            # Fold BN into conv weights/bias using running stats
            mean = self.bn.running_mean
            var = self.bn.running_var
            eps = self.bn.eps
            gamma = self.bn.weight
            beta = self.bn.bias

            device = self.conv.weight.device
            dtype = self.conv.weight.dtype

            mean = mean.to(device=device, dtype=dtype)
            invstd = (var + eps).rsqrt().to(device=device, dtype=dtype)
            gamma = gamma.to(device=device, dtype=dtype)
            beta = beta.to(device=device, dtype=dtype)

            # scale per-output-channel
            scale = (gamma * invstd).view(-1, 1, 1, 1)  # (out,1,1,1)
            w_fold = self.conv.weight * scale

            if self.conv.bias is not None:
                b = self.conv.bias.to(device=device, dtype=dtype)
            else:
                b = torch.zeros_like(mean, device=device, dtype=dtype)
            b_fold = (b - mean) * invstd * gamma + beta

            out = F.conv2d(x, w_fold, b_fold, stride=self.conv.stride,
                           padding=self.conv.padding, dilation=self.conv.dilation,
                           groups=self.conv.groups)
            return F.relu(out)


class BatchNormReLU2dSkip(nn.Module):
    """
    A special BN+ReLU placeholder used in the final stage so we can defer applying
    BN+ReLU until we compute the global average (fusing both steps). In eval mode
    this module is a no-op in forward (returns input unchanged) but still exposes
    the BN parameters (running stats, weight, bias, eps) so the caller can run the
    fused Triton reduction. In training mode it behaves like nn.BatchNorm2d + ReLU.
    """
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        # affine parameters
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))

        # running stats (buffers)
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, x):
        if self.training:
            # Use functional batch_norm so running stats update correctly
            out = F.batch_norm(
                x, self.running_mean, self.running_var,
                self.weight, self.bias,
                training=True, momentum=self.momentum, eps=self.eps
            )
            return F.relu(out)
        # In eval, skip BN+ReLU here; fused kernel will handle it while computing the global average.
        return x


class ModelNew(nn.Module):
    """
    Optimized RegNet-like model:
      - Intermediate Conv->BN->ReLU blocks are implemented via ConvBnReLU which folds
        BN into Conv at eval time to eliminate separate BN kernels.
      - The final stage's BN+ReLU is deferred (BatchNormReLU2dSkip) and computed together
        with global average pooling using a Triton kernel to avoid materializing the
        full post-BN activation map. The Triton kernel uses fp16 loads and fp32 accumulation.
      - Training behavior preserved: during training, standard Conv->BN->ReLU sequences run.
    """
    def __init__(self, input_channels, stages, block_widths, output_classes):
        super(ModelNew, self).__init__()

        self.stages = stages
        self.block_widths = block_widths

        layers = []
        current_channels = input_channels

        # Construct stages
        for i in range(stages):
            last = (i == stages - 1)
            out_ch = block_widths[i]
            if not last:
                # two folded ConvBnReLU blocks + MaxPool
                stage = nn.Sequential(
                    ConvBnReLU(current_channels, out_ch, kernel_size=3, padding=1),
                    ConvBnReLU(out_ch, out_ch, kernel_size=3, padding=1),
                    nn.MaxPool2d(kernel_size=2, stride=2)
                )
            else:
                # For final stage: fold first conv, keep second conv plain and a BN-skip
                conv1 = ConvBnReLU(current_channels, out_ch, kernel_size=3, padding=1)
                conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
                bn_skip = BatchNormReLU2dSkip(out_ch)
                stage = nn.Sequential(
                    conv1,
                    conv2,
                    bn_skip,
                    nn.MaxPool2d(kernel_size=2, stride=2)
                )
                # store bn_skip reference for fused global avg
                self.last_bn = bn_skip
            layers.append(stage)
            current_channels = out_ch

        self.feature_extractor = nn.Sequential(*layers)
        self.fc = nn.Linear(block_widths[-1], output_classes)

    def forward(self, x):
        """
        Forward pass through the RegNet model.
        In eval mode we:
          - run folded convs for intermediate BN+ReLU (ConvBnReLU)
          - run the last stage convs (conv1 folded by ConvBnReLU, conv2 plain)
          - then apply a fused Triton BN+ReLU+GlobalAvg over the output of the last stage
            to obtain (N, C) without materializing the full post-BN map.
        In training mode we keep PyTorch ops to preserve autograd and running-stat updates.
        """
        if not self.training:
            # Eval path: folded convs will be executed inside ConvBnReLU modules
            # feature_extractor returns the pre-final-BN feature map (after conv2 and pool)
            x = self.feature_extractor(x)
            # Apply fused BN+ReLU + GlobalAvg to get (N, C) directly without materializing post-BN activation map.
            x = _triton_bn_relu_global_avg(
                x, self.last_bn.running_mean, self.last_bn.running_var,
                self.last_bn.weight, self.last_bn.bias, self.last_bn.eps
            )
        else:
            # Training path: preserve standard behavior (Conv->BN->ReLU running-stat updates)
            x = self.feature_extractor(x)
            x = torch.mean(x, dim=[2, 3])

        x = self.fc(x)
        return x


# API helpers (no test harness)
def get_inputs():
    batch_size = 8
    input_channels = 3
    image_height, image_width = 224, 224
    return [torch.rand(batch_size, input_channels, image_height, image_width)]

def get_init_inputs():
    input_channels = 3
    stages = 3
    block_widths = [64, 128, 256]
    output_classes = 10
    return [input_channels, stages, block_widths, output_classes]