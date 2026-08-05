import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

# Autotune configs for elementwise ReLU kernel
RELU_AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK": 1024}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK": 2048}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK": 4096}, num_warps=8, num_stages=3),
]

@triton.autotune(configs=RELU_AUTOTUNE_CONFIGS, key=['n_elements'])
@triton.jit
def _triton_relu_kernel(x_ptr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    start = pid * BLOCK
    offs = start + tl.arange(0, BLOCK)
    mask = offs < n_elements
    vals = tl.load(x_ptr + offs, mask=mask, other=0.0)
    res = tl.where(vals > 0.0, vals, 0.0)
    tl.store(x_ptr + offs, res, mask=mask)


def triton_relu_inplace(x: torch.Tensor):
    """
    In-place ReLU implemented in Triton. Works only for CUDA float32 tensors.
    Falls back to torch.nn.functional.relu for other cases.
    """
    if not x.is_cuda or x.dtype != torch.float32:
        return F.relu(x, inplace=True)
    x = x.contiguous()
    n = x.numel()
    if n == 0:
        return x
    grid = lambda meta: ((n + meta["BLOCK"] - 1) // meta["BLOCK"],)
    _triton_relu_kernel[grid](x, n)
    return x


class ModelNew(nn.Module):
    """
    MobileNetV1-like architecture optimized for inference by:
      - Folding BatchNorm parameters into preceding Conv2d weights/biases when switched to eval(),
        removing BN execution overhead.
      - Using a Triton-based in-place ReLU kernel in CUDA fp32 inference to reduce kernel-launch &
        memory traffic overhead compared to PyTorch ReLU.
    Training behavior is unchanged (no folding, standard BN + ReLU).
    """
    def __init__(self, num_classes=1000, input_channels=3, alpha=1.0):
        super(ModelNew, self).__init__()

        def conv_bn_pair(inp, oup, stride):
            return [
                nn.Conv2d(inp, oup, kernel_size=3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(oup),
            ]

        def conv_dw_pair(inp, oup, stride):
            # Depthwise conv (groups=inp) + BN, then pointwise conv + BN
            return [
                nn.Conv2d(inp, inp, kernel_size=3, stride=stride, padding=1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.Conv2d(inp, oup, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(oup),
            ]

        layers = []
        relu_after_indices = set()  # indices in layers after which we should apply ReLU

        def append_module(mod):
            idx = len(layers)
            layers.append(mod)
            return idx

        # First conv
        append_module(nn.Conv2d(input_channels, int(32 * alpha), kernel_size=3, stride=2, padding=1, bias=False))
        bn_idx = append_module(nn.BatchNorm2d(int(32 * alpha)))
        relu_after_indices.add(bn_idx)

        # conv_dw blocks
        def add_conv_dw_block(inp, oup, stride):
            # depthwise
            idx0 = append_module(nn.Conv2d(inp, inp, kernel_size=3, stride=stride, padding=1, groups=inp, bias=False))
            idx1 = append_module(nn.BatchNorm2d(inp))
            relu_after_indices.add(idx1)
            # pointwise
            idx2 = append_module(nn.Conv2d(inp, oup, kernel_size=1, stride=1, padding=0, bias=False))
            idx3 = append_module(nn.BatchNorm2d(oup))
            relu_after_indices.add(idx3)

        add_conv_dw_block(int(32 * alpha), int(64 * alpha), 1)
        add_conv_dw_block(int(64 * alpha), int(128 * alpha), 2)
        add_conv_dw_block(int(128 * alpha), int(128 * alpha), 1)
        add_conv_dw_block(int(128 * alpha), int(256 * alpha), 2)
        add_conv_dw_block(int(256 * alpha), int(256 * alpha), 1)
        add_conv_dw_block(int(256 * alpha), int(512 * alpha), 2)
        # five times 512->512
        for _ in range(5):
            add_conv_dw_block(int(512 * alpha), int(512 * alpha), 1)
        add_conv_dw_block(int(512 * alpha), int(1024 * alpha), 2)
        add_conv_dw_block(int(1024 * alpha), int(1024 * alpha), 1)

        # AvgPool2d(7)
        append_module(nn.AvgPool2d(7))

        self.model = nn.ModuleList(layers)
        self.relu_after_indices = relu_after_indices
        self.fc = nn.Linear(int(1024 * alpha), num_classes)

        # Flag to avoid repeated folding
        self._bn_fused = False

    def fold_conv_bn(self, conv: nn.Conv2d, bn: nn.BatchNorm2d):
        """
        Fold BN parameters into conv weights and bias in-place. This uses BN's running statistics
        and is only correct for inference (eval) mode.
        """
        # Only fold if BN is a real BatchNorm2d and not already Identity
        if not isinstance(bn, nn.BatchNorm2d):
            return

        # Move parameters to conv device & dtype
        device = conv.weight.device
        dtype = conv.weight.dtype

        W = conv.weight.data
        if conv.bias is None:
            b = torch.zeros(W.size(0), device=device, dtype=dtype)
        else:
            b = conv.bias.data

        # BN params: if any are None, use defaults
        if bn.weight is None:
            gamma = torch.ones(bn.num_features, device=device, dtype=dtype)
        else:
            gamma = bn.weight.data.to(device=device, dtype=dtype)
        if bn.bias is None:
            beta = torch.zeros(bn.num_features, device=device, dtype=dtype)
        else:
            beta = bn.bias.data.to(device=device, dtype=dtype)

        running_mean = bn.running_mean.to(device=device, dtype=dtype)
        running_var = bn.running_var.to(device=device, dtype=dtype)
        eps = bn.eps

        invstd = gamma.div(torch.sqrt(running_var + eps))  # shape [C]

        # Reshape for broadcasting over convolution weight dims: (out_channels, 1, 1, 1)
        view_shape = [W.size(0)] + [1] * (W.dim() - 1)
        W.mul_(invstd.view(*view_shape))
        new_bias = (b - running_mean) * invstd + beta

        conv.weight.data = W
        conv.bias = nn.Parameter(new_bias)

    def fold_all_batchnorms(self):
        """
        Iterate over model modules and fold every BatchNorm2d into the preceding Conv2d when possible.
        Replace folded BatchNorm2d modules with nn.Identity to preserve layer indexing.
        """
        if self._bn_fused:
            return
        # Iterate indices so we can replace modules inside ModuleList by assignment
        for idx in sorted(self.relu_after_indices):
            # bn is expected at index idx
            if idx >= len(self.model):
                continue
            bn = self.model[idx]
            # preceding conv index is idx - 1
            conv_idx = idx - 1
            if conv_idx < 0 or conv_idx >= len(self.model):
                continue
            conv = self.model[conv_idx]
            if isinstance(bn, nn.BatchNorm2d) and isinstance(conv, nn.Conv2d):
                try:
                    self.fold_conv_bn(conv, bn)
                    # replace BN with Identity so forward indexing remains valid
                    self.model[idx] = nn.Identity()
                except Exception:
                    # Any folding failure: skip folding this pair
                    continue
        self._bn_fused = True

    def eval(self):
        """
        Override eval() so that when model is switched to evaluation mode, we fold BN into convs
        to eliminate BN compute during inference.
        """
        super(ModelNew, self).eval()
        # Perform folding in-place
        self.fold_all_batchnorms()
        return self

    def forward(self, x):
        # Forward through model list. Apply ReLU after positions recorded in relu_after_indices.
        for idx, layer in enumerate(self.model):
            x = layer(x)
            if idx in self.relu_after_indices:
                # Activation point: use Triton in-place ReLU in CUDA fp32 inference for speed.
                if x.is_cuda and (not self.training):
                    x = triton_relu_inplace(x)
                else:
                    x = F.relu(x, inplace=True)
        # After avgpool, flatten and fc
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x