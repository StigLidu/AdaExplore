import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

# Autotune configs for the fused BN->ReLU->pool->FC kernel.
# We autotune BLOCK_C (channel tile), BLOCK_K (class tile) and a small SPB (spatial inner loop).
# Favor small SPB (1) and larger BLOCK_C to leverage NHWC contiguous channel loads.
AUTOTUNE_CONFIGS_SUM = [
    triton.Config({"BLOCK_C": 512, "BLOCK_K": 256, "SPB": 1}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_C": 256, "BLOCK_K": 128, "SPB": 1}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_C": 256, "BLOCK_K": 256, "SPB": 1}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_C": 128, "BLOCK_K": 128, "SPB": 1}, num_warps=4, num_stages=2),
]

@triton.autotune(configs=AUTOTUNE_CONFIGS_SUM, key=['N', 'C', 'HW', 'num_classes'])
@triton.jit
def bn_relu_pool_fc_fused_kernel(
    x_ptr,         # input tensor flattened NHWC but stored as FP16: ((n*HW + s) * C + c)
    scale_ptr,     # per-channel BN scale: float32[C] (gamma * rstd). If identity, pass ones.
    shift_ptr,     # per-channel BN shift: float32[C] (beta - scale*running_mean). If none, pass zeros.
    weight_ptr,    # classifier weights flattened: float32[num_classes * C]
    bias_ptr,      # classifier bias pointer: float32[num_classes]
    out_ptr,       # output logits flattened: float32[N * num_classes]
    N, C, H, W, HW, inv_HW_f, num_classes,  # runtime values; inv_HW_f is float scalar to convert sum->mean
    BLOCK_C: tl.constexpr,     # channel block size
    BLOCK_K: tl.constexpr,     # class block size
    SPB: tl.constexpr,         # spatial elements per inner load
):
    """
    Fused Triton kernel that:
      - For each (k_block, n) program it scans channel tiles,
      - For each channel tile: loads scale/shift, then for each small spatial tile loads a contiguous channel vector
        (NHWC favors contiguous C) from FP16, casts to FP32, applies BN-affine -> ReLU and accumulates per-channel sums,
      - After finishing a channel tile it loads the corresponding weight tile [BLOCK_K, BLOCK_C] and accumulates partial logits.
    The kernel computes final logits = mean_s ReLU(scale*x + shift) @ weight.T + bias without materializing (N,C).
    Grid: (num_k_blocks, N)
    """
    k_block = tl.program_id(0)  # class tile index
    n = tl.program_id(1)        # batch index

    k_start = k_block * BLOCK_K
    k_idx = k_start + tl.arange(0, BLOCK_K)  # [BLOCK_K]
    k_mask = k_idx < num_classes

    # accumulator for BLOCK_K classes (FP32)
    out_acc = tl.zeros([BLOCK_K], dtype=tl.float32)

    # precompute base for n in flattened NHWC indexing: base for ((n*HW)+s) * C
    n_hw = n * HW

    c = 0
    while c < C:
        c_start = c
        c_idx = c_start + tl.arange(0, BLOCK_C)  # [BLOCK_C]
        ch_mask = c_idx < C  # [BLOCK_C]

        # accumulator for channel tile sums (sum over spatial positions of ReLU(scale*x + shift))
        acc_ch = tl.zeros([BLOCK_C], dtype=tl.float32)

        # load scale & shift for this channel tile once (float32)
        scale_c = tl.load(scale_ptr + c_idx, mask=ch_mask, other=1.0)  # [BLOCK_C]
        shift_c = tl.load(shift_ptr + c_idx, mask=ch_mask, other=0.0)  # [BLOCK_C]

        s = 0
        # iterate spatially in chunks of SPB. We load contiguous C for each s (vectorized along channels).
        while s < HW:
            s_idx = s + tl.arange(0, SPB)  # [SPB]
            sp_mask = s_idx < HW  # [SPB]

            # offs shape [SPB, BLOCK_C] -> contiguous channels for each spatial index
            offs = (n_hw + s_idx[:, None]) * C + c_idx[None, :]
            mask = sp_mask[:, None] & ch_mask[None, :]  # [SPB, BLOCK_C]

            # load FP16 activations, cast to FP32 for arithmetic
            vals = tl.load(x_ptr + offs, mask=mask, other=0.0)  # [SPB, BLOCK_C], FP16 in memory
            vals = tl.cast(vals, tl.float32)

            # apply BN affine (scale * x + shift) then ReLU in FP32
            vals = vals * scale_c[None, :] + shift_c[None, :]
            vals = tl.maximum(vals, 0.0)
            # sum across spatial chunk axis -> [BLOCK_C]
            acc_ch = acc_ch + tl.sum(vals, axis=0)
            s += SPB

        # convert sum -> mean by multiplying with inv_HW (float32)
        inv_HW = tl.cast(inv_HW_f, tl.float32)
        acc_ch = acc_ch * inv_HW  # [BLOCK_C]

        # Load classifier weight tile [BLOCK_K, BLOCK_C]
        w_off = k_idx[:, None] * C + c_idx[None, :]
        w_mask = k_mask[:, None] & ch_mask[None, :]
        w = tl.load(weight_ptr + w_off, mask=w_mask, other=0.0)  # [BLOCK_K, BLOCK_C]

        # partial logits: dot product of each class row with acc_ch
        partial = tl.sum(w * acc_ch[None, :], axis=1)  # [BLOCK_K]
        out_acc = out_acc + partial

        c += BLOCK_C

    # finalize: add bias
    b = tl.load(bias_ptr + k_idx, mask=k_mask, other=0.0)
    out_acc = out_acc + b

    out_offs = n * num_classes + k_idx
    tl.store(out_ptr + out_offs, out_acc, mask=k_mask)


def fused_relu_pool_and_linear_inference(x: torch.Tensor, linear: nn.Linear, bn: nn.BatchNorm2d) -> torch.Tensor:
    """
    Inference wrapper:
      - Fold BatchNorm running stats into scale/shift.
      - Pass NHWC activations as FP16 to the Triton fused kernel which computes per-channel means
        and directly multiplies with classifier weights (no N×C intermediate).
      - Kernel returns logits (FP32).
    """
    assert x.is_cuda and x.dtype == torch.float32
    assert linear.weight.is_cuda and linear.weight.dtype == torch.float32

    # Prepare BN folded scale and shift (float32)
    bn_w = bn.weight.contiguous()
    bn_b = bn.bias.contiguous()
    bn_rm = bn.running_mean.contiguous()
    bn_rv = bn.running_var.contiguous()
    eps_f = float(bn.eps)

    rstd = torch.rsqrt(bn_rv + eps_f)
    scale = (bn_w * rstd).contiguous()     # [C], float32
    shift = (bn_b - scale * bn_rm).contiguous()  # [C], float32

    # classifier params (keep float32)
    weight = linear.weight.contiguous()   # [num_classes, C]
    bias = linear.bias.contiguous() if linear.bias is not None else torch.zeros(weight.shape[0], device=weight.device, dtype=weight.dtype)

    # Prepare NHWC FP16 activations for Triton kernel to reduce memory bandwidth
    x_cl_h = x.permute(0, 2, 3, 1).contiguous().half()  # NHWC, FP16
    N, H, W, C = x_cl_h.shape
    HW = H * W
    inv_HW = float(1.0 / HW)
    num_classes = weight.shape[0]

    # Output logits (FP32), computed by Triton (fused)
    out = torch.empty((N, num_classes), device=x.device, dtype=torch.float32)

    grid = lambda meta: ((num_classes + meta["BLOCK_K"] - 1) // meta["BLOCK_K"], N)

    bn_relu_pool_fc_fused_kernel[grid](
        x_cl_h,
        scale.contiguous(),
        shift.contiguous(),
        weight,
        bias.contiguous(),
        out,
        N, C, H, W, HW, inv_HW, num_classes,
    )
    return out


def fused_relu_pool_and_linear_training(x: torch.Tensor, linear: nn.Linear) -> torch.Tensor:
    """
    Training wrapper:
      - Final BatchNorm is applied in PyTorch before calling this to update running stats.
      - Call the fused Triton kernel with identity scale/shift; kernel computes logits directly.
    """
    assert x.is_cuda and x.dtype == torch.float32
    assert linear.weight.is_cuda and linear.weight.dtype == torch.float32

    N, C, H, W = x.shape
    device = x.device
    scale = torch.ones(C, device=device, dtype=torch.float32)
    shift = torch.zeros(C, device=device, dtype=torch.float32)

    weight = linear.weight.contiguous()   # [num_classes, C]
    bias = linear.bias.contiguous() if linear.bias is not None else torch.zeros(weight.shape[0], device=weight.device, dtype=weight.dtype)

    x_cl_h = x.permute(0, 2, 3, 1).contiguous().half()  # NHWC FP16
    HW = H * W
    inv_HW = float(1.0 / HW)
    num_classes = weight.shape[0]

    out = torch.empty((N, num_classes), device=device, dtype=torch.float32)

    grid = lambda meta: ((num_classes + meta["BLOCK_K"] - 1) // meta["BLOCK_K"], N)

    bn_relu_pool_fc_fused_kernel[grid](
        x_cl_h,
        scale,
        shift,
        weight,
        bias,
        out,
        N, C, H, W, HW, inv_HW, num_classes,
    )
    return out


# DenseBlock and TransitionLayer retained from original design (PyTorch modules)
class DenseBlock(nn.Module):
    def __init__(self, num_layers: int, num_input_features: int, growth_rate: int):
        super(DenseBlock, self).__init__()
        layers = []
        for i in range(num_layers):
            layers.append(self._make_layer(num_input_features + i * growth_rate, growth_rate))
        self.layers = nn.ModuleList(layers)

    def _make_layer(self, in_features: int, growth_rate: int):
        return nn.Sequential(
            nn.BatchNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_features, growth_rate, kernel_size=3, padding=1, bias=False),
            nn.Dropout(0.0)
        )

    def forward(self, x):
        features = [x]
        for layer in self.layers:
            new_feature = layer(x)
            features.append(new_feature)
            x = torch.cat(features, 1)
        return x


class TransitionLayer(nn.Module):
    def __init__(self, num_input_features: int, num_output_features: int):
        super(TransitionLayer, self).__init__()
        self.transition = nn.Sequential(
            nn.BatchNorm2d(num_input_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_input_features, num_output_features, kernel_size=1, bias=False),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        return self.transition(x)


class ModelNew(nn.Module):
    """
    Optimized DenseNet-like model with Triton-fused final stage.

    - During training: final BatchNorm is applied in PyTorch (updates running stats). We then call
      a Triton kernel that performs ReLU -> spatial mean -> linear in one kernel (identity BN).
    - During inference: we fold BatchNorm running stats into scale/shift and call the Triton kernel that
      applies BN-affine -> ReLU -> spatial mean -> linear in one kernel.
    """
    def __init__(self, growth_rate: int = 32, num_classes: int = 1000):
        super(ModelNew, self).__init__()

        # Initial convolution and pooling
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        num_features = 64
        block_layers = [6, 12, 48, 32]  # DenseNet201-like

        self.dense_blocks = nn.ModuleList()
        self.transition_layers = nn.ModuleList()

        for i, num_layers in enumerate(block_layers):
            block = DenseBlock(num_layers=num_layers, num_input_features=num_features, growth_rate=growth_rate)
            self.dense_blocks.append(block)
            num_features = num_features + num_layers * growth_rate

            if i != len(block_layers) - 1:
                transition = TransitionLayer(num_input_features=num_features, num_output_features=num_features // 2)
                self.transition_layers.append(transition)
                num_features = num_features // 2

        # Final BN (kept as module to retain training semantics & running stats)
        self.final_bn = nn.BatchNorm2d(num_features)
        # Classifier remains standard Linear; its params are used by Triton kernel.
        self.classifier = nn.Linear(num_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)

        for i, block in enumerate(self.dense_blocks):
            x = block(x)
            if i != len(self.dense_blocks) - 1:
                x = self.transition_layers[i](x)

        if self.training:
            # Training: preserve BN semantics (batch statistics update), then fused kernel with identity BN
            x = self.final_bn(x)
            logits = fused_relu_pool_and_linear_training(x, self.classifier)
        else:
            # Inference: fold BN running stats into scale/shift and call fused kernel
            logits = fused_relu_pool_and_linear_inference(x, self.classifier, self.final_bn)
        return logits