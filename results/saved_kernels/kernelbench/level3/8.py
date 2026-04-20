import torch
import torch.nn as nn
import torch.nn.functional as F

# Triton is only used on the optimized CUDA eval path.
import triton
import triton.language as tl

# Autotune configurations tuned for NVIDIA A6000 (Ampere).
# Explore a mix of spatial tile sizes, channel tiles and vector widths.
AUTOTUNE_CONFIGS = [
    # Smaller tiles / medium channel tile for latency-balanced configs
    triton.Config({"BLOCK": 256,  "BLOCK_C": 128, "VEC": 8}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK": 512,  "BLOCK_C": 128, "VEC": 8}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK": 512,  "BLOCK_C": 64,  "VEC": 8}, num_warps=8, num_stages=3),

    # Larger spatial tiles to reduce launch overhead on Ampere; prefer VEC=8/16 when BLOCK_C is 64
    triton.Config({"BLOCK": 1024, "BLOCK_C": 64,  "VEC": 8},  num_warps=8, num_stages=3),
    triton.Config({"BLOCK": 1024, "BLOCK_C": 64,  "VEC": 16}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK": 2048, "BLOCK_C": 64,  "VEC": 8},  num_warps=8, num_stages=2),
    triton.Config({"BLOCK": 2048, "BLOCK_C": 64,  "VEC": 16}, num_warps=8, num_stages=2),

    # Very large per-program work point for throughput-bound cases
    triton.Config({"BLOCK": 4096, "BLOCK_C": 64,  "VEC": 8},  num_warps=8, num_stages=2),
]

@triton.autotune(configs=AUTOTUNE_CONFIGS, key=['N', 'H', 'W', 'C'])
@triton.jit
def _add_relu_nhwc(
    x_ptr,           # pointer to first input (NHWC flat pointer)
    id_ptr,          # pointer to identity (NHWC)
    out_ptr,         # pointer to output (NHWC)
    N, H, W, C,
    BLOCK: tl.constexpr,
    BLOCK_C: tl.constexpr,
    VEC: tl.constexpr,
):
    """
    Fused add + ReLU for NHWC (channels-last) layout.
    Grid: (N, tiles_hw, tiles_c)
    Each program handles a spatial tile of size BLOCK and a channel tile of size BLOCK_C.
    Inner channel loops operate in VEC-sized vectors for wide loads/stores.
    """
    n_hw = H * W
    tiles_hw = (n_hw + BLOCK - 1) // BLOCK
    tiles_c = (C + BLOCK_C - 1) // BLOCK_C

    n = tl.program_id(0)
    tile_hw = tl.program_id(1)
    tile_c = tl.program_id(2)

    hw_base = tile_hw * BLOCK
    c_base = tile_c * BLOCK_C

    hw_offs = tl.arange(0, BLOCK)
    hw_idx = hw_base + hw_offs
    mask_hw = hw_idx < n_hw

    pos_base = n * n_hw + hw_idx           # [BLOCK]
    base_ptr = pos_base * C                # pointer base for each spatial pos

    x_row_ptr = x_ptr + base_ptr
    id_row_ptr = id_ptr + base_ptr
    out_row_ptr = out_ptr + base_ptr

    # Process channel tile in VEC-sized vector chunks
    for c_start in range(0, BLOCK_C, VEC):
        c_off = tl.arange(0, VEC)
        c_idx = c_base + c_start + c_off
        mask_c = c_idx < C

        mask = mask_hw[:, None] & mask_c[None, :]

        addrs_x = x_row_ptr[:, None] + c_idx[None, :]
        addrs_id = id_row_ptr[:, None] + c_idx[None, :]
        addrs_out = out_row_ptr[:, None] + c_idx[None, :]

        x_val = tl.load(addrs_x, mask=mask, other=0.0)
        id_val = tl.load(addrs_id, mask=mask, other=0.0)

        y = x_val + id_val
        y = tl.where(y > 0.0, y, 0.0)

        tl.store(addrs_out, y, mask=mask)


def triton_add_relu_nhwc(x: torch.Tensor, identity: torch.Tensor):
    """
    Host wrapper for the Triton NHWC fused add+relu kernel.
    Works with channels_last memory format. Minimizes allocations by reusing buffers when possible.
    """
    assert x.is_cuda and identity.is_cuda, "Triton kernels require CUDA tensors."

    # Check if tensors are already channels-last contiguous; convert only if necessary.
    x_was_ch_last = (x.layout == torch.channels_last and x.is_contiguous(memory_format=torch.channels_last))
    id_was_ch_last = (identity.layout == torch.channels_last and identity.is_contiguous(memory_format=torch.channels_last))

    x_ch = x if x_was_ch_last else x.contiguous(memory_format=torch.channels_last)
    id_ch = identity if id_was_ch_last else identity.contiguous(memory_format=torch.channels_last)

    # Reuse x_ch storage as output to avoid extra allocation.
    out = x_ch

    # NHWC logical dims: N, H, W, C (PyTorch shape (N, C, H, W) with channels_last)
    N = out.shape[0]
    H = out.shape[2]
    W = out.shape[3]
    C = out.shape[1]
    n_hw = H * W

    grid = lambda meta: (
        N,
        (n_hw + meta['BLOCK'] - 1) // meta['BLOCK'],
        (C + meta['BLOCK_C'] - 1) // meta['BLOCK_C'],
    )

    _add_relu_nhwc[grid](out, id_ch, out, N, H, W, C)

    # Return in the caller's expected layout: if original x was NCHW, return contiguous NCHW.
    if not x_was_ch_last:
        return out.contiguous()
    return out


def _fold_conv_bn(conv: nn.Conv2d, bn: nn.BatchNorm2d):
    """
    Fold BatchNorm2d into Conv2d weights and bias for inference.
    Returns (W_fold, b_fold) on the same device/dtype as conv.weight.
    """
    W = conv.weight
    device = W.device
    dtype = W.dtype

    gamma = bn.weight.to(device=device, dtype=dtype)
    beta = bn.bias.to(device=device, dtype=dtype)
    running_mean = bn.running_mean.to(device=device, dtype=dtype)
    running_var = bn.running_var.to(device=device, dtype=dtype)
    eps = float(bn.eps)

    scale = gamma / torch.sqrt(running_var + eps)  # shape (out,)
    scale_w = scale.view(-1, 1, 1, 1)
    W_fold = W * scale_w

    if conv.bias is not None:
        conv_bias = conv.bias.to(device=device, dtype=dtype)
    else:
        conv_bias = torch.zeros(W.shape[0], device=device, dtype=dtype)

    b_fold = scale * (conv_bias - running_mean) + beta
    return W_fold, b_fold


class ModelNew(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(ModelNew, self).__init__()
        # Keep original modules for training/CPU correctness.
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Downsample block (Conv1x1 + BN)
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels * self.expansion, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels * self.expansion),
        )
        self.stride = stride

        # Cache folded weights/biases for inference to avoid recomputation across forwards.
        self._folded_device = None
        self._w1_fold = None
        self._b1_fold = None
        self._w2_fold = None
        self._b2_fold = None
        self._wds_fold = None
        self._bds_fold = None

    def _ensure_folded(self):
        """
        Lazily fold Conv+BN for the current device/dtype and cache results.
        """
        device = self.conv1.weight.device
        device_index = device.index if hasattr(device, "index") else None
        device_key = f"{device.type}:{device_index}"
        dtype = self.conv1.weight.dtype

        if self._folded_device == device_key and self._w1_fold is not None:
            return

        with torch.no_grad():
            w1, b1 = _fold_conv_bn(self.conv1, self.bn1)
            w2, b2 = _fold_conv_bn(self.conv2, self.bn2)
            ds_conv = self.downsample[0]
            ds_bn = self.downsample[1]
            wds, bds = _fold_conv_bn(ds_conv, ds_bn)

            # detach and ensure proper device/dtype/contiguity
            self._w1_fold = w1.detach().to(device=device, dtype=dtype).contiguous()
            self._b1_fold = b1.detach().to(device=device, dtype=dtype).contiguous()
            self._w2_fold = w2.detach().to(device=device, dtype=dtype).contiguous()
            self._b2_fold = b2.detach().to(device=device, dtype=dtype).contiguous()
            self._wds_fold = wds.detach().to(device=device, dtype=dtype).contiguous()
            self._bds_fold = bds.detach().to(device=device, dtype=dtype).contiguous()
            self._folded_device = device_key

    def forward(self, x):
        # Fallback to standard PyTorch ops for CPU or training mode for correctness.
        if (not x.is_cuda) or self.bn1.training or self.bn2.training:
            identity = x

            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.bn2(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            out = out + identity
            out = self.relu(out)
            return out

        # Optimized eval + CUDA path
        self._ensure_folded()

        # Convert input to channels_last for better memory access during conv and Triton kernel.
        x_ch = x
        if x_ch.layout != torch.channels_last or not x_ch.is_contiguous(memory_format=torch.channels_last):
            x_ch = x_ch.contiguous(memory_format=torch.channels_last)

        # conv1 (folded BN) - keep channels_last format
        out = F.conv2d(
            x_ch,
            self._w1_fold,
            self._b1_fold,
            stride=self.conv1.stride,
            padding=self.conv1.padding,
            dilation=self.conv1.dilation,
            groups=self.conv1.groups,
        )
        # ReLU (non-inplace to avoid aliasing with buffers used by Triton)
        out = F.relu(out)

        # Compute identity (downsample folded) before conv2 to allow potential GPU overlap
        identity = F.conv2d(
            x_ch,
            self._wds_fold,
            self._bds_fold,
            stride=self.downsample[0].stride,
            padding=self.downsample[0].padding,
            dilation=self.downsample[0].dilation,
            groups=self.downsample[0].groups,
        )

        # conv2 (folded BN)
        out = F.conv2d(
            out,
            self._w2_fold,
            self._b2_fold,
            stride=self.conv2.stride,
            padding=self.conv2.padding,
            dilation=self.conv2.dilation,
            groups=self.conv2.groups,
        )

        # Final fused add + relu implemented by optimized Triton NHWC kernel.
        out = triton_add_relu_nhwc(out, identity)
        return out