import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

# Autotune configs tuned for Ampere (A6000). Provide explicit 2D tiles (T_H x T_W)
# so the tuner can pick shapes that maximize coalesced loads along width.
AUTOTUNE_CONFIGS_DW = [
    triton.Config({"T_H": 8,  "T_W": 16}, num_warps=4, num_stages=2),
    triton.Config({"T_H": 16, "T_W": 16}, num_warps=4, num_stages=2),
    triton.Config({"T_H": 8,  "T_W": 32}, num_warps=8, num_stages=2),
    triton.Config({"T_H": 16, "T_W": 32}, num_warps=8, num_stages=3),
    triton.Config({"T_H": 32, "T_W": 32}, num_warps=8, num_stages=3),
    triton.Config({"T_H": 16, "T_W": 64}, num_warps=8, num_stages=3),
]


@triton.autotune(configs=AUTOTUNE_CONFIGS_DW, key=['N', 'C', 'out_HW', 'K', 'stride', 'pad'])
@triton.jit
def _depthwise_conv2d_kernel(
    inp_ptr,          # pointer to input (fp16), layout NCHW flattened
    w_ptr,            # pointer to weight (fp16) layout: (C, K, K) flattened as C*K*K
    b_ptr,            # pointer to bias (fp32) shape (C,)
    out_ptr,          # pointer to output (fp16)
    N, C, H, W,       # input dims
    out_H, out_W,     # output spatial dims
    K: tl.constexpr, pad, stride: tl.constexpr,   # kernel params marked constexpr for unrolling/prefetch sizing
    out_HW,           # out_H * out_W
    T_H: tl.constexpr,
    T_W: tl.constexpr,
):
    """
    Mixed-precision, tiled depthwise conv2d - optimized prefetching:
      - Inputs & weights are loaded as fp16 and cast to fp32 for accumulation.
      - Accumulators are fp32; bias is fp32.
      - For each ky we prefetch a contiguous horizontal patch of length
        total_len = (T_W - 1) * stride + K for each row in the tile and reuse it
        across the inner kx loop, reducing global loads by ~Kx.
      - After accumulation we cast fp32->fp16 once and store fp16 outputs.
      - K and stride are constexpr to enable compile-time arange & unrolling.
    """
    c = tl.program_id(0)   # channel index
    n = tl.program_id(1)   # batch index
    blk = tl.program_id(2) # block index along tiled output pixels

    blocks_per_row = (out_W + T_W - 1) // T_W
    block_y = blk // blocks_per_row
    block_x = blk - block_y * blocks_per_row

    start_oy = block_y * T_H
    start_ox = block_x * T_W

    offs_y = start_oy + tl.arange(0, T_H)   # (T_H,)
    offs_x = start_ox + tl.arange(0, T_W)   # (T_W,)

    oy = offs_y[:, None]   # (T_H,1)
    ox = offs_x[None, :]   # (1,T_W)

    oy_flat = oy.reshape(-1)  # (T_H*T_W,)
    ox_flat = ox.reshape(-1)  # (T_H*T_W,)

    mask_out = (oy_flat < out_H) & (ox_flat < out_W)

    in_base = inp_ptr + ((n * C + c) * H * W)
    out_base = out_ptr + ((n * C + c) * out_H * out_W)

    # accumulator (flattened tile)
    tile_size = T_H * T_W
    acc = tl.zeros((tile_size,), dtype=tl.float32)

    # weight base for this channel: c * K * K
    w_base = w_ptr + c * K * K

    # Preload K*K kernel weights once (fp16 -> cast to fp32) into a small list of scalars
    w_vals = []
    for idx in range(0, K * K):
        w_fp16 = tl.load(w_base + idx, other=0.0, dtype=tl.float16)
        w_vals.append(tl.cast(w_fp16, tl.float32))

    # small helper aranges (constexpr lengths)
    range_TW = tl.arange(0, T_W)   # (T_W,)
    range_TH = tl.arange(0, T_H)   # (T_H,)

    # iterate kernel spatially, prefetching a horizontal patch per ky and reusing it for all kx
    for ky in range(0, K):
        iy = oy * stride + ky - pad   # (T_H,1)
        mask_y = (iy >= 0) & (iy < H)  # (T_H,1)
        row_offsets = iy * W           # (T_H,1)

        # compute base input x coordinate (for left-most tile column) and prefetch length
        base_ix = start_ox * stride - pad   # scalar
        total_len = (T_W - 1) * stride + K  # constexpr (since T_W, K, stride are constexpr)

        # build contiguous patch offsets for each row: shape (T_H, total_len)
        idxs = tl.arange(0, total_len)  # (total_len,)
        # patch_offsets: (T_H, total_len) -> flatten below for a single global load per row block
        patch_offsets = (row_offsets + base_ix)[:, None] + idxs[None, :]

        # bounds mask for the patch (1, total_len) broadcast with mask_y (T_H,1)
        patch_x = base_ix + idxs  # (total_len,)
        mask_patch = (patch_x >= 0) & (patch_x < W)  # (total_len,)
        mask_patch = mask_patch[None, :]  # (1, total_len)
        mask_full = mask_y & mask_patch   # (T_H, total_len)
        mask_flat = mask_full.reshape(-1)

        flat_offsets = patch_offsets.reshape(-1)

        # load the entire patch for this ky (fp16) and cast to fp32
        vals_patch_fp16 = tl.load(in_base + flat_offsets, mask=mask_flat, other=0.0, dtype=tl.float16)
        vals_patch = tl.cast(vals_patch_fp16, tl.float32)  # (T_H * total_len,)
        # reshape for easy indexing: (T_H, total_len)
        vals_patch = vals_patch.reshape((T_H, total_len))

        # reuse the preloaded patch for each kx
        for kx in range(0, K):
            # indices within a row patch for each output column in the tile
            idxs_in_patch = range_TW * stride + kx   # (T_W,)
            # gather values for the tile: vals_patch[:, idxs_in_patch] -> (T_H, T_W)
            vals_kx = vals_patch[:, idxs_in_patch]  # gather along second dim
            # flatten and accumulate
            w_val = w_vals[ky * K + kx]
            acc += vals_kx.reshape(-1) * w_val

    # add bias (assume bias tensor always provided as fp32)
    bias = tl.load(b_ptr + c)
    acc = acc + bias

    # fused ReLU6 (depthwise conv in MBConv is followed by ReLU6)
    acc = tl.maximum(acc, 0.0)
    acc = tl.minimum(acc, 6.0)

    # cast accumulator to fp16 once and store
    acc_fp16 = tl.cast(acc, tl.float16)
    out_offsets = oy_flat * out_W + ox_flat
    tl.store(out_base + out_offsets, acc_fp16, mask=mask_out)


def _fuse_conv_bn(conv: nn.Conv2d, bn: nn.BatchNorm2d):
    """
    Fold BatchNorm2d parameters into Conv2d weights and biases for inference.
    Supports grouped convs (including depthwise).
    """
    W = conv.weight.clone().detach().float()  # (out_ch, in_ch/groups, kH, kW)
    if conv.bias is not None:
        b_conv = conv.bias.clone().detach().float()
    else:
        b_conv = torch.zeros(W.shape[0], dtype=torch.float32, device=W.device)

    running_mean = bn.running_mean.clone().detach().float()
    running_var = bn.running_var.clone().detach().float()
    eps = float(bn.eps)

    if bn.affine:
        gamma = bn.weight.clone().detach().float()
        beta = bn.bias.clone().detach().float()
        invstd = 1.0 / torch.sqrt(running_var + eps)
        shape = [W.shape[0]] + [1] * (W.dim() - 1)
        scale = (gamma * invstd).reshape(shape)  # (out_ch,1,1,1)
        W_fused = W * scale
        b_fused = beta + (b_conv - running_mean) * (gamma * invstd)
    else:
        invstd = 1.0 / torch.sqrt(running_var + eps)
        shape = [W.shape[0]] + [1] * (W.dim() - 1)
        scale = invstd.reshape(shape)
        W_fused = W * scale
        b_fused = (b_conv - running_mean) * invstd

    return W_fused.contiguous(), b_fused.contiguous()


def depthwise_conv_triton(x: torch.Tensor, conv_fused_dw: nn.Conv2d):
    """
    Executes the depthwise conv (BN folded into weights/bias) using Triton kernel that
    fuses bias + ReLU6.

    Changes for performance:
    - Inputs and weights are passed as fp16 so the kernel can load them as fp16 and cast
      to fp32 for accumulation (mixed precision).
    - Bias remains fp32 for numerical stability.
    - Kernel is tiled in 2D (T_H x T_W) and autotuned via AUTOTUNE_CONFIGS_DW.
    - Kernel writes fp16 outputs (after fp32 accumulation + clamp) to reduce downstream traffic.
    """
    assert x.is_cuda and conv_fused_dw.weight.is_cuda, "Inputs must be on CUDA"
    N, C, H, W = x.shape

    # kernel params
    kH, kW = conv_fused_dw.kernel_size
    assert kH == kW, "Only square kernels supported"
    K = kH
    stride = conv_fused_dw.stride[0]
    pad = conv_fused_dw.padding[0]

    out_H = (H + 2 * pad - K) // stride + 1
    out_W = (W + 2 * pad - K) // stride + 1
    out_HW = out_H * out_W

    # Ensure input is fp16 contiguous NCHW (kernel will load as fp16)
    x_contig = x.contiguous()
    if x_contig.dtype != torch.float16:
        x_contig = x_contig.half().contiguous()

    # Prepare weight layout as fp16 to halve weight memory traffic: (C, 1, K, K) -> (C, K, K)
    w = conv_fused_dw.weight.detach().to(device=x.device, dtype=torch.float16).contiguous()
    if w.dim() == 4:
        w_flat = w.view(C, K, K)
    else:
        w_flat = w.view(C, K, K)

    # keep bias as fp32
    b = conv_fused_dw.bias.detach().to(device=x.device, dtype=torch.float32).contiguous() if conv_fused_dw.bias is not None else torch.zeros(C, device=x.device, dtype=torch.float32)

    # Allocate output as fp16 to match kernel's fp16 stores and reduce memory traffic
    out = torch.empty((N, C, out_H, out_W), device=x.device, dtype=torch.float16)

    def grid(meta):
        T_H = meta["T_H"]
        T_W = meta["T_W"]
        blocks_H = (out_H + T_H - 1) // T_H
        blocks_W = (out_W + T_W - 1) // T_W
        num_blocks = blocks_H * blocks_W
        return (C, N, num_blocks)

    _depthwise_conv2d_kernel[grid](x_contig, w_flat, b, out, N, C, H, W, out_H, out_W, K, pad, stride, out_HW)

    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, expand_ratio):
        """
        MBConv block optimized for inference:

        - Training mode: original modules (Conv -> BN -> ReLU6) are preserved.
        - Inference mode:
            * BatchNorm folded into conv weights (for expand / depthwise / project).
            * Pointwise (1x1) convs (expand & project) are executed as fused Conv2d with bias,
              using channels_last + AMP fp16 to leverage cuDNN NHWC fast paths on Ampere.
            * Depthwise conv is executed via a high-throughput Triton kernel that fuses bias + ReLU6.
            * Identity/residual addition is fused in-memory where possible (matching layouts/dtypes).
        """
        super(ModelNew, self).__init__()

        self.use_residual = (stride == 1 and in_channels == out_channels)
        hidden_dim = in_channels * expand_ratio

        # Original modules (kept for training semantics)
        if expand_ratio != 1:
            self.expand_conv = nn.Sequential(
                nn.Conv2d(in_channels, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True)
            )
        else:
            self.expand_conv = None

        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=kernel_size, stride=stride,
                      padding=(kernel_size - 1) // 2, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True)
        )

        self.project_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        # Prepare fused Conv2d modules for inference (BN folded into weights + bias)
        # Expand fused (1x1) - prefer channels_last layout for weights to favor cuDNN NHWC
        self._expand_conv_fused = None
        if self.expand_conv is not None:
            conv_e = self.expand_conv[0]
            bn_e = self.expand_conv[1]
            Wf_e, bf_e = _fuse_conv_bn(conv_e, bn_e)
            Wf_e = Wf_e.contiguous(memory_format=torch.channels_last)
            bf_e = bf_e.contiguous()
            conv_fused_e = nn.Conv2d(
                conv_e.in_channels,
                conv_e.out_channels,
                kernel_size=conv_e.kernel_size,
                stride=conv_e.stride,
                padding=conv_e.padding,
                dilation=conv_e.dilation,
                groups=conv_e.groups,
                bias=True,
            )
            conv_fused_e.weight = nn.Parameter(Wf_e)
            conv_fused_e.bias = nn.Parameter(bf_e)
            self._expand_conv_fused = conv_fused_e

        # Depthwise fused - will run using Triton kernel (weights + bias folded)
        conv_dw = self.depthwise_conv[0]
        bn_dw = self.depthwise_conv[1]
        Wf_dw, bf_dw = _fuse_conv_bn(conv_dw, bn_dw)
        # Keep fused depthwise in a small Conv2d container for ease of device movement/serialization.
        conv_fused_dw = nn.Conv2d(
            conv_dw.in_channels,
            conv_dw.out_channels,
            kernel_size=conv_dw.kernel_size,
            stride=conv_dw.stride,
            padding=conv_dw.padding,
            dilation=conv_dw.dilation,
            groups=conv_dw.groups,
            bias=True,
        )
        conv_fused_dw.weight = nn.Parameter(Wf_dw)
        conv_fused_dw.bias = nn.Parameter(bf_dw)
        self._depthwise_conv_fused = conv_fused_dw

        # Project fused (1x1) - prefer channels_last layout for weights
        conv_p = self.project_conv[0]
        bn_p = self.project_conv[1]
        Wf_p, bf_p = _fuse_conv_bn(conv_p, bn_p)
        Wf_p = Wf_p.contiguous(memory_format=torch.channels_last)
        bf_p = bf_p.contiguous()
        conv_fused_p = nn.Conv2d(
            conv_p.in_channels,
            conv_p.out_channels,
            kernel_size=conv_p.kernel_size,
            stride=conv_p.stride,
            padding=conv_p.padding,
            dilation=conv_p.dilation,
            groups=conv_p.groups,
            bias=True,
        )
        conv_fused_p.weight = nn.Parameter(Wf_p)
        conv_fused_p.bias = nn.Parameter(bf_p)
        self._project_conv_fused = conv_fused_p

        # Tune cudnn + TF32 for Ampere devices; allows fast pointwise convs
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True

    def forward(self, x):
        identity = x

        # Training path: preserve original MBConv block (Conv -> BN -> ReLU6)
        if self.training:
            if self.expand_conv is not None:
                x = self.expand_conv(x)
            x = self.depthwise_conv(x)
            x = self.project_conv(x)
            if self.use_residual:
                x = x + identity
            return x

        # Inference path: optimized
        # We'll run expand (if present) and project as fused 1x1 convs using channels_last + AMP fp16.
        # The depthwise conv runs via Triton in fp32 NCHW and fuses bias+ReLU6.

        idt = identity  # keep original identity for residual add

        # Expand (1x1) if present - keep NCHW to avoid format copies
        if self._expand_conv_fused is not None:
            x = x.contiguous()  # NCHW
            with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
                x = self._expand_conv_fused(x)
                x = F.relu6(x, inplace=True)
            # Triton expects fp16 NCHW inputs; cast without changing memory format
            x = x.to(dtype=torch.float16).contiguous()
        else:
            x = x.contiguous().to(dtype=torch.float16)

        # Depthwise via Triton (expects fp16 NCHW inputs)
        x = depthwise_conv_triton(x, self._depthwise_conv_fused)

        # Project conv (1x1) - execute in NCHW under autocast to avoid format copies
        x = x.contiguous()  # ensure contiguous NCHW
        # autocast will run conv in fp16 for speed
        with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
            x = self._project_conv_fused(x)
        # Convert back to fp32 for residual add
        x = x.to(dtype=torch.float32)

        if self.use_residual:
            # Prepare identity for addition: match dtype and memory format
            idt_proc = idt
            if idt_proc.dtype != x.dtype:
                idt_proc = idt_proc.to(dtype=x.dtype)
            # If x is channels_last, convert identity accordingly to enable fast add
            if x.is_contiguous(memory_format=torch.channels_last):
                idt_proc = idt_proc.contiguous(memory_format=torch.channels_last)
            x = x + idt_proc

        return x