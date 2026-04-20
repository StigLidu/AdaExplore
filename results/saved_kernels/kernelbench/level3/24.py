import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

# Autotune configs tuned for NVIDIA A6000 (Ampere) with larger tiles, larger K tiles and NUM_TILES as a constexpr.
# We include NUM_TILES in the config so the Triton autotuner can compile-time specialize and unroll the inner tile loop.
AUTOTUNE_CONFIGS_FUSED = [
    triton.Config({"TILE_M": 128, "BLOCK_N": 256, "BLOCK_K": 128, "BLOCK_J": 128, "NUM_TILES": 2}, num_warps=8, num_stages=3),
    triton.Config({"TILE_M": 64,  "BLOCK_N": 256, "BLOCK_K": 128, "BLOCK_J": 128, "NUM_TILES": 2}, num_warps=8, num_stages=3),
    triton.Config({"TILE_M": 64,  "BLOCK_N": 256, "BLOCK_K": 64,  "BLOCK_J": 128, "NUM_TILES": 2}, num_warps=8, num_stages=2),
    triton.Config({"TILE_M": 128, "BLOCK_N": 128, "BLOCK_K": 128, "BLOCK_J": 128, "NUM_TILES": 2}, num_warps=8, num_stages=3),
    triton.Config({"TILE_M": 64,  "BLOCK_N": 128, "BLOCK_K": 64,  "BLOCK_J": 64,  "NUM_TILES": 2}, num_warps=4, num_stages=2),
    triton.Config({"TILE_M": 32,  "BLOCK_N": 256, "BLOCK_K": 128, "BLOCK_J": 128, "NUM_TILES": 2}, num_warps=8, num_stages=2),
    # Some smaller blocks for different occupancy
    triton.Config({"TILE_M": 32,  "BLOCK_N": 64,  "BLOCK_K": 64,  "BLOCK_J": 64,  "NUM_TILES": 2}, num_warps=4, num_stages=2),
]

@triton.autotune(configs=AUTOTUNE_CONFIGS_FUSED, key=['B', 'S', 'K', 'N', 'OUTC'])
@triton.jit
def _conv1x1_bn_relu_pool_fc_kernel(
    x_ptr,       # device tensor: input X (NHWC channels-last if host passed that layout) (may be fp16)
    stride_B,    # stride in elements for batch dim
    stride_C,    # stride in elements for channel dim
    stride_H,    # stride in elements for height dim
    stride_W,    # stride in elements for width dim
    W_spatial,   # width (W) to compute h,w from flattened row index
    W_ptr,       # pointer to conv weights W_t (K, N) in fp16 (row-major)
    bias_ptr,    # pointer to conv folded bias (N,) in fp32
    fc_ptr,      # pointer to fc weights (N, OUTC) in fp16 (row-major)
    out_ptr,     # pointer to output logits (B, OUTC) in fp32
    B, S, K, N, OUTC,
    lda, ldb, ldf, ldo,   # row strides for W/FC buffers (in elements)
    invS,                 # 1.0 / S
    NUM_TILES: tl.constexpr, TILE_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_J: tl.constexpr
):
    """
    Fused kernel:
      - Reads X in the memory layout implied by the provided strides (host should pass channels-last for best perf).
      - Uses fp16 weights for conv and fc to reduce memory traffic.
      - Accumulates in fp32, applies folded BN bias, ReLU per spatial element, sums across spatial and multiplies by fc.
      - NUM_TILES is a compile-time parameter to allow unrolling the per-tile loop.
    """
    pid_b = tl.program_id(0)  # batch index
    pid_n = tl.program_id(1)  # conv-output channel block index
    pid_j = tl.program_id(2)  # class output block index

    batch = pid_b

    col_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)     # conv-out-channel indices
    col_mask = col_offsets < N

    class_offsets = pid_j * BLOCK_J + tl.arange(0, BLOCK_J)   # class indices
    class_mask = class_offsets < OUTC

    # Accumulator for output classes for this (batch, conv-block, class-block)
    acc_logits = tl.zeros((BLOCK_J,), dtype=tl.float32)

    # Load bias for conv block (fp32)
    bias_vals = tl.load(bias_ptr + col_offsets, mask=col_mask, other=0.0)  # (BLOCK_N,)

    # Load a block of fc weights: rows = conv channels (BLOCK_N), cols = classes (BLOCK_J) in fp16
    fc_ptrs = fc_ptr + (col_offsets[:, None] * ldf + class_offsets[None, :])
    fc_mask = col_mask[:, None] & class_mask[None, :]
    fc_block = tl.load(fc_ptrs, mask=fc_mask, other=0.0)  # fp16 values

    # Accumulator for per-conv-channel sum across spatial rows
    block_sum = tl.zeros((BLOCK_N,), dtype=tl.float32)

    # Process spatial rows in batches of NUM_TILES tiles; NUM_TILES is constexpr so this loop can be unrolled.
    row_start = 0
    while row_start < S:
        # Fixed-size per-batch tile accumulator (NUM_TILES, TILE_M, BLOCK_N)
        tile_acc = tl.zeros((NUM_TILES, TILE_M, BLOCK_N), dtype=tl.float32)

        k = 0
        while k < K:
            k_block = tl.arange(0, BLOCK_K)
            k_pos = k + k_block
            mask_k = k_pos < K

            # Compute common offsets for this k-block
            batch_offset = batch * stride_B
            channel_offsets = k_pos[None, :] * stride_C                      # (1, BLOCK_K)

            # Load W block once for this k-block and cast to fp32
            w_ptrs = W_ptr + (k_pos[:, None] * ldb + col_offsets[None, :])
            w_mask = mask_k[:, None] & col_mask[None, :]
            w_vals = tl.load(w_ptrs, mask=w_mask, other=0.0)
            w_f32 = tl.cast(w_vals, tl.float32)

            # For each tile in the compile-time NUM_TILES, load the corresponding X tile for this k-block and update accumulator
            for t in range(NUM_TILES):
                tile_row_start = row_start + t * TILE_M
                row_offsets = tile_row_start + tl.arange(0, TILE_M)       # (TILE_M,)
                row_mask = row_offsets < S
                h = row_offsets // W_spatial
                w = row_offsets % W_spatial

                spatial_offsets = (h[:, None] * stride_H) + (w[:, None] * stride_W)  # (TILE_M, 1)
                x_ptrs = x_ptr + (batch_offset + spatial_offsets + channel_offsets)   # (TILE_M, BLOCK_K)
                x_mask = row_mask[:, None] & mask_k[None, :]

                # Load X for this tile and k-block; cast and accumulate
                x_vals = tl.load(x_ptrs, mask=x_mask, other=0.0)
                x_f32 = tl.cast(x_vals, tl.float32)

                # Accumulate into per-tile accumulator
                tile_acc[t] += tl.dot(x_f32, w_f32)

            k += BLOCK_K

        # After finishing all K-blocks, finalize each tile: add bias, apply ReLU per spatial element,
        # zero-out invalid rows, sum across TILE_M and accumulate to block_sum
        for t in range(NUM_TILES):
            tile_row_start = row_start + t * TILE_M
            row_offsets = tile_row_start + tl.arange(0, TILE_M)
            row_mask = row_offsets < S

            t_acc = tile_acc[t]  # (TILE_M, BLOCK_N)
            t_acc += bias_vals[None, :]
            t_acc = tl.maximum(t_acc, 0.0)
            t_acc = tl.where(row_mask[:, None], t_acc, 0.0)
            tile_block_sum = tl.sum(t_acc, axis=0)  # (BLOCK_N,)
            block_sum += tile_block_sum

        row_start += TILE_M * NUM_TILES

    # After spatial accumulation, do small GEMM: (1 x BLOCK_N) @ (BLOCK_N x BLOCK_J) -> (1 x BLOCK_J)
    fc_block_f32 = tl.cast(fc_block, tl.float32)
    logits_inc = tl.dot(block_sum[None, :], fc_block_f32)  # (1, BLOCK_J)
    acc_logits += logits_inc[0]

    # Multiply by invS to compute mean and store into output
    out_ptrs = out_ptr + (batch * ldo + class_offsets)
    store_mask = class_mask
    tl.store(out_ptrs, acc_logits * invS, mask=store_mask)


def fused_conv_bn_relu_pool_fc(x: torch.Tensor, conv_w: torch.Tensor, conv_bias: torch.Tensor, fc_w: torch.Tensor):
    """
    Host wrapper for the fused Triton kernel.
    - x: (B, C_in, H, W) device tensor (preferably fp16 to reduce bandwidth)
    - conv_w: (out, in) float32 folded conv weights (W_fold)
    - conv_bias: (out,) float32 folded bias
    - fc_w: (OUTC, out) float32 fc weights
    Returns logits (B, OUTC) float32 on same device.
    """
    assert x.is_cuda and conv_w.is_cuda and conv_bias.is_cuda and fc_w.is_cuda

    B, C_in, H, W = x.shape
    S = H * W
    device = x.device

    # Convert weights to fp16 to lower device memory traffic
    # conv_w is (out, in) -> we want W_t = conv_w.T shape (K=in, N=out) in row-major
    conv_w_t = conv_w.t().contiguous().half()   # (K, N) fp16
    # fc_w is (OUTC, out) -> we want fc_t = fc_w.T shape (N=out, OUTC) in row-major
    fc_w_t = fc_w.t().contiguous().half()       # (N, OUTC) fp16

    # Ensure x is in channels-last (NHWC) contiguous fp16 layout to get coalesced channel loads on Ampere
    x_fp = x.contiguous(memory_format=torch.channels_last).half()

    # Prepare output tensor
    out = torch.empty((B, fc_w.shape[0]), device=device, dtype=torch.float32)

    # Strides in elements for x_fp (channels-last: N, H, W, C).
    # Kernel still expects arguments in order (stride_B, stride_C, stride_H, stride_W),
    # so map the channels-last strides into that ordering.
    sN, sH, sW, sC = x_fp.stride()
    stride_B, stride_C, stride_H, stride_W = sN, sC, sH, sW

    # Row strides for W/FC buffers (in elements)
    lda = C_in    # for conv_w_t (K rows -> each row length N)
    ldb = conv_w_t.shape[1]  # N
    ldf = fc_w_t.shape[1]    # OUTC
    ldo = out.shape[1]       # OUTC

    grid = lambda meta: (
        B,
        (conv_w_t.shape[1] + meta["BLOCK_N"] - 1) // meta["BLOCK_N"],     # number of conv blocks
        (out.shape[1] + meta["BLOCK_J"] - 1) // meta["BLOCK_J"],          # number of class blocks
    )

    # Launch kernel: pass tensor objects (Triton accepts device Tensor arguments)
    _conv1x1_bn_relu_pool_fc_kernel[grid](
        x_fp, stride_B, stride_C, stride_H, stride_W, W,
        conv_w_t, conv_bias, fc_w_t, out,
        B, S, C_in, conv_w_t.shape[1], out.shape[1],
        lda, ldb, ldf, ldo,
        1.0 / float(S)
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        """
        EfficientNetB2-like model with a highly fused Triton kernel that:
          - Fuses conv1x1 (with BN folded) + ReLU + global-avg-pool + final FC multiplication into a single kernel
            that reads the NCHW activation tensor directly via strides, uses fp16 weights, and accumulates in fp32.
          - Other layers remain as PyTorch modules for correctness (MBConv blocks, initial layers).
        """
        super(ModelNew, self).__init__()

        # Base feature extractor (same as original)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

        # MBConv blocks
        self.mbconv1 = self._make_mbconv_block(32, 96, 1, 3)
        self.mbconv2 = self._make_mbconv_block(96, 144, 2, 6)
        self.mbconv3 = self._make_mbconv_block(144, 192, 2, 6)
        self.mbconv4 = self._make_mbconv_block(192, 288, 2, 6)
        self.mbconv5 = self._make_mbconv_block(288, 384, 1, 6)

        # Final conv and bn
        self.conv_final = nn.Conv2d(384, 1408, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_final = nn.BatchNorm2d(1408)
        self.relu_final = nn.ReLU(inplace=True)

        # Keep a nn.Linear module for parameter storage
        self.fc = nn.Linear(1408, num_classes)

    def _make_mbconv_block(self, in_channels, out_channels, stride, expand_ratio):
        layers = []
        expanded_channels = in_channels * expand_ratio

        # Expansion phase
        if expand_ratio != 1:
            layers.append(nn.Conv2d(in_channels, expanded_channels, kernel_size=1, stride=1, padding=0, bias=False))
            layers.append(nn.BatchNorm2d(expanded_channels))
            layers.append(nn.ReLU(inplace=True))

        # Depthwise convolution
        layers.append(nn.Conv2d(expanded_channels, expanded_channels, kernel_size=3, stride=stride, padding=1, groups=expanded_channels, bias=False))
        layers.append(nn.BatchNorm2d(expanded_channels))
        layers.append(nn.ReLU(inplace=True))

        # Squeeze and Excitation
        layers.append(nn.AdaptiveAvgPool2d((1, 1)))
        layers.append(nn.Conv2d(expanded_channels, expanded_channels // 4, kernel_size=1, stride=1, padding=0, bias=False))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(expanded_channels // 4, expanded_channels, kernel_size=1, stride=1, padding=0, bias=False))
        layers.append(nn.Sigmoid())

        # Output phase
        layers.append(nn.Conv2d(expanded_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass:
          - Base feature extractor in PyTorch.
          - If training or not on CUDA, use the standard PyTorch path to preserve BN behaviour.
          - Otherwise (inference on CUDA), fold BN into conv weights and run the fused Triton kernel.
        """
        # Feature extraction
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.mbconv1(x)
        x = self.mbconv2(x)
        x = self.mbconv3(x)
        x = self.mbconv4(x)
        x = self.mbconv5(x)

        device = x.device
        dtype = x.dtype

        # If training or CPU, fallback to reference implementation
        if self.bn_final.training or not x.is_cuda:
            conv_out = self.relu_final(self.bn_final(self.conv_final(x)))
            pooled = F.adaptive_avg_pool2d(conv_out, (1, 1))
            pooled = torch.flatten(pooled, 1)
            logits = F.linear(pooled, self.fc.weight, self.fc.bias)
            return logits

        # Inference + CUDA path: use the fused Triton kernel

        # Fold BN into conv weights: conv_final.weight shape (out, in, 1, 1)
        conv_w = self.conv_final.weight.to(device=device, dtype=torch.float32).contiguous()
        conv_w_flat = conv_w.view(conv_w.shape[0], conv_w.shape[1])  # (out, in)

        bn = self.bn_final
        running_mean = bn.running_mean.to(device=device, dtype=torch.float32)
        running_var = bn.running_var.to(device=device, dtype=torch.float32)
        eps = torch.tensor(bn.eps, device=device, dtype=torch.float32)

        if bn.affine:
            gamma = bn.weight.to(device=device, dtype=torch.float32)
            beta = bn.bias.to(device=device, dtype=torch.float32)
        else:
            gamma = torch.ones_like(running_mean, device=device, dtype=torch.float32)
            beta = torch.zeros_like(running_mean, device=device, dtype=torch.float32)

        invstd = torch.rsqrt(running_var + eps)
        scale = gamma * invstd
        bias = beta - scale * running_mean

        # Fold scale into conv weights and bias
        W_fold = conv_w_flat * scale.view(-1, 1)   # (out, in)
        b_fold = bias                               # (out,)

        # Prepare fc weights (OUTC, out)
        fc_w = self.fc.weight.to(device=device, dtype=torch.float32).contiguous()  # (OUTC, out)
        fc_b = self.fc.bias.to(device=device, dtype=torch.float32) if self.fc.bias is not None else None

        # Run fused kernel: pass activation tensor and folded weights
        logits = fused_conv_bn_relu_pool_fc(x, W_fold, b_fold, fc_w)  # (B, num_classes) float32

        # Add fc bias if present
        if fc_b is not None:
            logits.add_(fc_b.view(1, -1))

        return logits.to(dtype=dtype)


# Keep the same input helpers as the original file
batch_size = 2
num_classes = 1000

def get_inputs():
    return [torch.rand(batch_size, 3, 224, 224)]

def get_init_inputs():
    return [num_classes]