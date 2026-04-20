import torch
import torch.nn as nn
import torch.nn.functional as F

# Try to import Triton. If unavailable, fall back to efficient .copy_ implementations.
try:
    import triton
    import triton.language as tl

    # Autotune configs chosen to favor very large coalesced transfers typical for A6000 (Ampere).
    AUTOTUNE_CONFIGS_COPY = [
        triton.Config({"BLOCK": 8192, "CHANNELS_PER_PROG": 64}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK": 4096, "CHANNELS_PER_PROG": 64}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK": 4096, "CHANNELS_PER_PROG": 32}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK": 2048, "CHANNELS_PER_PROG": 32}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK": 1024, "CHANNELS_PER_PROG": 16}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK": 512,  "CHANNELS_PER_PROG": 8},  num_warps=4, num_stages=2),
        triton.Config({"BLOCK": 256,  "CHANNELS_PER_PROG": 4},  num_warps=4, num_stages=2),
    ]

    @triton.autotune(configs=AUTOTUNE_CONFIGS_COPY, key=['N', 'C_src', 'elems_per_channel'])
    @triton.jit
    def _batched_channel_copy_kernel(
        src_ptr,             # pointer to source tensor flattened (fp32)
        dst_ptr,             # pointer to destination tensor flattened (fp32)
        N,                   # batch size
        C_src,               # number of channels in src
        total_C,             # number of channels in destination
        dst_channel_offset,  # channel offset in destination where C_src should be written
        elems_per_channel,   # H * W
        BLOCK: tl.constexpr,
        CHANNELS_PER_PROG: tl.constexpr
    ):
        # tile_idx indexes (sample * tiles_per_sample + tile_in_sample)
        tile_idx = tl.program_id(0)
        block_idx = tl.program_id(1)

        # number of tiles (in channel dimension) per sample
        tiles_per_sample = (C_src + CHANNELS_PER_PROG - 1) // CHANNELS_PER_PROG
        total_tiles = N * tiles_per_sample
        # bounds check for program ids
        if tile_idx >= total_tiles:
            return

        sample = tile_idx // tiles_per_sample
        tile_in_sample = tile_idx % tiles_per_sample
        channel_base = tile_in_sample * CHANNELS_PER_PROG

        start = block_idx * BLOCK
        offs = tl.arange(0, BLOCK)
        cur = start + offs
        mask_spatial = cur < elems_per_channel

        # loop over the small constexpr channel tile for vectorized coalesced loads/stores
        for c_inner in range(CHANNELS_PER_PROG):
            c = channel_base + c_inner
            # runtime check on channel bounds
            if c < C_src:
                base_src = (sample * C_src + c) * elems_per_channel
                base_dst = (sample * total_C + dst_channel_offset + c) * elems_per_channel
                vals = tl.load(src_ptr + base_src + cur, mask=mask_spatial, other=0.0)
                tl.store(dst_ptr + base_dst + cur, vals, mask=mask_spatial)

    def triton_batched_channel_copy(src: torch.Tensor, dst: torch.Tensor, dst_channel_offset: int):
        """
        Copy src (N, C_src, H, W) into dst (N, total_C, H, W) at channel offset dst_channel_offset.
        Uses Triton for large contiguous workloads and falls back to .copy_ for small/non-contiguous tensors.
        Heuristics tuned for Ampere (A6000) to balance kernel-launch overhead and throughput.
        """
        assert src.is_cuda and dst.is_cuda, "tensors must be on CUDA for Triton path"
        assert src.dtype == dst.dtype, "src and dst must have same dtype"

        N, C_src, H, W = src.shape
        total_C = dst.shape[1]
        elems_per_channel = H * W
        n_elements = src.numel()

        # If tensors are non-contiguous or extremely small, rely on efficient device memcpy.
        if not src.is_contiguous() or not dst.is_contiguous():
            dst[:, dst_channel_offset:dst_channel_offset + C_src, :, :].copy_(src)
            return

        # Heuristics to decide whether to use Triton:
        # - Avoid Triton for very small total elements due to launch overhead.
        # - Avoid Triton when spatial tile is tiny (e.g., H*W small) so per-program work is too small.
        SMALL_TOTAL_ELEMS = 128 * 1024   # 128K elements threshold
        SMALL_SPATIAL = 16               # if H*W small, prefer memcpy

        if n_elements <= SMALL_TOTAL_ELEMS or elems_per_channel <= SMALL_SPATIAL:
            dst[:, dst_channel_offset:dst_channel_offset + C_src, :, :].copy_(src)
            return

        # Use Triton for medium-to-large workloads for better throughput on A6000
        grid = lambda meta: (
            N * ((C_src + meta['CHANNELS_PER_PROG'] - 1) // meta['CHANNELS_PER_PROG']),
            (elems_per_channel + meta['BLOCK'] - 1) // meta['BLOCK']
        )
        _batched_channel_copy_kernel[grid](src, dst, N, C_src, total_C, dst_channel_offset, elems_per_channel)

except Exception:
    # Fall back if Triton isn't available: use .copy_ which is efficient for contiguous GPU tensors.
    def triton_batched_channel_copy(src: torch.Tensor, dst: torch.Tensor, dst_channel_offset: int):
        dst[:, dst_channel_offset:dst_channel_offset + src.shape[1], :, :].copy_(src)


class DenseBlockNew(nn.Module):
    """
    DenseBlock optimized to:
      - preallocate the concatenation buffer once per forward,
      - write initial input and produced features into the buffer with large-device memcpy or Triton for large transfers,
      - remove Dropout (probability 0.0),
      - optionally fold BatchNorm into Conv in eval mode to save BN kernel.
    """
    def __init__(self, num_layers: int, num_input_features: int, growth_rate: int):
        super(DenseBlockNew, self).__init__()
        self.num_layers = num_layers
        self.num_input_features = num_input_features
        self.growth_rate = growth_rate

        # Build BatchNorm and Conv lists (avoid tiny Sequentials)
        bns = []
        convs = []
        for i in range(num_layers):
            in_ch = num_input_features + i * growth_rate
            bns.append(nn.BatchNorm2d(in_ch))
            convs.append(nn.Conv2d(in_ch, growth_rate, kernel_size=3, padding=1, bias=False))
        self.bns = nn.ModuleList(bns)
        self.convs = nn.ModuleList(convs)
        # track whether bn->conv fusion applied (only in eval mode)
        self._bn_fused = False

    def _maybe_fuse_bn_into_conv(self):
        """
        Fuse BatchNorm parameters into Conv weights/bias when in eval() to remove BN kernel.
        This is done once and cached for the module instance.
        """
        if self._bn_fused:
            return
        # Only fuse when in eval mode
        if self.training:
            return

        # Iterate and fuse where possible
        for idx in range(len(self.convs)):
            bn = self.bns[idx]
            conv = self.convs[idx]
            if not isinstance(bn, nn.BatchNorm2d):
                continue
            # Prepare BN params (handle affine/non-affine)
            running_mean = bn.running_mean if bn.running_mean is not None else torch.zeros(conv.out_channels, device=conv.weight.device)
            running_var = bn.running_var if bn.running_var is not None else torch.ones(conv.out_channels, device=conv.weight.device)
            eps = bn.eps
            if bn.weight is not None:
                gamma = bn.weight.detach().to(conv.weight.device)
                beta = bn.bias.detach().to(conv.weight.device) if bn.bias is not None else torch.zeros_like(gamma)
            else:
                gamma = torch.ones(conv.out_channels, device=conv.weight.device)
                beta = torch.zeros(conv.out_channels, device=conv.weight.device)

            # conv.weight shape: (out_ch, in_ch, kH, kW)
            W = conv.weight.detach()
            scale = (gamma / torch.sqrt(running_var + eps)).reshape(-1, 1, 1, 1)
            fused_W = W * scale
            fused_bias = beta - gamma * running_mean / torch.sqrt(running_var + eps)

            # Replace conv weights and ensure conv has a bias tensor
            conv.weight.data.copy_(fused_W)
            conv.bias = nn.Parameter(fused_bias)

            # Replace the BN by Identity to avoid it in forward
            self.bns[idx] = nn.Identity()

        self._bn_fused = True

    def forward(self, x: torch.Tensor):
        N, C_in, H, W = x.shape
        total_channels = C_in + self.num_layers * self.growth_rate

        # Preallocate the output buffer that will hold concatenated features
        out = x.new_empty((N, total_channels, H, W))

        # Copy initial input into output buffer. Use Triton for large contiguous workloads.
        triton_batched_channel_copy(x, out, dst_channel_offset=0)

        # If in eval mode attempt BN->Conv fusion once to save BN kernel calls
        if not self.training:
            # Only attempt fusion if not already done
            self._maybe_fuse_bn_into_conv()

        cur_channels = C_in

        # To avoid many small Triton launches for produced features, prefer device memcpy (copy_) for per-layer writes.
        # Triton path is reserved for initial large copy and large externally provided tensors.
        for bn, conv in zip(self.bns, self.convs):
            # Provide a view of the currently filled channels as input (no copy)
            input_view = out[:, :cur_channels, :, :]

            # Apply BatchNorm (which may now be Identity) then ReLU inplace then Conv
            x_bn = bn(input_view)
            x_relu = F.relu(x_bn, inplace=True)
            new_feature = conv(x_relu)  # shape: (N, growth_rate, H, W)

            # Default to efficient device memcpy for per-layer writes. This avoids repeated Triton launches.
            out[:, cur_channels:cur_channels + self.growth_rate, :, :].copy_(new_feature)

            cur_channels += self.growth_rate

        return out


class TransitionLayer(nn.Module):
    """
    Transition layer implemented explicitly to reduce tiny Sequential overhead.
    """
    def __init__(self, num_input_features: int, num_output_features: int):
        super(TransitionLayer, self).__init__()
        self.bn = nn.BatchNorm2d(num_input_features)
        self.conv = nn.Conv2d(num_input_features, num_output_features, kernel_size=1, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.bn(x)
        x = F.relu(x, inplace=True)
        x = self.conv(x)
        x = self.pool(x)
        return x


class ModelNew(nn.Module):
    """
    DenseNet-like model optimized for throughput:
      - DenseBlock replaced with DenseBlockNew which preallocates concatenation buffers
        and uses Triton-accelerated batched copies for large initial moves.
      - Removed no-op Dropout layers and reduced tiny Sequential overhead.
      - BatchNorm folded into Conv in eval mode inside DenseBlock to eliminate BN kernel where possible.
      - Final pooling replaced with spatial mean to avoid adaptive pooling + view overhead.
    """
    def __init__(self, growth_rate: int = 32, num_classes: int = 1000):
        super(ModelNew, self).__init__()

        # Explicit initial layers (avoid tiny Sequential)
        self.conv0 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn0 = nn.BatchNorm2d(64)
        # Use functional ReLU in forward to avoid extra module overhead
        self.maxpool0 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Dense block configuration (DenseNet121)
        num_features = 64
        block_layers = [6, 12, 24, 16]

        self.dense_blocks = nn.ModuleList()
        self.transition_layers = nn.ModuleList()

        for i, num_layers in enumerate(block_layers):
            block = DenseBlockNew(num_layers=num_layers, num_input_features=num_features, growth_rate=growth_rate)
            self.dense_blocks.append(block)
            num_features = num_features + num_layers * growth_rate

            if i != len(block_layers) - 1:
                transition = TransitionLayer(num_input_features=num_features, num_output_features=num_features // 2)
                self.transition_layers.append(transition)
                num_features = num_features // 2

        # Final batch norm and classifier
        self.final_bn = nn.BatchNorm2d(num_features)
        self.classifier = nn.Linear(num_features, num_classes)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv0(x)
        x = self.bn0(x)
        x = F.relu(x, inplace=True)
        x = self.maxpool0(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)

        for i, block in enumerate(self.dense_blocks):
            x = block(x)
            if i != len(self.dense_blocks) - 1:
                x = self.transition_layers[i](x)

        x = self.final_bn(x)
        x = F.relu(x, inplace=True)
        # Use spatial mean instead of adaptive_avg_pool2d + view
        x = x.mean(dim=(2, 3))
        x = self.classifier(x)
        return x