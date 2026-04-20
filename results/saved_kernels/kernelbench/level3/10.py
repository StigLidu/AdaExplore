import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

# Triton kernel: In-place elementwise Add (out += identity) followed by ReLU on NCHW layout.
# Two kernels are provided:
#  - _add_relu_inplace_nchw_contig: fast path for contiguous NCHW tensors. Vectorizes over spatial
#    dimension (treated as a single linear dimension S=H*W) and uses BLOCK_S as a multiple of 4.
#  - _add_relu_inplace_nchw_tile: generic kernel that uses a merged "spatial_block" program-id
#    (spatial_block -> bh, bw) to reduce grid dimensionality compared to separate bh/bw ids.
@triton.jit
def _add_relu_inplace_nchw_contig_maskless(
    out_ptr,        # pointer to tensor to be modified (NCHW contiguous)
    id_ptr,         # pointer to identity tensor (NCHW contiguous)
    C, N, S,        # S = H * W
    out_stride_n, out_stride_c,
    id_stride_n, id_stride_c,
    BLOCK_C: tl.constexpr,   # channels per tile (constexpr)
    BLOCK_S: tl.constexpr,   # spatial elements per tile (constexpr, multiple of 16)
):
    # program ids: c_block, n, spatial_block
    c_block = tl.program_id(0)
    n = tl.program_id(1)
    s_block = tl.program_id(2)

    # channel indices for this program (assumed full BLOCK_C; caller ensures c_block only iterates full blocks)
    c_base = c_block * BLOCK_C
    ch_idx = c_base + tl.arange(0, BLOCK_C)                 # [BLOCK_C]

    # spatial (linearized H*W) offsets for this program (assumed full BLOCK_S; caller ensures s_block only iterates full blocks)
    s_base = s_block * BLOCK_S
    offs_s = s_base + tl.arange(0, BLOCK_S)                 # [BLOCK_S]

    # compute base addresses per channel (contiguous spatial)
    base_out = n * out_stride_n + ch_idx * out_stride_c     # [BLOCK_C]
    base_out = base_out[:, None]                            # [BLOCK_C,1]
    base_id = n * id_stride_n + ch_idx * id_stride_c
    base_id = base_id[:, None]

    offs_s = offs_s[None, :]                                # [1, BLOCK_S]
    addrs_out = base_out + offs_s                           # [BLOCK_C, BLOCK_S]
    addrs_id = base_id + offs_s

    # flatten addresses and perform maskless loads/stores (safe because we only launch for full blocks)
    addrs_out = tl.reshape(addrs_out, (-1,))
    addrs_id = tl.reshape(addrs_id, (-1,))

    out_vals = tl.load(out_ptr + addrs_out)
    id_vals = tl.load(id_ptr + addrs_id)

    res = out_vals + id_vals
    res = tl.maximum(res, 0.0)

    tl.store(out_ptr + addrs_out, res)


@triton.jit
def _add_relu_inplace_nchw_contig_masked(
    out_ptr,        # pointer to tensor to be modified (NCHW contiguous)
    id_ptr,         # pointer to identity tensor (NCHW contiguous)
    C, N, S,        # S = H * W
    out_stride_n, out_stride_c,
    id_stride_n, id_stride_c,
    s_block_offset,             # linear offset (in elements) to add to s_base for this launch (regular arg)
    BLOCK_C: tl.constexpr,      # channels per tile (constexpr)
    BLOCK_S: tl.constexpr,      # spatial elements per tile (constexpr)
):
    # program ids: c_block, n, spatial_block (spatial_block enumerates only the tail blocks launched)
    c_block = tl.program_id(0)
    n = tl.program_id(1)
    s_block = tl.program_id(2)

    # channel indices for this program (may be partial; still mask channels)
    c_base = c_block * BLOCK_C
    ch_idx = c_base + tl.arange(0, BLOCK_C)                 # [BLOCK_C]
    mask_c = ch_idx < C

    # spatial offsets for this (tail) block - include s_block_offset
    s_base = s_block_offset + s_block * BLOCK_S
    offs_s = s_base + tl.arange(0, BLOCK_S)                 # [BLOCK_S]
    mask_s = offs_s < S

    base_out = n * out_stride_n + ch_idx * out_stride_c     # [BLOCK_C]
    base_out = base_out[:, None]                            # [BLOCK_C,1]
    base_id = n * id_stride_n + ch_idx * id_stride_c
    base_id = base_id[:, None]

    offs_s = offs_s[None, :]
    addrs_out = base_out + offs_s
    addrs_id = base_id + offs_s

    mask = mask_c[:, None] & mask_s[None, :]
    addrs_out = tl.reshape(addrs_out, (-1,))
    addrs_id = tl.reshape(addrs_id, (-1,))
    mask = tl.reshape(mask, (-1,))

    out_vals = tl.load(out_ptr + addrs_out, mask=mask, other=0.0)
    id_vals = tl.load(id_ptr + addrs_id, mask=mask, other=0.0)

    res = out_vals + id_vals
    res = tl.maximum(res, 0.0)

    tl.store(out_ptr + addrs_out, res, mask=mask, other=0.0)


@triton.jit
def _add_relu_inplace_nchw_tile_maskless(
    out_ptr,        # pointer to tensor to be modified (NCHW layout)
    id_ptr,         # pointer to identity tensor (NCHW layout)
    C, N, H, W,
    out_stride_n, out_stride_c, out_stride_h, out_stride_w,
    id_stride_n, id_stride_c, id_stride_h, id_stride_w,
    num_w_blocks_full,          # number of full blocks along W for the interior grid (regular arg)
    BLOCK_C: tl.constexpr,      # tile size in channel dimension (full)
    BLOCK_H: tl.constexpr,      # tile size in H dimension (full)
    BLOCK_W: tl.constexpr,      # tile size in W dimension (full)
):
    # block indices: c_block, n, spatial_block (merged HxW for the interior full grid)
    c_block = tl.program_id(0)
    n = tl.program_id(1)
    spatial_block = tl.program_id(2)

    # recover bh, bw from merged spatial_block (interior full grid)
    bh = spatial_block // num_w_blocks_full
    bw = spatial_block % num_w_blocks_full

    # channel indices for this program (assumed full BLOCK_C)
    c_base = c_block * BLOCK_C
    ch_idx = c_base + tl.arange(0, BLOCK_C)                 # [BLOCK_C]

    # spatial offsets (assumed full BLOCK_H,BLOCK_W)
    offs_h = bh * BLOCK_H + tl.arange(0, BLOCK_H)           # [BLOCK_H]
    offs_w = bw * BLOCK_W + tl.arange(0, BLOCK_W)           # [BLOCK_W]

    # compute strides and address base per channel
    base_out = n * out_stride_n + ch_idx * out_stride_c           # [BLOCK_C]
    base_out = base_out[:, None, None]                            # [BLOCK_C,1,1]
    base_id = n * id_stride_n + ch_idx * id_stride_c
    base_id = base_id[:, None, None]

    offs_h_out = (offs_h * out_stride_h)[None, :, None]         # [1,BLOCK_H,1]
    offs_w_out = (offs_w * out_stride_w)[None, None, :]         # [1,1,BLOCK_W]

    addrs_out = base_out + offs_h_out + offs_w_out                  # [BLOCK_C, BLOCK_H, BLOCK_W]
    addrs_id = base_id + (offs_h * id_stride_h)[None, :, None] + (offs_w * id_stride_w)[None, None, :]

    addrs_out = tl.reshape(addrs_out, (-1,))
    addrs_id = tl.reshape(addrs_id, (-1,))

    out_vals = tl.load(out_ptr + addrs_out)
    id_vals = tl.load(id_ptr + addrs_id)

    res = out_vals + id_vals
    res = tl.maximum(res, 0.0)

    tl.store(out_ptr + addrs_out, res)


@triton.jit
def _add_relu_inplace_nchw_tile_masked(
    out_ptr,        # pointer to tensor to be modified (NCHW layout)
    id_ptr,         # pointer to identity tensor (NCHW layout)
    C, N, H, W,
    out_stride_n, out_stride_c, out_stride_h, out_stride_w,
    id_stride_n, id_stride_c, id_stride_h, id_stride_w,
    bh_offset,        # starting block index in H (regular arg)
    bw_offset,        # starting block index in W (regular arg)
    tail_num_w,       # number of W-blocks in this tail group (regular arg)
    BLOCK_C: tl.constexpr,      # tile size in channel dimension
    BLOCK_H: tl.constexpr,      # tile size in H dimension
    BLOCK_W: tl.constexpr,      # tile size in W dimension
):
    # spatial_block enumerates blocks within the tail group; map to bh,bw using offsets
    c_block = tl.program_id(0)
    n = tl.program_id(1)
    spatial_block = tl.program_id(2)

    bh = bh_offset + (spatial_block // tail_num_w)
    bw = bw_offset + (spatial_block % tail_num_w)

    # channel indices for this program (may be partial; still mask channels)
    c_base = c_block * BLOCK_C
    ch_idx = c_base + tl.arange(0, BLOCK_C)                 # [BLOCK_C]
    mask_c = ch_idx < C

    # spatial offsets
    offs_h = bh * BLOCK_H + tl.arange(0, BLOCK_H)           # [BLOCK_H]
    offs_w = bw * BLOCK_W + tl.arange(0, BLOCK_W)           # [BLOCK_W]
    mask_h = offs_h < H
    mask_w = offs_w < W

    # compute strides and address base per channel
    base_out = n * out_stride_n + ch_idx * out_stride_c           # [BLOCK_C]
    base_out = base_out[:, None, None]                            # [BLOCK_C,1,1]
    base_id = n * id_stride_n + ch_idx * id_stride_c
    base_id = base_id[:, None, None]

    offs_h_out = (offs_h * out_stride_h)[None, :, None]         # [1,BLOCK_H,1]
    offs_w_out = (offs_w * out_stride_w)[None, None, :]         # [1,1,BLOCK_W]

    offs_h_id = (offs_h * id_stride_h)[None, :, None]
    offs_w_id = (offs_w * id_stride_w)[None, None, :]

    addrs_out = base_out + offs_h_out + offs_w_out                  # [BLOCK_C, BLOCK_H, BLOCK_W]
    addrs_id = base_id + offs_h_id + offs_w_id

    mask = mask_c[:, None, None] & mask_h[None, :, None] & mask_w[None, None, :]
    addrs_out = tl.reshape(addrs_out, (-1,))
    addrs_id = tl.reshape(addrs_id, (-1,))
    mask = tl.reshape(mask, (-1,))

    out_vals = tl.load(out_ptr + addrs_out, mask=mask, other=0.0)
    id_vals = tl.load(id_ptr + addrs_id, mask=mask, other=0.0)

    res = out_vals + id_vals
    res = tl.maximum(res, 0.0)

    tl.store(out_ptr + addrs_out, res, mask=mask, other=0.0)


def triton_add_relu_inplace_nchw(out: torch.Tensor, identity: torch.Tensor):
    """
    In-place out = relu(out + identity) using a Triton kernel tiled for NCHW layout.
    Two execution paths:
      - fast contiguous path (vectorized spatial tile) when both tensors are contiguous NCHW
      - generic merged-spatial path otherwise

    Tuning notes:
      - Use small BLOCK_C (8) and large spatial tiles for contiguous path to maximize contiguous vector loads.
      - For merged spatial path, make BLOCK_W large (64/32/16/8) and BLOCK_H small (1..4) so bw (W-block)
        varies fastest and threads in a warp access consecutive W addresses.
    """
    assert out.is_cuda and identity.is_cuda and out.dtype == torch.float32 and identity.dtype == torch.float32
    assert out.shape == identity.shape, "out and identity must have same shape"

    N, C, H, W = out.shape
    out_stride_n, out_stride_c, out_stride_h, out_stride_w = out.stride()
    id_stride_n, id_stride_c, id_stride_h, id_stride_w = identity.stride()

    # Fast contiguous path: handle common contiguous NCHW case with vectorized spatial loads/stores.
    if out.is_contiguous() and identity.is_contiguous():
        S = H * W
        # Prefer a small channel tile to reduce register pressure and allow long spatial tiles.
        BLOCK_C = 8 if C >= 8 else max(1, C)
        # Compute the maximum spatial tile such that BLOCK_C * BLOCK_S <= 1024 and prefer multiples of 16.
        if S >= 16:
            max_s = max(16, 1024 // BLOCK_C)
            # round down to multiple of 16
            BLOCK_S = (max_s // 16) * 16
            if BLOCK_S < 16:
                BLOCK_S = 16
            BLOCK_S = min(BLOCK_S, S)
        else:
            # Small S: use a multiple-of-4 fallback to keep vector loads friendly.
            BLOCK_S = max(4, (S // 4) * 4)

        num_c_blocks = (C + BLOCK_C - 1) // BLOCK_C
        num_s_blocks = (S + BLOCK_S - 1) // BLOCK_S

        grid = (num_c_blocks, N, num_s_blocks)

        _add_relu_inplace_nchw_contig[grid](
            out, identity,
            C, N, S,
            out_stride_n, out_stride_c,
            id_stride_n, id_stride_c,
            BLOCK_C=BLOCK_C, BLOCK_S=BLOCK_S
        )
        return out

    # Generic merged-spatial path: compute BLOCK_C,BLOCK_H,BLOCK_W and merge bh/bw into spatial_block.
    def _choose_block_w(W):
        if W >= 64:
            return 64
        elif W >= 32:
            return 32
        elif W >= 16:
            return 16
        elif W >= 8:
            return 8
        else:
            return 1

    BLOCK_W = _choose_block_w(W)
    # Use a small channel tile to avoid register pressure; if channels are tiny, let BLOCK_C = C.
    BLOCK_C = 8 if C >= 8 else max(1, C)
    # Prefer small H tile (1..4) so W-blocking yields long contiguous runs.
    BLOCK_H = 1024 // (BLOCK_C * BLOCK_W)
    if BLOCK_H <= 0:
        BLOCK_H = 1
    elif BLOCK_H > 4:
        BLOCK_H = 4

    num_c_blocks = (C + BLOCK_C - 1) // BLOCK_C
    num_h_blocks = (H + BLOCK_H - 1) // BLOCK_H
    num_w_blocks = (W + BLOCK_W - 1) // BLOCK_W
    num_spatial_blocks = num_h_blocks * num_w_blocks

    grid = (num_c_blocks, N, num_spatial_blocks)

    _add_relu_inplace_nchw_tile[grid](
        out, identity,
        C, N, H, W,
        out_stride_n, out_stride_c, out_stride_h, out_stride_w,
        id_stride_n, id_stride_c, id_stride_h, id_stride_w,
        num_w_blocks,
        BLOCK_C=BLOCK_C, BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W
    )
    return out


# Utility: fold conv + bn into conv weight & bias
def fold_conv_bn(conv: nn.Conv2d, bn: nn.BatchNorm2d):
    """
    Fold BatchNorm parameters into Conv2d weights and biases (using running stats).
    Returns (w_fold, b_fold) as tensors (not Parameters).
    """
    w = conv.weight
    device = w.device
    dtype = w.dtype

    if conv.bias is not None:
        conv_bias = conv.bias.to(device=device, dtype=dtype)
    else:
        conv_bias = torch.zeros(w.shape[0], device=device, dtype=dtype)

    running_mean = bn.running_mean.to(device=device, dtype=dtype)
    running_var = bn.running_var.to(device=device, dtype=dtype)
    eps = bn.eps

    if bn.affine:
        gamma = bn.weight.to(device=device, dtype=dtype)
        beta = bn.bias.to(device=device, dtype=dtype)
    else:
        gamma = torch.ones(w.shape[0], device=device, dtype=dtype)
        beta = torch.zeros(w.shape[0], device=device, dtype=dtype)

    denom = torch.sqrt(running_var + eps)
    scale = gamma / denom  # (out_channels,)

    w_fold = w * scale.reshape(-1, 1, 1, 1)
    b_fold = beta - running_mean * scale + conv_bias * scale

    return w_fold.detach(), b_fold.detach()


class BottleneckNew(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BottleneckNew, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        # caches for folded weights (filled on-demand or by parent)
        self._folded = False
        self._has_folded_downsample = False
        self._w1_f = None
        self._b1_f = None
        self._w2_f = None
        self._b2_f = None
        self._w3_f = None
        self._b3_f = None
        self._wds_f = None
        self._bds_f = None
        self._ds_stride = None
        self._ds_padding = None
        self._ds_dilation = None

    def _compute_folded_weights(self, device=None, dtype=None):
        if self._folded:
            return

        # compute folded weights on conv/bn pairs
        self._w1_f, self._b1_f = fold_conv_bn(self.conv1, self.bn1)
        self._w2_f, self._b2_f = fold_conv_bn(self.conv2, self.bn2)
        self._w3_f, self._b3_f = fold_conv_bn(self.conv3, self.bn3)

        # downsample folding if present
        self._has_folded_downsample = False
        if self.downsample is not None:
            ds_conv = None
            ds_bn = None
            if isinstance(self.downsample, nn.Sequential):
                for mod in self.downsample:
                    if isinstance(mod, nn.Conv2d) and ds_conv is None:
                        ds_conv = mod
                    elif isinstance(mod, nn.BatchNorm2d) and ds_bn is None:
                        ds_bn = mod
                if ds_conv is not None and ds_bn is not None:
                    self._wds_f, self._bds_f = fold_conv_bn(ds_conv, ds_bn)
                    self._ds_stride = ds_conv.stride
                    self._ds_padding = ds_conv.padding
                    self._ds_dilation = ds_conv.dilation
                    self._has_folded_downsample = True
            else:
                try:
                    convs = [m for m in self.downsample.modules() if isinstance(m, nn.Conv2d)]
                    bns = [m for m in self.downsample.modules() if isinstance(m, nn.BatchNorm2d)]
                    if len(convs) >= 1 and len(bns) >= 1:
                        ds_conv = convs[0]
                        ds_bn = bns[0]
                        self._wds_f, self._bds_f = fold_conv_bn(ds_conv, ds_bn)
                        self._ds_stride = ds_conv.stride
                        self._ds_padding = ds_conv.padding
                        self._ds_dilation = ds_conv.dilation
                        self._has_folded_downsample = True
                except Exception:
                    self._has_folded_downsample = False

        # move folded tensors to desired device/dtype to avoid future transfers
        if device is None or dtype is None:
            params = list(self.parameters())
            device = params[0].device if len(params) > 0 else torch.device("cuda")
            dtype = params[0].dtype if len(params) > 0 else torch.float32

        self._w1_f = self._w1_f.to(device=device, dtype=dtype)
        self._b1_f = self._b1_f.to(device=device, dtype=dtype)
        self._w2_f = self._w2_f.to(device=device, dtype=dtype)
        self._b2_f = self._b2_f.to(device=device, dtype=dtype)
        self._w3_f = self._w3_f.to(device=device, dtype=dtype)
        self._b3_f = self._b3_f.to(device=device, dtype=dtype)
        if self._has_folded_downsample:
            self._wds_f = self._wds_f.to(device=device, dtype=dtype)
            self._bds_f = self._bds_f.to(device=device, dtype=dtype)

        self._folded = True

    def train(self, mode: bool = True):
        res = super(BottleneckNew, self).train(mode)
        if mode:
            # invalidate folded cache when switching back to training
            self._folded = False
            self._has_folded_downsample = False
        return res

    def eval(self):
        res = super(BottleneckNew, self).eval()
        # mark folded stale so parent can precompute if desired
        self._folded = False
        self._has_folded_downsample = False
        return res

    def forward(self, x):
        identity = x

        # Fast folded path when all BNs are in eval mode
        can_fold = (not self.bn1.training) and (not self.bn2.training) and (not self.bn3.training)
        if can_fold:
            # compute folded weights lazily (or use parent's precompute)
            self._compute_folded_weights(device=x.device, dtype=x.dtype)

            # conv1 (folded conv+bn)
            out = F.conv2d(x, self._w1_f, self._b1_f, stride=self.conv1.stride, padding=self.conv1.padding, dilation=self.conv1.dilation)
            out = F.relu(out, inplace=True)

            # conv2 (folded)
            out = F.conv2d(out, self._w2_f, self._b2_f, stride=self.conv2.stride, padding=self.conv2.padding, dilation=self.conv2.dilation)
            out = F.relu(out, inplace=True)

            # conv3 (folded)
            out = F.conv2d(out, self._w3_f, self._b3_f, stride=self.conv3.stride, padding=self.conv3.padding, dilation=self.conv3.dilation)

            # identity: folded downsample if available else compute via module
            if self.downsample is not None:
                if self._has_folded_downsample:
                    identity = F.conv2d(identity, self._wds_f, self._bds_f, stride=self._ds_stride, padding=self._ds_padding, dilation=self._ds_dilation)
                else:
                    identity = self.downsample(identity)

            # in-place fused add + ReLU via Triton to minimize extra allocations and memory traffic
            out = triton_add_relu_inplace_nchw(out, identity)
            return out

        # Fallback (training or cannot fold): standard PyTorch path
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ModelNew(nn.Module):
    def __init__(self, layers, num_classes=1000):
        """
        Optimized ResNet-like model that uses:
         - conv+bn folding in eval to reduce kernels and memory traffic
         - Triton fused in-place add+ReLU kernel for the residual connection
         - Precomputation hooks to fold weights eagerly in eval to avoid first-pass overhead
        """
        super(ModelNew, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        block = BottleneckNew

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # stem folded caches
        self._stem_folded = False
        self._conv1_wf = None
        self._conv1_bf = None

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def _compute_stem_fold(self, device=None, dtype=None):
        if self._stem_folded:
            return
        if device is None or dtype is None:
            params = list(self.parameters())
            if len(params) > 0:
                device = params[0].device
                dtype = params[0].dtype
            else:
                device = torch.device("cuda")
                dtype = torch.float32
        self._conv1_wf, self._conv1_bf = fold_conv_bn(self.conv1, self.bn1)
        self._conv1_wf = self._conv1_wf.to(device=device, dtype=dtype)
        self._conv1_bf = self._conv1_bf.to(device=device, dtype=dtype)
        self._stem_folded = True

    def _precompute_all_folds(self, device=None, dtype=None):
        """
        Precompute folded weights for stem and all BottleneckNew modules.
        This avoids repeated CPU->GPU transfers on the first eval forward pass.
        """
        if device is None or dtype is None:
            params = list(self.parameters())
            if len(params) > 0:
                device = params[0].device
                dtype = params[0].dtype
            else:
                device = torch.device("cuda")
                dtype = torch.float32

        # stem
        if not self._stem_folded:
            self._compute_stem_fold(device=device, dtype=dtype)

        # iterate modules and compute folded weights for BottleneckNew
        for m in self.modules():
            if isinstance(m, BottleneckNew):
                if not m._folded:
                    m._compute_folded_weights(device=device, dtype=dtype)

    def train(self, mode: bool = True):
        res = super(ModelNew, self).train(mode)
        if mode:
            # Invalidate precomputed folded caches when switching to train
            self._stem_folded = False
            for m in self.modules():
                if isinstance(m, BottleneckNew):
                    m._folded = False
                    m._has_folded_downsample = False
        return res

    def eval(self):
        res = super(ModelNew, self).eval()
        # Precompute all folded weights eagerly to avoid first-forward overhead in eval.
        # Explicitly pass device/dtype derived from the model parameters so folded tensors are
        # created on the correct device/dtype and avoid blocking device-to-device casts later.
        try:
            params = list(self.parameters())
            if len(params) > 0:
                device = params[0].device
                dtype = params[0].dtype
            else:
                # Fallback to a sensible default if there are no parameters (shouldn't normally happen).
                device = torch.device("cuda")
                dtype = torch.float32
            self._precompute_all_folds(device=device, dtype=dtype)
        except Exception:
            # If for some reason precompute fails, it's safe to continue; folding will happen lazily.
            pass
        return res

    def forward(self, x):
        # Fast folded stem if possible
        if not self.bn1.training:
            # use precomputed folded stem if available, else compute lazily
            if not self._stem_folded:
                self._compute_stem_fold(device=x.device, dtype=x.dtype)
            # folded tensors are moved to correct device/dtype during _compute_stem_fold;
            # use them directly to avoid per-forward .to(...) allocations.
            if self._conv1_wf.device != x.device or self._conv1_wf.dtype != x.dtype:
                # One-time fallback move to keep behavior safe; this will update the stored folded tensors so
                # subsequent forwards do not pay the cast cost.
                self._conv1_wf = self._conv1_wf.to(device=x.device, dtype=x.dtype)
                self._conv1_bf = self._conv1_bf.to(device=x.device, dtype=x.dtype)
            x = F.conv2d(x, self._conv1_wf, self._conv1_bf, stride=self.conv1.stride, padding=self.conv1.padding, dilation=self.conv1.dilation)
            x = F.relu(x, inplace=True)
        else:
            x = self.conv1(x)
            x = self.bn1(x)
            x = F.relu(x, inplace=True)

        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x