# --------------------------------------------------------
# Swin MLP (ModelNew) - Enhanced Triton LayerNorm (wider autotune & BLOCK_N up to 16)
# - Aggressively autotuned, blocked, vectorized LayerNorm kernel handling multiple rows per program.
# - Additional autotune configs (including BLOCK_N=16) to better utilize Ampere A6000 SM resources.
# - Careful contiguous handling / cached buffers for non-affine path to reduce Python-side overhead.
# - Micro-optimized MLP paths (avoid Dropout allocation when drop==0, use F.gelu).
# - Replace AdaptiveAvgPool1d + transpose with direct mean over sequence dimension.
# Functionally equivalent to the original Model but optimized for throughput on A6000.
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import repeat
import collections.abc
from torch.utils.checkpoint import checkpoint
import triton
import triton.language as tl

# Expanded autotune configurations to explore a wider space for Ampere
AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK": 128, "BLOCK_N": 4},  num_warps=4, num_stages=2),
    triton.Config({"BLOCK": 128, "BLOCK_N": 8},  num_warps=4, num_stages=2),
    triton.Config({"BLOCK": 128, "BLOCK_N": 16}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK": 256, "BLOCK_N": 4},  num_warps=4, num_stages=2),
    triton.Config({"BLOCK": 256, "BLOCK_N": 8},  num_warps=8, num_stages=3),
    triton.Config({"BLOCK": 256, "BLOCK_N": 16}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK": 512, "BLOCK_N": 4},  num_warps=8, num_stages=3),
    triton.Config({"BLOCK": 512, "BLOCK_N": 8},  num_warps=8, num_stages=3),
    triton.Config({"BLOCK": 1024, "BLOCK_N": 4}, num_warps=8, num_stages=3),
]

@triton.autotune(configs=AUTOTUNE_CONFIGS, key=['N', 'C'])
@triton.jit
def _triton_layer_norm_kernel(
    x_ptr,           # pointer to input (N*C)
    out_ptr,         # pointer to output (N*C)
    gamma_ptr,       # pointer to weight (C)
    beta_ptr,        # pointer to bias (C)
    N,               # number of rows
    C,               # number of channels (columns)
    eps,             # epsilon
    BLOCK: tl.constexpr,    # number of columns per tile
    BLOCK_N: tl.constexpr,  # number of rows handled per program
    AFFINE: tl.constexpr,   # whether to apply affine (1) or skip (0)
    INPUT_FP16: tl.constexpr,  # whether input/output are fp16 storage (1) or fp32 (0)
):
    """
    Blocked, vectorized LayerNorm in Triton.
    Each program handles BLOCK_N rows, iterates columns in tiles of BLOCK.
    All computation in fp32 for accuracy. Supports skipping affine params and fp16 storage.
    """
    # row block index and row indices handled by this program
    row_block = tl.program_id(0)
    row_start = row_block * BLOCK_N
    rows = row_start + tl.arange(0, BLOCK_N)           # [BLOCK_N]
    row_mask = rows < N                                # [BLOCK_N]

    # column offsets for a tile
    offs = tl.arange(0, BLOCK)                         # [BLOCK]
    num_col_blocks = (C + BLOCK - 1) // BLOCK

    # accumulators per row (fp32)
    sum_val = tl.zeros((BLOCK_N,), dtype=tl.float32)
    sumsq_val = tl.zeros((BLOCK_N,), dtype=tl.float32)

    C_f = tl.cast(C, tl.float32)
    eps_f = tl.cast(eps, tl.float32)

    # First pass: accumulate sums and sumsq across column tiles
    for b in range(num_col_blocks):
        col_offs = b * BLOCK + offs                    # [BLOCK]
        col_mask = col_offs < C                        # [BLOCK]

        row_base = rows * C                            # [BLOCK_N]
        ptrs = x_ptr + row_base[:, None] + col_offs[None, :]  # [BLOCK_N, BLOCK]

        vals = tl.load(ptrs, mask=col_mask[None, :], other=0.0)  # shape [BLOCK_N, BLOCK]
        vals = tl.cast(vals, tl.float32)

        # accumulate per-row sums
        sum_val = sum_val + tl.sum(vals, 1)
        sumsq_val = sumsq_val + tl.sum(vals * vals, 1)

    # compute per-row mean and rstd
    mean = sum_val / C_f                               # [BLOCK_N]
    var = sumsq_val / C_f - mean * mean               # [BLOCK_N]
    rstd = tl.rsqrt(var + eps_f)                      # [BLOCK_N]

    # Second pass: normalize and optionally apply affine params and store
    for b in range(num_col_blocks):
        col_offs = b * BLOCK + offs
        col_mask = col_offs < C

        row_base = rows * C
        ptrs = x_ptr + row_base[:, None] + col_offs[None, :]

        vals = tl.load(ptrs, mask=col_mask[None, :], other=0.0)
        vals = tl.cast(vals, tl.float32)               # [BLOCK_N, BLOCK]

        normalized = (vals - mean[:, None]) * rstd[:, None]  # [BLOCK_N, BLOCK]

        if AFFINE:
            # load affine params (per-column) and broadcast across rows
            g = tl.cast(tl.load(gamma_ptr + col_offs, mask=col_mask, other=1.0), tl.float32)  # [BLOCK]
            be = tl.cast(tl.load(beta_ptr + col_offs, mask=col_mask, other=0.0), tl.float32)  # [BLOCK]
            out_tile = normalized * g[None, :] + be[None, :]
        else:
            out_tile = normalized

        # cast to storage dtype if requested
        if INPUT_FP16:
            out_store = tl.cast(out_tile, tl.float16)
        else:
            out_store = out_tile

        store_mask = col_mask[None, :] & row_mask[:, None]
        dst_ptrs = out_ptr + row_base[:, None] + col_offs[None, :]
        tl.store(dst_ptrs, out_store, mask=store_mask)


def triton_layer_norm(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float, affine: bool = True):
    """
    Wrapper for the Triton LayerNorm kernel with CPU fallback.
    Supports input tensors on CUDA (fp32 or fp16 storage).
    """
    if not x.is_cuda:
        if affine:
            return F.layer_norm(x, (x.shape[-1],), weight, bias, eps)
        else:
            return F.layer_norm(x, (x.shape[-1],), None, None, eps)

    orig_shape = x.shape
    C = orig_shape[-1]
    N = int(x.numel() // C)

    x_contig = x.contiguous().view(N, C)
    # allocate output with same dtype and device as input
    out = torch.empty_like(x_contig)

    device = x_contig.device
    dtype = x_contig.dtype

    # Prepare weight/bias tensors (if affine)
    if weight is None:
        # create appropriate dtype tensors for the kernel
        weight = torch.ones(C, device=device, dtype=dtype)
    if bias is None:
        bias = torch.zeros(C, device=device, dtype=dtype)

    # Ensure contiguous device buffers for efficient tl.load
    w = weight.contiguous()
    b = bias.contiguous()

    grid = lambda meta: ((N + meta['BLOCK_N'] - 1) // meta['BLOCK_N'],)

    # Pass AFFINE and INPUT_FP16 as constexpr specialization flags
    _triton_layer_norm_kernel[grid](
        x_contig, out, w, b, N, C, float(eps),
        AFFINE=int(bool(affine)),
        INPUT_FP16=int(dtype == torch.float16)
    )
    return out.view(*orig_shape)


class TritonLayerNorm(nn.Module):
    """
    Drop-in replacement for nn.LayerNorm using the optimized Triton kernel.
    - Caches ones/zeros device buffers for non-affine case to avoid allocations.
    - Ensures parameters are contiguous before kernel invocation.
    """
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super().__init__()
        if isinstance(normalized_shape, (list, tuple)):
            assert len(normalized_shape) == 1, "TritonLayerNorm only supports last-dim normalization"
            normalized_shape = normalized_shape[0]
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(normalized_shape, dtype=torch.float32))
            self.bias = nn.Parameter(torch.zeros(normalized_shape, dtype=torch.float32))
            # caches not used in affine path but keep attributes for uniformity
            self.register_buffer('_ones_cache', None)
            self.register_buffer('_zeros_cache', None)
        else:
            # No trainable params; keep caches for ones/zeros on device
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
            self.register_buffer('_ones_cache', None)
            self.register_buffer('_zeros_cache', None)

    def forward(self, x: torch.Tensor):
        if not x.is_cuda:
            if self.elementwise_affine:
                return F.layer_norm(x, (self.normalized_shape,), self.weight, self.bias, self.eps)
            else:
                return F.layer_norm(x, (self.normalized_shape,), None, None, self.eps)

        # Validate shape
        assert x.shape[-1] == self.normalized_shape, f"Expected last dim {self.normalized_shape}, got {x.shape[-1]}"

        # If affine, use params (cast to input dtype if needed)
        if self.elementwise_affine:
            # If input is fp16, cast params to fp16 for storage loads; kernel will cast to fp32 internally.
            in_dtype = x.dtype
            if self.weight.dtype != in_dtype:
                w = self.weight.to(in_dtype).contiguous()
                b = self.bias.to(in_dtype).contiguous()
            else:
                w = self.weight.contiguous()
                b = self.bias.contiguous()
            return triton_layer_norm(x, w, b, self.eps, affine=True)
        else:
            # non-affine: use cached ones/zeros buffers matching input dtype and device
            device = x.device
            in_dtype = x.dtype
            need_new = (self._ones_cache is None) or (self._ones_cache.device != device) or (self._ones_cache.numel() != self.normalized_shape) or (self._ones_cache.dtype != in_dtype)
            if need_new:
                self._ones_cache = torch.ones(self.normalized_shape, device=device, dtype=in_dtype)
                self._zeros_cache = torch.zeros(self.normalized_shape, device=device, dtype=in_dtype)
            return triton_layer_norm(x, self._ones_cache, self._zeros_cache, self.eps, affine=False)


# Helper utilities (unchanged)
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))
    return parse
to_2tuple = _ntuple(2)


class Mlp(nn.Module):
    """
    Lightweight MLP optimized for throughput:
      - fc1 -> GELU -> optional dropout -> fc2 -> optional dropout
      - avoids Dropout allocation when drop == 0
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, out_features)
        # Lightweight activation instance
        self.act = act_layer()
        self.drop_prob = float(drop)
        self.use_dropout = self.drop_prob > 0.0
        if self.use_dropout:
            self.drop = nn.Dropout(self.drop_prob)
        else:
            self.drop = None

    def forward(self, x):
        x = self.fc1(x)
        # use functional gelu could be slightly faster in some builds; keep module-based for compatibility
        x = F.gelu(x)
        if self.drop is not None:
            x = self.drop(x)
        x = self.fc2(x)
        if self.drop is not None:
            x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class SwinMLPBlock(nn.Module):
    r""" Swin MLP Block. """
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.padding = [self.window_size - self.shift_size, self.shift_size,
                        self.window_size - self.shift_size, self.shift_size]  # P_l,P_r,P_t,P_b

        self.norm1 = norm_layer(dim)
        # use group convolution to implement multi-head MLP
        self.spatial_mlp = nn.Conv1d(self.num_heads * self.window_size ** 2,
                                     self.num_heads * self.window_size ** 2,
                                     kernel_size=1,
                                     groups=self.num_heads)

        self.drop_path = nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # shift (with padding) if necessary
        if self.shift_size > 0:
            P_l, P_r, P_t, P_b = self.padding
            # pad ordering: (left, right, top, bottom) with channels last -> pad dims accordingly
            shifted_x = F.pad(x, [0, 0, P_l, P_r, P_t, P_b], "constant", 0)
        else:
            shifted_x = x
        _, _H, _W, _ = shifted_x.shape

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, ws, ws, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, ws*ws, C

        # Window/Shifted-Window Spatial MLP (grouped 1x1 conv)
        x_windows_heads = x_windows.view(-1, self.window_size * self.window_size, self.num_heads, C // self.num_heads)
        x_windows_heads = x_windows_heads.transpose(1, 2)  # nW*B, nH, ws*ws, C//nH
        x_windows_heads = x_windows_heads.reshape(-1, self.num_heads * self.window_size * self.window_size,
                                                  C // self.num_heads)
        spatial_mlp_windows = self.spatial_mlp(x_windows_heads)
        spatial_mlp_windows = spatial_mlp_windows.view(-1, self.num_heads, self.window_size * self.window_size,
                                                       C // self.num_heads).transpose(1, 2)
        spatial_mlp_windows = spatial_mlp_windows.reshape(-1, self.window_size * self.window_size, C)

        # merge windows
        spatial_mlp_windows = spatial_mlp_windows.reshape(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(spatial_mlp_windows, self.window_size, _H, _W)  # B H' W' C

        # reverse shift / crop
        if self.shift_size > 0:
            P_l, P_r, P_t, P_b = self.padding
            x = shifted_x[:, P_t:-P_b, P_l:-P_r, :].contiguous()
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN with residuals
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class PatchMerging(nn.Module):
    r""" Patch Merging Layer. """
    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x


class BasicLayer(nn.Module):
    """ A basic Swin MLP layer for one stage. """
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., drop=0., drop_path=0.,
                 norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinMLPBlock(dim=dim, input_resolution=input_resolution,
                         num_heads=num_heads, window_size=window_size,
                         shift_size=0 if (i % 2 == 0) else window_size // 2,
                         mlp_ratio=mlp_ratio,
                         drop=drop,
                         drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                         norm_layer=norm_layer)
            for i in range(depth)
        ])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint(blk, x)
            else:
                x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x


class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding """
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        # enforce expected input size
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops


class ModelNew(nn.Module):
    """
    Swin MLP optimized with an enhanced Triton LayerNorm and micro-optimizations.
    - Replaces default nn.LayerNorm with TritonLayerNorm when norm_layer is nn.LayerNorm.
    - Avoids Dropout allocation when drop==0.
    - Uses mean pooling over sequence dimension instead of AdaptiveAvgPool1d.
    """
    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, patch_norm=True,
                 use_checkpoint=False, **kwargs):
        super().__init__()

        # If the user requested the default nn.LayerNorm, replace with TritonLayerNorm for speed.
        effective_norm = TritonLayerNorm if norm_layer is nn.LayerNorm else norm_layer

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=effective_norm if self.patch_norm else None)
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # avoid allocating dropout if drop_rate == 0
        self.pos_drop = None
        self.pos_drop_prob = float(drop_rate)
        if self.pos_drop_prob > 0.0:
            self.pos_drop = nn.Dropout(p=self.pos_drop_prob)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               drop=drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=effective_norm,
                               downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                               use_checkpoint=use_checkpoint)
            self.layers.append(layer)

        self.norm = effective_norm(self.num_features)
        # avoid AdaptiveAvgPool overhead; use a simple Linear head after mean pooling
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        x = self.patch_embed(x)
        if self.pos_drop is not None:
            x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)  # B L C
        # global average pool over sequence dimension L -> (B, C)
        x = x.mean(dim=1)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


# default input sizes (kept for compatibility)
batch_size = 10
image_size = 224

def get_inputs():
    return [torch.rand(batch_size, 3, image_size, image_size)]

def get_init_inputs():
    return []