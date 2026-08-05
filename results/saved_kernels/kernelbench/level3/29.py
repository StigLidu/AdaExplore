# --------------------------------------------------------
# Swin Transformer (optimized with Triton kernels)
# This file replaces window partition/reverse and elementwise adds
# with Triton kernels to reduce memory traffic and Python overhead.
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import repeat
import collections.abc

# Triton imports for custom kernels
import triton
import triton.language as tl

# NOTE: Removed the small Triton add kernel because launching a separate Triton kernel
# for simple elementwise residual additions adds overhead and is typically slower than
# PyTorch's optimized CUDA add. Use torch.add (which dispatches to optimized CUDA kernels).
def triton_add(x: torch.Tensor, y: torch.Tensor):
    """
    Use PyTorch's addition for residuals. This avoids Triton kernel launch overhead
    for very low-arithmetic-intensity elementwise adds.
    Behaves like x + y (out-of-place) and preserves original code's semantics.
    """
    assert x.shape == y.shape, "Shapes must match for elementwise add."
    if not x.is_cuda or not y.is_cuda:
        return x + y
    # Ensure efficient CUDA path (make contiguous if needed).
    if not x.is_contiguous():
        x = x.contiguous()
    if not y.is_contiguous():
        y = y.contiguous()
    # Use PyTorch's add (efficient CUDA kernel). Return a new tensor (matches original semantics).
    return torch.add(x, y)


# Triton kernels: fused partition -> head-major layout and inverse (head-major -> image layout).
# These kernels map the full flattened output index space to the corresponding input index in the image,
# but write/read in the head-major layout consumed/produced by the group Conv1d, avoiding large transposes.
PART_BLOCKS = [
    triton.Config({"BLOCK_SIZE": 8192}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_SIZE": 16384}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_SIZE": 32768}, num_warps=8, num_stages=3),
]

@triton.autotune(configs=PART_BLOCKS, key=['n_elements'])
@triton.jit
def _triton_window_partition_headmajor_kernel(
    inp_ptr, out_ptr,
    B, H, W, C,
    ws, nW_H, nW_W, n_windows_per_img,
    num_heads, C_head,
    n_elements,
    BLOCK_SIZE: tl.constexpr
):
    """
    Writes output in layout: (B * n_windows_per_img, num_heads * ws * ws, C_head)
    Flattened output index layout: (((b * n_windows_per_img + win) * (num_heads*ws*ws) + hp) * C_head) + c_inner
    where hp = head * (ws*ws) + p and p in [0, ws*ws)
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    idx = offs

    # compute location from flattened output index
    c_inner = idx % C_head
    tmp0 = idx // C_head

    hp = tmp0 % (num_heads * ws * ws)          # combined head and position
    tmp1 = tmp0 // (num_heads * ws * ws)

    win = tmp1 % n_windows_per_img
    b = tmp1 // n_windows_per_img

    head = hp // (ws * ws)
    p = hp % (ws * ws)

    h_in_win = p // ws
    w_in_win = p % ws

    win_row = win // nW_W
    win_col = win % nW_W

    in_h = win_row * ws + h_in_win
    in_w = win_col * ws + w_in_win

    # map to original input channel index
    c = head * C_head + c_inner

    # compute input flattened index ((b*H + in_h)*W + in_w)*C + c
    input_idx = ((b * H + in_h) * W + in_w) * C + c

    vals = tl.load(inp_ptr + input_idx, mask=mask, other=0.0)
    tl.store(out_ptr + idx, vals, mask=mask)


@triton.autotune(configs=PART_BLOCKS, key=['n_elements'])
@triton.jit
def _triton_window_reverse_headmajor_kernel(
    inp_ptr, out_ptr,
    B, H, W, C,
    ws, nW_H, nW_W, n_windows_per_img,
    num_heads, C_head,
    n_elements,
    BLOCK_SIZE: tl.constexpr
):
    """
    Reads input in head-major layout (the output of the Conv1d):
      (B * n_windows_per_img, num_heads * ws * ws, C_head)
    and writes back to image layout (B, H, W, C).
    The flattened input index matches the same scheme as partition kernel.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    idx = offs

    # flattened input idx corresponds to head-major layout
    c_inner = idx % C_head
    tmp0 = idx // C_head

    hp = tmp0 % (num_heads * ws * ws)
    tmp1 = tmp0 // (num_heads * ws * ws)

    win = tmp1 % n_windows_per_img
    b = tmp1 // n_windows_per_img

    head = hp // (ws * ws)
    p = hp % (ws * ws)

    h_in_win = p // ws
    w_in_win = p % ws

    win_row = win // nW_W
    win_col = win % nW_W

    in_h = win_row * ws + h_in_win
    in_w = win_col * ws + w_in_win

    c = head * C_head + c_inner

    # compute output image flattened index ((b*H + in_h)*W + in_w)*C + c
    out_idx = ((b * H + in_h) * W + in_w) * C + c

    vals = tl.load(inp_ptr + idx, mask=mask, other=0.0)
    tl.store(out_ptr + out_idx, vals, mask=mask)


def window_partition(x, window_size, num_heads):
    """
    Triton-accelerated fused window partition that writes the head-major layout directly:
      returns tensor of shape (B * n_windows_per_img, num_heads * ws * ws, C_head)
    where C_head = C // num_heads.
    """
    if not x.is_cuda:
        # fallback to original implementation, then convert to head-major layout
        B, H, W, C = x.shape
        ws = window_size
        x_view = x.view(B, H // ws, ws, W // ws, ws, C)
        windows = x_view.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, ws, ws, C)  # nW*B, ws, ws, C
        # convert to head-major layout
        nW_B = windows.shape[0]
        C_head = C // num_heads
        windows_flat = windows.view(nW_B, ws * ws, num_heads, C_head).transpose(1, 2).reshape(nW_B, num_heads * ws * ws, C_head)
        return windows_flat

    B, H, W, C = x.shape
    ws = window_size
    assert H % ws == 0 and W % ws == 0, "H and W must be divisible by window_size for this partition kernel."
    assert C % num_heads == 0, "C must be divisible by num_heads."

    nW_H = H // ws
    nW_W = W // ws
    n_windows_per_img = nW_H * nW_W

    n_elements = B * H * W * C
    C_head = C // num_heads

    inp = x.contiguous()
    out_shape = (B * n_windows_per_img, num_heads * ws * ws, C_head)
    out = torch.empty(out_shape, dtype=inp.dtype, device=inp.device)

    inp_flat = inp.view(-1)
    out_flat = out.view(-1)

    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    _triton_window_partition_headmajor_kernel[grid](
        inp_flat, out_flat,
        B, H, W, C,
        ws, nW_H, nW_W, n_windows_per_img,
        num_heads, C_head,
        n_elements
    )
    return out


def window_reverse(windows_headmajor, window_size, H, W, num_heads):
    """
    Triton-accelerated fused window reverse that consumes head-major layout:
      windows_headmajor: (B * n_windows_per_img, num_heads * ws * ws, C_head)
    and returns x in image layout (B, H, W, C).
    """
    if not windows_headmajor.is_cuda:
        # fallback: convert from head-major back to windows and reverse as Python does
        nW_B = windows_headmajor.shape[0]
        _, L, C_head = windows_headmajor.shape
        ws = window_size
        num_heads = num_heads
        C = C_head * num_heads
        windows = windows_headmajor.view(nW_B, num_heads, ws * ws, C_head).transpose(1, 2).reshape(nW_B, ws, ws, C)
        B = int(nW_B / (H * W / (ws * ws)))
        x = windows.view(B, H // ws, W // ws, ws, ws, C)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, C)
        return x

    ws = window_size
    total_windows = windows_headmajor.shape[0]
    C_head = windows_headmajor.shape[2]
    C = C_head * num_heads
    nW_H = H // ws
    nW_W = W // ws
    n_windows_per_img = nW_H * nW_W
    B = total_windows // n_windows_per_img

    n_elements = B * H * W * C

    inp = windows_headmajor.contiguous()
    out = torch.empty((B, H, W, C), dtype=inp.dtype, device=inp.device)

    inp_flat = inp.view(-1)
    out_flat = out.view(-1)

    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    _triton_window_reverse_headmajor_kernel[grid](
        inp_flat, out_flat,
        B, H, W, C,
        ws, nW_H, nW_W, n_windows_per_img,
        num_heads, C_head,
        n_elements
    )
    return out


# The rest of the model is largely unchanged but uses the Triton-accelerated helpers above.

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SwinMLPBlock(nn.Module):
    r""" Swin MLP Block.
    """
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

        # shift
        if self.shift_size > 0:
            P_l, P_r, P_t, P_b = self.padding
            shifted_x = F.pad(x, [0, 0, P_l, P_r, P_t, P_b], "constant", 0)
        else:
            shifted_x = x
        _, _H, _W, _ = shifted_x.shape

        # partition windows (fused Triton-accelerated) into head-major layout
        # produce shape: (nW*B, num_heads * ws * ws, C_head)
        x_windows_heads = window_partition(shifted_x, self.window_size, self.num_heads)

        # Window/Shifted-Window Spatial MLP: feed head-major layout directly to Conv1d
        spatial_mlp_windows_heads = self.spatial_mlp(x_windows_heads)  # shape preserved: (nW*B, num_heads*ws*ws, C_head)

        # merge windows: consume head-major output and write back into image layout in one Triton kernel
        shifted_x = window_reverse(spatial_mlp_windows_heads, self.window_size, _H, _W, self.num_heads)  # B H' W' C

        # reverse shift
        if self.shift_size > 0:
            P_l, P_r, P_t, P_b = self.padding
            x = shifted_x[:, P_t:-P_b, P_l:-P_r, :].contiguous()
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN with Triton-accelerated elementwise adds for residuals
        res1 = self.drop_path(x)
        if shortcut.is_cuda and res1.is_cuda:
            x = triton_add(shortcut.contiguous(), res1.contiguous())
        else:
            x = shortcut + res1

        mlp_out = self.mlp(self.norm2(x))
        res2 = self.drop_path(mlp_out)
        if x.is_cuda and res2.is_cuda:
            x = triton_add(x.contiguous(), res2.contiguous())
        else:
            x = x + res2

        return x


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.
    """
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
    """ A basic Swin MLP layer for one stage.
    """
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
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                import torch.utils.checkpoint as checkpoint
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))
    return parse
to_2tuple = _ntuple(2)


class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding
    """
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
        # FIXME look at relaxing size constraints
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
    r""" Swin MLP (optimized variant using Triton kernels)
    """
    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, patch_norm=True,
                 use_checkpoint=False, **kwargs):
        super().__init__()

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

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
                               norm_layer=norm_layer,
                               downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                               use_checkpoint=use_checkpoint)
            self.layers.append(layer)

        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        x = self.patch_embed(x)
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)  # B L C
        x = self.avgpool(x.transpose(1, 2))  # B C 1
        x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


# Input helpers retained at bottom
batch_size = 10
image_size = 224

def get_inputs():
    return [torch.rand(batch_size, 3, image_size, image_size)]

def get_init_inputs():
    return []