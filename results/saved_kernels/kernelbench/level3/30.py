# Optimized Swin Transformer V2 with improved Triton-accelerated window partitioning/reverse
# - Builds on a flattened-window Triton implementation.
# - Tuned block sizes and tile sizes for Ampere (A6000) to increase memory throughput.
# - Minimizes Python-side copies and ensures kernels vectorize over channels.
# - Exports the optimized model as ModelNew.
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import collections
from itertools import repeat

# Triton imports for custom kernels
import triton
import triton.language as tl

def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))
    return parse
to_2tuple = _ntuple(2)

# -----------------------
# Triton kernels (tuned)
# -----------------------

# Larger channel block for Ampere, and larger token tile to amortize index math
BLOCK_C_DEFAULT = 256  # vectorized channel block (constexpr)
TILE_T_DEFAULT = 16    # number of tokens handled per program (constexpr)

@triton.jit
def _partition_kernel_flat(x_ptr, out_ptr,
                           B, H, W, C, ws, num_win_per_img, window_area, total_wins,
                           BLOCK_C: tl.constexpr, TILE_T: tl.constexpr):
    """
    Partition (B, H, W, C) -> (total_wins, window_area, C)
    Each program handles TILE_T tokens and BLOCK_C contiguous channels.
    program_id(0): token-tile index across all tokens
    program_id(1): channel-block index
    """
    token_tile = tl.program_id(0)
    c_block = tl.program_id(1)

    n_tokens = total_wins * window_area

    # channel offsets for this block (constexpr)
    c_offsets = c_block * BLOCK_C + tl.arange(0, BLOCK_C)
    mask_c = c_offsets < C

    # Process TILE_T tokens per program (unrolled)
    for tt in range(TILE_T):
        token_global = token_tile * TILE_T + tt
        token_mask = token_global < n_tokens

        # decode win_idx and token_idx
        win_idx = token_global // window_area
        token_idx = token_global - win_idx * window_area

        # token_idx -> local window coords
        i = token_idx // ws
        j = token_idx - i * ws

        img_idx = win_idx // num_win_per_img
        local_win = win_idx - img_idx * num_win_per_img

        num_win_w = W // ws
        tile_row = local_win // num_win_w
        tile_col = local_win - tile_row * num_win_w

        in_h = tile_row * ws + i
        in_w = tile_col * ws + j

        # source base index in flattened (B,H,W,C) memory
        src_base = ((img_idx * H + in_h) * W + in_w) * C
        src_ptrs = x_ptr + src_base + c_offsets

        # load channel block for this token
        vals = tl.load(src_ptrs, mask=(token_mask & mask_c), other=0.0)

        # destination in flattened windows layout: (win_idx, token_idx, C)
        dst_base = (win_idx * window_area + token_idx) * C
        dst_ptrs = out_ptr + dst_base + c_offsets

        tl.store(dst_ptrs, vals, mask=(token_mask & mask_c))


@triton.jit
def _reverse_kernel_flat(w_ptr, out_ptr,
                         B, H, W, C, ws, num_win_per_img, window_area, total_wins,
                         BLOCK_C: tl.constexpr, TILE_T: tl.constexpr):
    """
    Reverse (total_wins, window_area, C) -> (B, H, W, C)
    Each program handles TILE_T tokens and BLOCK_C contiguous channels.
    program_id(0): token-tile index across all tokens
    program_id(1): channel-block index
    """
    token_tile = tl.program_id(0)
    c_block = tl.program_id(1)

    n_tokens = total_wins * window_area

    c_offsets = c_block * BLOCK_C + tl.arange(0, BLOCK_C)
    mask_c = c_offsets < C

    for tt in range(TILE_T):
        token_global = token_tile * TILE_T + tt
        token_mask = token_global < n_tokens

        win_idx = token_global // window_area
        token_idx = token_global - win_idx * window_area

        i = token_idx // ws
        j = token_idx - i * ws

        img_idx = win_idx // num_win_per_img
        local_win = win_idx - img_idx * num_win_per_img

        num_win_w = W // ws
        tile_row = local_win // num_win_w
        tile_col = local_win - tile_row * num_win_w

        in_h = tile_row * ws + i
        in_w = tile_col * ws + j

        # source in windows flattened layout
        src_base = (win_idx * window_area + token_idx) * C
        src_ptrs = w_ptr + src_base + c_offsets
        vals = tl.load(src_ptrs, mask=(token_mask & mask_c), other=0.0)

        # destination in dense layout
        dst_base = ((img_idx * H + in_h) * W + in_w) * C
        dst_ptrs = out_ptr + dst_base + c_offsets
        tl.store(dst_ptrs, vals, mask=(token_mask & mask_c))


def triton_window_partition_flat(x: torch.Tensor, window_size: int, block_c: int = BLOCK_C_DEFAULT, tile_t: int = TILE_T_DEFAULT):
    """
    Partition windows on CUDA using Triton and produce flattened windows:
      out shape = (total_wins, window_area, C)
    """
    assert x.is_cuda, "triton_window_partition_flat requires CUDA tensor"
    assert x.ndim == 4, "input must be 4D (B, H, W, C)"
    B, H, W, C = x.shape
    ws = int(window_size)
    assert H % ws == 0 and W % ws == 0, "H and W must be divisible by window_size"

    # Avoid copy when already contiguous
    x_contig = x if x.is_contiguous() else x.contiguous()
    num_win_per_img = (H // ws) * (W // ws)
    total_wins = num_win_per_img * B
    window_area = ws * ws

    out = torch.empty((total_wins, window_area, C), dtype=x.dtype, device=x.device)

    n_tokens = total_wins * window_area
    n_token_tiles = (n_tokens + tile_t - 1) // tile_t
    n_c_blocks = (C + block_c - 1) // block_c
    grid = (n_token_tiles, n_c_blocks)

    _partition_kernel_flat[grid](
        x_contig, out,
        B, H, W, C, ws, num_win_per_img, window_area, total_wins,
        BLOCK_C=block_c, TILE_T=tile_t
    )
    return out


def triton_window_reverse_flat(windows: torch.Tensor, window_size: int, H: int, W: int, block_c: int = BLOCK_C_DEFAULT, tile_t: int = TILE_T_DEFAULT):
    """
    Reverse partition on CUDA using Triton from flattened windows:
      windows shape = (total_wins, window_area, C)
    returns: (B, H, W, C)
    """
    assert windows.is_cuda, "triton_window_reverse_flat requires CUDA tensor"
    assert windows.ndim == 3, "windows must be 3D (total_wins, window_area, C)"
    total_wins, window_area, C = windows.shape
    ws = int(window_size)
    assert window_area == ws * ws
    num_win_per_img = (H // ws) * (W // ws)
    assert num_win_per_img > 0
    assert total_wins % num_win_per_img == 0
    B = total_wins // num_win_per_img

    out = torch.empty((B, H, W, C), dtype=windows.dtype, device=windows.device)

    n_tokens = total_wins * window_area
    n_token_tiles = (n_tokens + tile_t - 1) // tile_t
    n_c_blocks = (C + block_c - 1) // block_c
    grid = (n_token_tiles, n_c_blocks)

    _reverse_kernel_flat[grid](
        windows, out,
        B, H, W, C, ws, num_win_per_img, window_area, total_wins,
        BLOCK_C=block_c, TILE_T=tile_t
    )
    return out

# -----------------------
# Model code (optimized usage of Triton partition/reverse)
# -----------------------

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


def window_partition(x, window_size):
    """
    CPU fallback: (B, H, W, C) -> (num_windows*B, window_area, C)
    """
    B, H, W, C = x.shape
    ws = window_size
    x = x.view(B, H // ws, ws, W // ws, ws, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, ws * ws, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    CPU fallback reverse expecting flattened windows: (total_wins, window_area, C)
    """
    total_wins, window_area, C = windows.shape
    ws = window_size
    B = int(total_wins / ((H // ws) * (W // ws)))
    x = windows.view(B, H // ws, W // ws, ws, ws, C)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, C)
    return x


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    Logic is unchanged; accepts input (num_windows*B, N, C)
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.,
                 pretrained_window_size=[0, 0]):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.pretrained_window_size = pretrained_window_size
        self.num_heads = num_heads

        self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))), requires_grad=True)

        # mlp to generate continuous relative position bias
        # keep as module to be trainable
        self.cpb_mlp = nn.Sequential(nn.Linear(2, 512, bias=True),
                                     nn.ReLU(inplace=True),
                                     nn.Linear(512, num_heads, bias=False))

        # get relative_coords_table (small tensor)
        relative_coords_h = torch.arange(-(self.window_size[0] - 1), self.window_size[0], dtype=torch.float32)
        relative_coords_w = torch.arange(-(self.window_size[1] - 1), self.window_size[1], dtype=torch.float32)
        relative_coords_table = torch.stack(
            torch.meshgrid([relative_coords_h,
                            relative_coords_w], indexing='ij')).permute(1, 2, 0).contiguous().unsqueeze(0)  # 1, 2*Wh-1, 2*Ww-1, 2
        if pretrained_window_size[0] > 0:
            relative_coords_table[:, :, :, 0] /= (pretrained_window_size[0] - 1)
            relative_coords_table[:, :, :, 1] /= (pretrained_window_size[1] - 1)
        else:
            relative_coords_table[:, :, :, 0] /= (self.window_size[0] - 1)
            relative_coords_table[:, :, :, 1] /= (self.window_size[1] - 1)
        relative_coords_table *= 8  # normalize to -8, 8
        relative_coords_table = torch.sign(relative_coords_table) * torch.log2(
            torch.abs(relative_coords_table) + 1.0) / np.log2(8)

        self.register_buffer("relative_coords_table", relative_coords_table)

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(dim))
            self.v_bias = nn.Parameter(torch.zeros(dim))
        else:
            self.q_bias = None
            self.v_bias = None
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        x: (num_windows*B, N, C)
        """
        B_, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # cosine attention
        attn = (F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1))
        # clamp & exp as in original
        logit_scale = torch.clamp(self.logit_scale.to(x.device),
                                  max=torch.log(torch.tensor(1. / 0.01, device=x.device))).exp()
        attn = attn * logit_scale

        relative_position_bias_table = self.cpb_mlp(self.relative_coords_table).view(-1, self.num_heads)
        relative_position_bias = relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        relative_position_bias = 16 * torch.sigmoid(relative_position_bias)
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block using Triton-optimized flattened-window partition/reverse.
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, pretrained_window_size=0):
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

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop,
            pretrained_window_size=to_2tuple(pretrained_window_size))

        self.drop_path = nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            # Use the same flattened layout for mask windows
            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_area, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        if shifted_x.is_cuda:
            # use Triton flattened partition: (total_wins, window_area, C)
            x_windows = triton_window_partition_flat(shifted_x, self.window_size,
                                                     block_c=BLOCK_C_DEFAULT, tile_t=TILE_T_DEFAULT)
        else:
            x_windows = window_partition(shifted_x, self.window_size)

        # Ensure layout matches WindowAttention: (num_windows*B, N, C)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C

        # merge windows (flattened windows layout)
        attn_windows = attn_windows.view(-1, self.window_size * self.window_size, C)

        # reverse partition
        if attn_windows.is_cuda:
            # ensure contiguous to avoid hidden copies inside Triton wrapper
            attn_windows = attn_windows if attn_windows.is_contiguous() else attn_windows.contiguous()
            shifted_x = triton_window_reverse_flat(attn_windows, self.window_size, H, W,
                                                   block_c=BLOCK_C_DEFAULT, tile_t=TILE_T_DEFAULT)
        else:
            shifted_x = window_reverse(attn_windows, self.window_size, H, W)

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(self.norm1(x))

        # FFN
        x = x + self.drop_path(self.norm2(self.mlp(x)))

        return x


class PatchMerging(nn.Module):
    r""" Patch Merging Layer. """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(2 * dim)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1)
        x = x.view(B, -1, 4 * C)

        x = self.reduction(x)
        x = self.norm(x)

        return x


class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage. """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                 pretrained_window_size=0):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer,
                                 pretrained_window_size=pretrained_window_size)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
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
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x


class Model(nn.Module):
    r""" Swin Transformer """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., qkv_bias=True,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, patch_norm=True,
                 use_checkpoint=False, pretrained_window_sizes=[0, 0, 0, 0], **kwargs):
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
                               qkv_bias=qkv_bias,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                               use_checkpoint=use_checkpoint,
                               pretrained_window_size=pretrained_window_sizes[i_layer])
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

# Export optimized model under ModelNew name
ModelNew = Model

# Input spec (kept for compatibility)
batch_size = 10
image_size = 224

def get_inputs():
    return [torch.rand(batch_size, 3, image_size, image_size)]

def get_init_inputs():
    return []