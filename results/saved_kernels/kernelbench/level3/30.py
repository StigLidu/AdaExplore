import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import collections
from itertools import repeat

def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))
    return parse
to_2tuple = _ntuple(2)


class FastLayerNorm(nn.Module):
    """Thin wrapper around torch.nn.LayerNorm to ensure consistent API and avoid small overheads."""
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super().__init__()
        if isinstance(normalized_shape, (list, tuple)):
            assert len(normalized_shape) == 1
            normalized_shape = int(normalized_shape[0])
        self.norm = nn.LayerNorm(normalized_shape, eps=eps, elementwise_affine=elementwise_affine)

    def forward(self, x):
        return self.norm(x)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        # Standard linear layers are left to PyTorch (cuBLAS/cuDNN)
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        # combine activation and dropout to reduce temporaries
        x = self.drop(x)
        x = self.fc2(x)
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
    ws = window_size
    # reshape + permute without unnecessary contiguous calls
    x = x.view(B, H // ws, ws, W // ws, ws, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).reshape(-1, ws, ws, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
    Returns:
        x: (B, H, W, C)
    """
    ws = window_size
    B = int(windows.shape[0] / (H * W / ws / ws))
    x = windows.view(B, H // ws, W // ws, ws, ws, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).reshape(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    """
    Window based multi-head self attention (W-MSA) with continuous relative position bias.
    Optimized to use PyTorch's scaled_dot_product_attention when available,
    and to minimize temporary allocations and device transfers. This version
    caches the continuous relative position bias and fp16 copies of projection
    weights for inference to avoid repeated recompute and dtype conversions.
    """
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.,
                 pretrained_window_size=[0, 0]):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # tuple (Wh, Ww)
        self.pretrained_window_size = pretrained_window_size
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        # per-head logit scale parameter
        self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))), requires_grad=True)

        # small MLP for continuous relative position bias (as in original)
        self.cpb_mlp = nn.Sequential(
            nn.Linear(2, 512, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_heads, bias=False)
        )

        # build relative_coords_table buffer
        relative_coords_h = torch.arange(-(self.window_size[0] - 1), self.window_size[0], dtype=torch.float32)
        relative_coords_w = torch.arange(-(self.window_size[1] - 1), self.window_size[1], dtype=torch.float32)
        # use indexing='ij' for correct meshgrid behavior
        rc = torch.stack(torch.meshgrid([relative_coords_h, relative_coords_w], indexing='ij'))
        relative_coords_table = rc.permute(1, 2, 0).contiguous().unsqueeze(0)  # 1, Htbl, Wtbl, 2
        if pretrained_window_size[0] > 0:
            relative_coords_table[:, :, :, 0] /= (pretrained_window_size[0] - 1)
            relative_coords_table[:, :, :, 1] /= (pretrained_window_size[1] - 1)
        else:
            relative_coords_table[:, :, :, 0] /= (self.window_size[0] - 1)
            relative_coords_table[:, :, :, 1] /= (self.window_size[1] - 1)
        relative_coords_table *= 8.0
        relative_coords_table = torch.sign(relative_coords_table) * torch.log2(torch.abs(relative_coords_table) + 1.0) / np.log2(8)
        self.register_buffer("relative_coords_table", relative_coords_table)

        # compute pairwise relative position index for tokens inside window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))  # 2, Wh, Ww
        coords_flatten = coords.flatten(1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)
        # flattened index for faster indexing
        self.register_buffer("relative_position_index_flat", relative_position_index.view(-1))

        # fused qkv projection
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

        # constants
        self._logit_scale_clamp_max = 4.605170185988091  # ln(100)
        self._sqrt_head = float(self.head_dim) ** 0.5

        # fast path availability
        self._has_sdp = hasattr(F, "scaled_dot_product_attention")

        # inference caches (populated lazily when in eval mode)
        self._cached_rel_bias = None             # fp32 rel bias kept on device
        self._cached_rel_bias_fp16 = None        # fp16 rel bias for SDP path
        self._qkv_weight_fp16 = None             # fp16 copy of qkv weight (on device)
        self._proj_weight_fp16 = None            # fp16 copy of proj weight (on device)
        self._qkv_bias_fp16 = None               # fp16 copy of assembled qkv bias (if present)

    def forward(self, x, mask=None):
        """
        x: (num_windows*B, N, C)
        mask: (nW, N, N) or None
        returns: (num_windows*B, N, C)
        """
        B_, N, C = x.shape
        device = x.device

        # qkv bias assembled on same device to avoid transfers
        qkv_bias = None
        if self.q_bias is not None:
            zeros = torch.zeros_like(self.v_bias, device=device, requires_grad=False)
            qkv_bias = torch.cat((self.q_bias.to(device), zeros, self.v_bias.to(device)))

        # Inference (eval) fast path: reuse cached rel_bias and fp16 weight copies to avoid repeated allocations and casts.
        if (not self.training) and self._has_sdp:
            # ensure caches are populated on the correct device
            if (self._cached_rel_bias is None) or (self._cached_rel_bias.device != device):
                # compute rel_bias once and cache (fp32)
                cpb_out = self.cpb_mlp(self.relative_coords_table.to(device))  # (1, Htbl, Wtbl, nH)
                cpb_flat = cpb_out.view(-1, self.num_heads)
                rel_bias = cpb_flat[self.relative_position_index_flat.to(device)]  # (N*N, nH)
                rel_bias = rel_bias.view(self.num_heads, N, N)  # (nH, N, N)
                rel_bias = 16.0 * torch.sigmoid(rel_bias)  # (nH, N, N)
                self._cached_rel_bias = rel_bias
                self._cached_rel_bias_fp16 = rel_bias.half()

                # cache fp16 copies of weights on the current device
                self._qkv_weight_fp16 = self.qkv.weight.half().to(device)
                self._proj_weight_fp16 = self.proj.weight.half().to(device)
                # cache qkv_bias in fp16 if available
                if qkv_bias is not None:
                    self._qkv_bias_fp16 = qkv_bias.half()
                else:
                    self._qkv_bias_fp16 = None

            # perform qkv projection directly in fp16 using cached fp16 weights to avoid a cast per-forward
            # x may be fp32; operate on fp16 view to leverage TensorCores
            x_fp16 = x.half()
            if self._qkv_bias_fp16 is not None:
                qkv_fp16 = F.linear(x_fp16, self._qkv_weight_fp16, bias=self._qkv_bias_fp16)  # (B_, N, 3*C) fp16
            else:
                qkv_fp16 = F.linear(x_fp16, self._qkv_weight_fp16)  # (B_, N, 3*C) fp16

            # reshape to (3, B_, nH, N, head_dim) in fp16
            qkv = qkv_fp16.view(B_, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]  # each: (B_, nH, N, head_dim) fp16

            # cosine attention: normalize q and k (in fp16)
            q = F.normalize(q, p=2, dim=-1)
            k = F.normalize(k, p=2, dim=-1)

            # use cached rel_bias fp16
            rel_bias_fp16 = self._cached_rel_bias_fp16  # (nH, N, N)

            # logit scale -> clamp and exp, match dtype to q for scaling
            logit_scale = self.logit_scale.clamp(max=self._logit_scale_clamp_max).exp().to(q.device)  # fp32
            scale = (logit_scale * self._sqrt_head).view(1, self.num_heads, 1, 1).to(q.dtype)  # cast to fp16
            q_scaled = q * scale  # (B_, nH, N, head_dim) fp16

            # prepare rp bias and mask using expand without extra copies
            rp = rel_bias_fp16.unsqueeze(0).expand(B_, -1, -1, -1)  # (B_, nH, N, N) fp16

            if mask is not None:
                nW = mask.shape[0]
                rep = B_ // nW
                mask_exp = mask.unsqueeze(0).expand(rep, -1, -1, -1).reshape(B_, 1, N, N).to(device)
                attn_mask = mask_exp.half() + rp
            else:
                attn_mask = rp

            # scaled dot product attention in fp16
            attn_out = F.scaled_dot_product_attention(q_scaled, k, v, attn_mask=attn_mask, dropout_p=self.attn_drop.p, is_causal=False)  # fp16

            # project the result using cached fp16 proj weight, then cast to fp32 to match module dtype externally
            out_fp16 = attn_out.transpose(1, 2).reshape(B_, N, C)  # fp16
            out_proj_fp16 = F.linear(out_fp16, self._proj_weight_fp16)  # fp16
            out = out_proj_fp16.to(torch.float32)
            out = self.proj_drop(out)
            return out

        # Training or fallback path (original behavior, fp32 computations)
        # fused linear projection for qkv (training keeps fp32 weights for gradients)
        qkv = F.linear(x, self.qkv.weight, bias=qkv_bias)  # (B_, N, 3*C)
        # reshape to (3, B_, nH, N, head_dim)
        qkv = qkv.view(B_, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each: (B_, nH, N, head_dim)

        # cosine attention: normalize q and k
        q = F.normalize(q, p=2, dim=-1)
        k = F.normalize(k, p=2, dim=-1)

        # compute relative position bias once per forward, minimize temporaries and ensure device-coherent tensors
        # move coordinate table and index to device for cpb computation to avoid implicit transfers
        rc_table = self.relative_coords_table.to(device)
        idx_flat = self.relative_position_index_flat.to(device)
        cpb_out = self.cpb_mlp(rc_table)  # (1, Htbl, Wtbl, nH) on device
        cpb_flat = cpb_out.view(-1, self.num_heads)
        rel_bias = cpb_flat[idx_flat]  # (N*N, nH)
        rel_bias = rel_bias.view(self.num_heads, N, N)  # (nH, N, N)
        rel_bias = 16.0 * torch.sigmoid(rel_bias)  # (nH, N, N)
        # keep a fp16 copy for the SDP path to avoid repeated casts
        rel_bias_fp16 = rel_bias.half()

        # logit scale -> clamp and exp, keep on device and match dtype when needed
        logit_scale = self.logit_scale.clamp(max=self._logit_scale_clamp_max).exp().to(q.device)  # (nH,1,1)

        if self._has_sdp:
            # Use scaled_dot_product_attention for best performance.
            # Transform cosine attention to scaled-dot-product by scaling q by (logit_scale * sqrt(head_dim))
            scale = (logit_scale * self._sqrt_head).view(1, self.num_heads, 1, 1)  # (1, nH, 1, 1)
            q_scaled = q * scale  # (B_, nH, N, head_dim)

            # Prepare rel-bias for SDP path in fp16 and expand without copies
            rp = rel_bias_fp16.unsqueeze(0).expand(B_, -1, -1, -1)  # (B_, nH, N, N) fp16

            if mask is not None:
                nW = mask.shape[0]
                rep = B_ // nW
                # use expand to avoid extra allocation, then move to device and cast once
                mask_exp = mask.unsqueeze(0).expand(rep, -1, -1, -1).reshape(B_, 1, N, N).to(device).half()
                attn_mask = mask_exp + rp
            else:
                attn_mask = rp

            # cast q,k,v to fp16 once for SDP path (TensorCore-friendly)
            q_scaled_fp16 = q_scaled.half()
            k_fp16 = k.half()
            v_fp16 = v.half()

            # run attention in fp16 and cast back to fp32 for the final projection to preserve proj precision
            attn_out = F.scaled_dot_product_attention(q_scaled_fp16, k_fp16, v_fp16, attn_mask=attn_mask, dropout_p=self.attn_drop.p, is_causal=False)  # fp16
            out = attn_out.to(torch.float32).transpose(1, 2).reshape(B_, N, C)  # cast to fp32 before proj
            out = self.proj(out)
            out = self.proj_drop(out)
            return out
        else:
            # fallback manual attention computation:
            # keep matmuls in fp16 for speed, accumulate in fp32
            q_fp16 = q.half()
            k_fp16 = k.half()
            v_fp16 = v.half()
            attn_fp16 = q_fp16 @ k_fp16.transpose(-2, -1)  # (B_, nH, N, N) fp16
            attn = attn_fp16.to(torch.float32) * logit_scale.view(1, self.num_heads, 1, 1)
            attn = attn + rel_bias.unsqueeze(0)
            if mask is not None:
                nW = mask.shape[0]
                # use expand to avoid allocations and keep shapes consistent
                attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
                attn = attn.view(-1, self.num_heads, N, N)
                attn = F.softmax(attn, dim=-1)
            else:
                attn = F.softmax(attn, dim=-1)
            attn = self.attn_drop(attn)
            out = (attn @ v).transpose(1, 2).reshape(B_, N, C)
            out = self.proj(out)
            out = self.proj_drop(out)
            return out


class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block with micro-optimizations. """
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=FastLayerNorm, pretrained_window_size=0):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
                                    qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop,
                                    pretrained_window_size=to_2tuple(pretrained_window_size))

        self.drop_path = nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        # precompute attention mask if shift is used
        if self.shift_size > 0:
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
            mask_windows = window_partition(img_mask, self.window_size)  # nW, ws, ws, 1
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
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, ws, ws, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, N, C

        # attention
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, N, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        x = x.view(B, H * W, C)

        # Apply norm1 and residual in a single step to reduce temporaries
        x = shortcut + self.drop_path(self.norm1(x))

        # FFN
        x = x + self.drop_path(self.norm2(self.mlp(x)))

        return x


class PatchMerging(nn.Module):
    r""" Patch Merging Layer. """
    def __init__(self, input_resolution, dim, norm_layer=FastLayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        # Linear reduction: 4*C -> 2*C
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(2 * dim)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."
        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], dim=-1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.reduction(x)
        x = self.norm(x)

        return x


class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage. """
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=FastLayerNorm, downsample=None, use_checkpoint=False,
                 pretrained_window_size=0):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks: alternate shift sizes
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim, input_resolution=input_resolution,
                num_heads=num_heads, window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop, attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                pretrained_window_size=pretrained_window_size
            )
            for i in range(depth)
        ])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        # iterate through blocks (checkpointing omitted for default path)
        for blk in self.blocks:
            x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding """
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

        # conv2d patch projection
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)  # B, Ph*Pw, C
        if self.norm is not None:
            x = self.norm(x)
        return x


class ModelNew(nn.Module):
    """ Optimized Swin Transformer (ModelNew) with fused attention and micro-optimizations. """
    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., qkv_bias=True,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=FastLayerNorm, patch_norm=True,
                 use_checkpoint=False, pretrained_window_sizes=[0, 0, 0, 0], **kwargs):
        super().__init__()

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        # number of output features before head
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        # Patch embedding
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # Minor optimization: use dropout module but default drop_rate is 0
        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth schedule (kept as original)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # Build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2 ** i_layer),
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
                pretrained_window_size=pretrained_window_sizes[i_layer]
            )
            self.layers.append(layer)

        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        x = self.patch_embed(x)
        x = self.pos_drop(x)

        # sequentially apply layers
        for layer in self.layers:
            x = layer(x)

        # final norm and pooling
        x = self.norm(x)  # B L C
        x = self.avgpool(x.transpose(1, 2))  # B C 1
        x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


# Input shapes as in original
batch_size = 10
image_size = 224

def get_inputs():
    return [torch.rand(batch_size, 3, image_size, image_size)]

def get_init_inputs():
    return []