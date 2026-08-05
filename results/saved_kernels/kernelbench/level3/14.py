import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

# Autotune configurations for the tiled BN+ReLU kernel (BLOCK_S = spatial tile, BLOCK_C = channel tile)
# Favor larger spatial tiles (multiples of common vector widths) to increase arithmetic intensity and reduce launches.
AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK_S": 512,   "BLOCK_C": 4},  num_warps=4, num_stages=2),
    triton.Config({"BLOCK_S": 1024,  "BLOCK_C": 4},  num_warps=4, num_stages=2),
    triton.Config({"BLOCK_S": 2048,  "BLOCK_C": 4},  num_warps=8, num_stages=2),
    triton.Config({"BLOCK_S": 2048,  "BLOCK_C": 8},  num_warps=8, num_stages=2),
    triton.Config({"BLOCK_S": 4096,  "BLOCK_C": 8},  num_warps=8, num_stages=2),
]


@triton.autotune(configs=AUTOTUNE_CONFIGS, key=['N','C','H','W'])
@triton.jit
def _bn_relu_kernel(
    x_ptr,           # pointer to input tensor (fp32)
    gamma_ptr,       # pointer to gamma (C,)
    beta_ptr,        # pointer to beta (C,)
    mean_ptr,        # pointer to running_mean (C,)
    var_ptr,         # pointer to running_var (C,)
    out_ptr,         # pointer to output tensor (fp32)
    N, C, H, W,      # tensor shapes (ints)
    eps,             # float eps
    BLOCK_S: tl.constexpr,  # spatial tile size (number of spatial positions per channel)
    BLOCK_C: tl.constexpr   # channel tile size
):
    # 2D tiling: program_id(0) -> channel block index, program_id(1) -> spatial block index
    c_block = tl.program_id(0)
    s_block = tl.program_id(1)

    hw = H * W
    n_spatial = N * hw

    c_start = c_block * BLOCK_C
    s_start = s_block * BLOCK_S

    offs_c = c_start + tl.arange(0, BLOCK_C)                   # shape [BLOCK_C]
    offs_s = s_start + tl.arange(0, BLOCK_S)                   # shape [BLOCK_S]

    mask_c = offs_c < C
    mask_s = offs_s < n_spatial

    # Load per-channel parameters once for this channel tile
    gamma = tl.load(gamma_ptr + offs_c, mask=mask_c, other=1.0)   # [BLOCK_C]
    beta = tl.load(beta_ptr + offs_c, mask=mask_c, other=0.0)     # [BLOCK_C]
    mean = tl.load(mean_ptr + offs_c, mask=mask_c, other=0.0)     # [BLOCK_C]
    var = tl.load(var_ptr + offs_c, mask=mask_c, other=1.0)       # [BLOCK_C]

    # Use rsqrt and pre-multiply by gamma for slightly cheaper compute
    inv = gamma * tl.rsqrt(var + eps)                              # [BLOCK_C]

    # compute n (batch index) and s_local (within-HW) for spatial positions
    n_idx = offs_s // hw                 # [BLOCK_S]
    s_local = offs_s % hw               # [BLOCK_S]

    # base for each spatial position: n * (C*HW)
    base_n = n_idx * (C * hw)           # [BLOCK_S]
    c_base = offs_c * hw                # [BLOCK_C]

    # broadcast to form offsets matrix of shape [BLOCK_C, BLOCK_S]
    # offs_matrix[c, s] = c_base[c] + base_n[s] + s_local[s]
    offs_matrix = c_base[:, None] + base_n[None, :] + s_local[None, :]

    # combined mask for tile
    mask_tile = mask_c[:, None] & mask_s[None, :]

    # Load input tile (BLOCK_C x BLOCK_S)
    x_tile = tl.load(x_ptr + offs_matrix, mask=mask_tile, other=0.0)

    # Apply per-channel affine transform broadcast across spatial
    out_tile = (x_tile - mean[:, None]) * inv[:, None] + beta[:, None]

    # ReLU using tl.where (robust and avoids relying on specific tl.max overloads)
    out_tile = tl.where(out_tile > 0.0, out_tile, 0.0)

    # Store results
    tl.store(out_ptr + offs_matrix, out_tile, mask=mask_tile)


def triton_bn_relu(x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor,
                   running_mean: torch.Tensor, running_var: torch.Tensor, eps: float):
    """
    Apply BatchNorm (using running statistics) followed by ReLU using Triton tiled kernel.
    Expects x to be a CUDA float32 tensor in NCHW layout.
    gamma, beta, running_mean, running_var should be 1D tensors on the same device as x.
    NOTE: To avoid per-forward device transfers, ensure model.to(device) has been called so BN params
    and running stats live on the same device as the input.
    """
    assert x.is_cuda, "Triton BN+ReLU expects CUDA tensors."

    # Avoid unnecessary copies: only make contiguous if required
    if not x.is_contiguous():
        x = x.contiguous()

    N, C, H, W = x.shape
    n_spatial = N * H * W

    device = x.device
    dtype = torch.float32

    # Provide defaults if affine parameters are missing
    if gamma is None:
        gamma = torch.ones(C, device=device, dtype=dtype)
    else:
        # Require parameters to be on the same device to avoid per-forward .to(...) overhead.
        assert gamma.device == device, "gamma must be on the same device as input. Call model.to(device) before forward."

    if beta is None:
        beta = torch.zeros(C, device=device, dtype=dtype)
    else:
        assert beta.device == device, "beta must be on the same device as input. Call model.to(device) before forward."

    assert running_mean.device == device and running_var.device == device, "running_mean/var must be on same device as input. Call model.to(device) before forward."

    # Ensure contiguous parameter tensors for efficient device loads (no copy if already contiguous)
    gamma = gamma.contiguous()
    beta = beta.contiguous()
    running_mean = running_mean.contiguous()
    running_var = running_var.contiguous()

    out = torch.empty_like(x)

    # 2D grid: (num_channel_blocks, num_spatial_blocks)
    grid = lambda meta: (
        (C + meta["BLOCK_C"] - 1) // meta["BLOCK_C"],
        (n_spatial + meta["BLOCK_S"] - 1) // meta["BLOCK_S"],
    )

    # Launch the autotuned tiled kernel; autotuner will set BLOCK_S and BLOCK_C as constexpr
    _bn_relu_kernel[grid](
        x, gamma, beta, running_mean, running_var, out,
        N, C, H, W, float(eps)
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, num_layers: int, num_input_features: int, growth_rate: int):
        """
        Optimized DenseBlock-like module:
        - Replaces BatchNorm2d + ReLU (in eval mode) with a fused Triton kernel (per-channel affine + ReLU).
        - Keeps convolution layers as native torch.nn.Conv2d (optimized highly in cuDNN/CUDA).
        """
        super(ModelNew, self).__init__()
        self.num_layers = num_layers
        self.growth_rate = growth_rate

        self.layers_bn = nn.ModuleList()
        self.layers_conv = nn.ModuleList()
        for i in range(num_layers):
            in_features = num_input_features + i * growth_rate
            bn = nn.BatchNorm2d(in_features)
            conv = nn.Conv2d(in_features, growth_rate, kernel_size=3, padding=1, bias=False)
            self.layers_bn.append(bn)
            self.layers_conv.append(conv)
        # Dropout was 0.0 in the original model; omitted.

    def forward(self, x):
        features = [x]
        # We'll maintain "x" as the concatenated feature map across iterations, matching original behavior
        for i in range(self.num_layers):
            bn = self.layers_bn[i]
            # Use the Triton fused kernel only when BatchNorm is in eval mode (uses running stats)
            if not bn.training and x.is_cuda:
                # Use BN params directly; ensure model.to(device) has been called once so params are on the correct device.
                weight = bn.weight if bn.affine else None
                bias = bn.bias if bn.affine else None
                running_mean = bn.running_mean
                running_var = bn.running_var
                x_bn = triton_bn_relu(x, weight, bias, running_mean, running_var, bn.eps)
            else:
                # Fallback to PyTorch implementation (supports training mode)
                x_bn = F.batch_norm(x, bn.running_mean, bn.running_var,
                                    bn.weight, bn.bias, training=bn.training,
                                    momentum=bn.momentum, eps=bn.eps)
                x_bn = F.relu(x_bn, inplace=False)

            new_feature = self.layers_conv[i](x_bn)
            features.append(new_feature)
            # Concatenate along channel dimension
            x = torch.cat(features, dim=1)
        return x


# Same input-generation helpers as original to remain compatible with evaluation harnesses
batch_size = 10
num_layers = 6
num_input_features = 32
growth_rate = 32
height, width = 224, 224

def get_inputs():
    return [torch.rand(batch_size, num_input_features, height, width)]

def get_init_inputs():
    return [num_layers, num_input_features, growth_rate]