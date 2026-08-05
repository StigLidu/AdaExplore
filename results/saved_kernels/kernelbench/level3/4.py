import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

# Autotune configurations for the Triton kernels
AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK": 128},  num_warps=2, num_stages=2),
    triton.Config({"BLOCK": 256},  num_warps=4, num_stages=2),
    triton.Config({"BLOCK": 512},  num_warps=8, num_stages=2),
    triton.Config({"BLOCK": 1024}, num_warps=8, num_stages=3),
]

@triton.autotune(configs=AUTOTUNE_CONFIGS, key=['n_elems'])
@triton.jit
def _relu_maxpool2d_kernel(
    inp_ptr,         # input pointer (N, C, H, W) flattened
    out_ptr,         # output pointer (N, C, Hout, Wout) flattened
    N, C, H, W,      # input sizes
    Hout, Wout,      # output sizes
    k_h, k_w,        # kernel height/width (e.g., 2,2)
    s_h, s_w,        # stride height/width (e.g., 2,2)
    n_elems,         # total number of output elements (N*C*Hout*Wout)
    BLOCK: tl.constexpr,
):
    """
    Triton kernel that fuses ReLU followed by 2D max-pooling (kernel k_h x k_w, stride s_h x s_w).
    Each program processes BLOCK output elements (flattened).
    """
    block_start = tl.program_id(0) * BLOCK
    offs = block_start + tl.arange(0, BLOCK)
    mask = offs < n_elems

    # compute output coordinates from flattened offsets:
    # offs -> n, c, h_out, w_out
    w_out = offs % Wout
    tmp = offs // Wout
    h_out = tmp % Hout
    tmp = tmp // Hout
    c = tmp % C
    n = tmp // C

    # compute top-left input coordinates for the pooling window
    h_in = h_out * s_h
    w_in = w_out * s_w

    # Compute flattened input indices for the k_h x k_w block.
    # For a 2x2 window:
    # idx0 = ((n*C + c)*H + (h_in + 0))*W + (w_in + 0)
    # idx1 = idx0 + 1
    # idx2 = idx0 + W
    # idx3 = idx2 + 1
    # Generalize using k_h,k_w loops but unrolled for common 2x2 case for efficiency.

    # Make sure all intermediate computations are tl tensors
    base = ((n * C + c) * H + h_in) * W + w_in  # shape: [BLOCK]
    # masks for each element in the pooling window
    # For general safety, check bounds for each position
    # position (0,0)
    mask0 = mask & (h_in < H) & (w_in < W)
    v0 = tl.load(inp_ptr + base, mask=mask0, other=-1e9)
    # position (0,1)
    mask1 = mask & (h_in < H) & (w_in + 1 < W)
    v1 = tl.load(inp_ptr + base + 1, mask=mask1, other=-1e9)
    # position (1,0)
    mask2 = mask & (h_in + 1 < H) & (w_in < W)
    v2 = tl.load(inp_ptr + base + W, mask=mask2, other=-1e9)
    # position (1,1)
    mask3 = mask & (h_in + 1 < H) & (w_in + 1 < W)
    v3 = tl.load(inp_ptr + base + W + 1, mask=mask3, other=-1e9)

    # Apply ReLU to each loaded value
    zero = tl.zeros_like(v0)
    v0r = tl.where(v0 > 0.0, v0, zero)
    v1r = tl.where(v1 > 0.0, v1, zero)
    v2r = tl.where(v2 > 0.0, v2, zero)
    v3r = tl.where(v3 > 0.0, v3, zero)

    # pairwise maximums to compute max over the 2x2 window
    m01 = tl.where(v0r > v1r, v0r, v1r)
    m23 = tl.where(v2r > v3r, v2r, v3r)
    mout = tl.where(m01 > m23, m01, m23)

    # store flattened output
    tl.store(out_ptr + offs, mout, mask=mask)


def triton_relu_maxpool2d(x: torch.Tensor, kernel_size=2, stride=2):
    """
    Wrapper that calls the Triton kernel to perform ReLU followed by max-pooling.
    Assumes 4D tensor in (N, C, H, W), dtype float32, CUDA device.
    """
    assert x.is_cuda and x.dtype == torch.float32, "triton_relu_maxpool2d expects CUDA float32 tensor"
    x = x.contiguous()
    N, C, H, W = x.shape
    k_h = k_w = kernel_size
    s_h = s_w = stride
    # compute output dimensions (standard floor behavior)
    Hout = (H - k_h) // s_h + 1
    Wout = (W - k_w) // s_w + 1
    out = torch.empty((N, C, Hout, Wout), device=x.device, dtype=x.dtype)

    n_elems = N * C * Hout * Wout

    # grid based on BLOCK meta
    grid = lambda meta: ((n_elems + meta["BLOCK"] - 1) // meta["BLOCK"],)

    # launch kernel
    _relu_maxpool2d_kernel[grid](
        x, out,
        N, C, H, W,
        Hout, Wout,
        k_h, k_w,
        s_h, s_w,
        n_elems,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, num_classes):
        """
        Optimized LeNet-5 using Triton kernels to fuse ReLU + MaxPool operations.
        """
        super(ModelNew, self).__init__()

        # Convolutional layers remain as PyTorch implementations
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1)

        # Fully connected layers (left as standard PyTorch)
        self.fc1 = nn.Linear(in_features=16 * 5 * 5, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(in_features=84, out_features=num_classes)

    def forward(self, x):
        # conv1 -> fused ReLU + 2x2 maxpool
        x = self.conv1(x)
        x = triton_relu_maxpool2d(x, kernel_size=2, stride=2)

        # conv2 -> fused ReLU + 2x2 maxpool
        x = self.conv2(x)
        x = triton_relu_maxpool2d(x, kernel_size=2, stride=2)

        # Flatten for fully connected layers
        x = x.view(-1, 16 * 5 * 5)

        # Fully connected layers with ReLU between them (use PyTorch ops)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Helper functions to match original module interface (kept minimal)
batch_size = 4096
num_classes = 20

def get_inputs():
    # Return a sample input tensor (on CUDA if available)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return [torch.rand(batch_size, 1, 32, 32, device=device, dtype=torch.float32)]

def get_init_inputs():
    return [num_classes]