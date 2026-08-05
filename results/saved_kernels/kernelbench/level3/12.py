import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

# Tuned block sizes for Ampere (A6000)
_RELU_BLOCK = 1024
_FUSED_BLOCK = 1024

# Simple in-place ReLU Triton kernel (vector-friendly)
@triton.jit
def _relu_kernel(x_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    vals = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    out = tl.where(vals > 0.0, vals, 0.0)
    tl.store(x_ptr + offsets, out, mask=mask)

def triton_relu_(x: torch.Tensor):
    """
    In-place ReLU via Triton. Falls back to torch.relu_ on CPU.
    """
    if not x.is_cuda:
        return x.relu_()
    if not x.is_contiguous():
        x = x.contiguous()
    n_elements = x.numel()
    grid = ((n_elements + _RELU_BLOCK - 1) // _RELU_BLOCK,)
    # Launch in-place: kernel writes back to same pointer
    _relu_kernel[grid](x, n_elements, BLOCK_SIZE=_RELU_BLOCK)
    return x

# Fused ReLU + 2x2 MaxPool (stride=2) Triton kernel
# Input layout: N, C, H, W (contiguous). Output: N, C, H//2, W//2
@triton.jit
def _relu_maxpool2_kernel(inp_ptr, out_ptr,
                          N, C, H, W, H_out, W_out,
                          n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # offsets index into output flattened as ((n*C + c)*H_out + h_out)*W_out + w_out
    offs = offsets

    w_out = offs % W_out
    t = offs // W_out
    h_out = t % H_out
    t = t // H_out
    c = t % C
    n = t // C

    # compute input coordinates (top-left of 2x2 window)
    h_in = h_out * 2
    w_in = w_out * 2

    # flatten input index: ((n*C + c)*H + h_in)*W + w_in
    base_idx = ((n * C + c) * H + h_in) * W + w_in

    # Offsets for 4 values in the 2x2 window
    idx0 = base_idx
    idx1 = base_idx + 1
    idx2 = base_idx + W
    idx3 = base_idx + W + 1

    # Load (use a very negative other to avoid affecting max when out-of-bounds,
    # though shapes should ensure in-bounds accesses)
    neg_inf = -1e20
    x0 = tl.load(inp_ptr + idx0, mask=mask, other=neg_inf)
    x1 = tl.load(inp_ptr + idx1, mask=mask, other=neg_inf)
    x2 = tl.load(inp_ptr + idx2, mask=mask, other=neg_inf)
    x3 = tl.load(inp_ptr + idx3, mask=mask, other=neg_inf)

    # Apply ReLU then max over the four values
    x0 = tl.where(x0 > 0.0, x0, 0.0)
    x1 = tl.where(x1 > 0.0, x1, 0.0)
    x2 = tl.where(x2 > 0.0, x2, 0.0)
    x3 = tl.where(x3 > 0.0, x3, 0.0)

    m1 = tl.maximum(x0, x1)
    m2 = tl.maximum(x2, x3)
    out = tl.maximum(m1, m2)

    tl.store(out_ptr + offsets, out, mask=mask)

def triton_relu_maxpool2(x: torch.Tensor) -> torch.Tensor:
    """
    Fused ReLU + 2x2 MaxPool (stride 2). Returns a new downsampled tensor.
    Falls back to F.relu + F.max_pool2d on CPU.
    """
    if not x.is_cuda:
        return F.max_pool2d(F.relu(x), kernel_size=2, stride=2)

    # Ensure contiguous input
    x = x.contiguous()
    N, C, H, W = x.shape
    assert H % 2 == 0 and W % 2 == 0, "H and W must be divisible by 2 for 2x2 pool"
    H_out, W_out = H // 2, W // 2

    out = torch.empty((N, C, H_out, W_out), device=x.device, dtype=x.dtype)

    n_elements = out.numel()
    grid = ((n_elements + _FUSED_BLOCK - 1) // _FUSED_BLOCK,)

    _relu_maxpool2_kernel[grid](x, out,
                                N, C, H, W, H_out, W_out,
                                n_elements,
                                BLOCK_SIZE=_FUSED_BLOCK)
    return out

# Marker module for where pooling should occur (replaces MaxPool2d in the module list)
class PoolMark(nn.Module):
    def forward(self, x):
        # Acts as placeholder; fused kernel will be applied in the forward loop
        return x

class ModelNew(nn.Module):
    """
    VGG19-like model where:
      - Elementwise ReLUs are run in-place via Triton for conv/linear outputs.
      - The final ReLU before each 2x2 MaxPool is fused with the pooling into a single Triton kernel
        that performs ReLU and 2x2 max pooling in one pass, reducing memory traffic.
    """
    def __init__(self, num_classes=1000):
        super(ModelNew, self).__init__()

        # Build features: replace ReLU with Identity placeholders and MaxPool2d with PoolMark.
        # We'll run Triton ReLU after each Conv2d, except the conv immediately before a PoolMark,
        # for which we'll run the fused relu+pool kernel.
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.Identity(),  # placeholder for ReLU
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.Identity(),
            PoolMark(),

            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.Identity(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.Identity(),
            PoolMark(),

            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.Identity(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.Identity(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.Identity(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.Identity(),
            PoolMark(),

            # Block 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.Identity(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.Identity(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.Identity(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.Identity(),
            PoolMark(),

            # Block 5
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.Identity(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.Identity(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.Identity(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.Identity(),
            PoolMark()
        )

        # Classifier: keep linear layers, apply Triton in-place ReLU after each Linear
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.Identity(),
            nn.Dropout(p=0.0),
            nn.Linear(4096, 4096),
            nn.Identity(),
            nn.Dropout(p=0.0),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        # Iterate through feature modules. Apply Triton kernels opportunistically:
        # - After each Conv2d: if the conv is immediately before a PoolMark (i+2), run fused relu+pool.
        # - Otherwise, run in-place Triton ReLU on the conv output.
        i = 0
        L = len(self.features)
        while i < L:
            layer = self.features[i]
            x = layer(x)

            if isinstance(layer, nn.Conv2d):
                # Check if this conv is the one right before a PoolMark (pattern conv, Identity, PoolMark)
                if (i + 2) < L and isinstance(self.features[i + 2], PoolMark):
                    # Use fused Triton ReLU + 2x2 MaxPool (stride 2)
                    x = triton_relu_maxpool2(x)
                    # continue; placeholders (Identity and PoolMark) will be skipped logically,
                    # but the loop will still visit them; they are no-ops so it's fine.
                else:
                    # Regular in-place Triton ReLU
                    triton_relu_(x)
            # For other layers (Identity, PoolMark) do nothing; they are placeholders
            i += 1

        x = torch.flatten(x, 1)

        # Classifier: apply linear layers and then in-place Triton ReLU after each Linear
        for layer in self.classifier:
            x = layer(x)
            if isinstance(layer, nn.Linear):
                triton_relu_(x)

        return x

# Test helpers analogous to the original module
batch_size = 10
num_classes = 1000

def get_inputs():
    return [torch.rand(batch_size, 3, 224, 224, dtype=torch.float32)]

def get_init_inputs():
    return [num_classes]