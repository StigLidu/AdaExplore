import torch
import torch.nn as nn

# Configuration / shape parameters
BATCH = 4
CHANNELS = 3
DEPTH = 5
HEIGHT = 64
WIDTH = 64

# Padding for ConstantPad3d: (left, right, top, bottom, front, back)
# After padding: H -> HEIGHT + top + bottom, W -> WIDTH + left + right
PAD = (1, 1, 2, 2, 0, 0)  # produces H'=68, W'=66 which are divisible by pool kernel 2

class Model(nn.Module):
    """
    Model demonstrating a mixed 3D->2D processing pipeline:
      - ConstantPad3d pads the 5D volumetric input (N, C, D, H, W).
      - Depth dimension is folded into channels to form a 4D tensor (N, C*D', H', W').
      - MaxPool2d (with indices) performs spatial downsampling.
      - A simple affine scaling is applied to pooled features.
      - MaxUnpool2d uses saved indices to restore the pooled spatial resolution.
      - The result is unfolded back to the original 5D layout and clipped via Hardtanh.
    """
    def __init__(self,
                 pad: tuple = PAD,
                 pool_kernel: int = 2,
                 pool_stride: int = 2,
                 unpool_kernel: int = 2,
                 unpool_stride: int = 2,
                 pad_value: float = 0.5,
                 hard_min: float = -0.5,
                 hard_max: float = 0.7):
        super(Model, self).__init__()
        # 3D constant padding layer
        self.pad3d = nn.ConstantPad3d(pad, value=pad_value)
        # MaxPool2d returns indices needed for MaxUnpool2d
        self.pool2d = nn.MaxPool2d(kernel_size=pool_kernel,
                                   stride=pool_stride,
                                   return_indices=True)
        # MaxUnpool2d to invert the pooling
        self.unpool2d = nn.MaxUnpool2d(kernel_size=unpool_kernel,
                                       stride=unpool_stride)
        # HardTanh activation to clamp outputs
        self.hardtanh = nn.Hardtanh(min_val=hard_min, max_val=hard_max)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (N, C, D, H, W)

        Returns:
            Tensor of shape (N, C, D', H, W) where D' is the padded depth (or same if no depth padding).
        """
        # 1) Pad the 3D volume
        # x_padded: (N, C, D_p, H_p, W_p)
        x_padded = self.pad3d(x)

        # 2) Fold depth into channels to apply 2D spatial pooling
        # We permute to (N, D_p, C, H_p, W_p) then reshape to (N, C * D_p, H_p, W_p)
        N, C, Dp, Hp, Wp = x_padded.shape
        x_permuted = x_padded.permute(0, 2, 1, 3, 4).contiguous()
        x_4d = x_permuted.view(N, C * Dp, Hp, Wp)

        # 3) MaxPool2d with indices for unpooling later
        pooled, indices = self.pool2d(x_4d)

        # 4) Simple affine transform on pooled features (element-wise arithmetic)
        # This demonstrates an intermediate tensor operation between pool/unpool
        scaled = pooled * 0.75 + 0.1

        # 5) MaxUnpool2d to restore the spatial resolution
        # Provide output_size equal to the original 4D shape before pooling
        unpooled = self.unpool2d(scaled, indices, output_size=x_4d.shape)

        # 6) Unfold channels back into the depth dimension
        # unpooled: (N, C * Dp, H_p, W_p) -> reshape to (N, Dp, C, H_p, W_p) -> permute to (N, C, Dp, H_p, W_p)
        unflat = unpooled.view(N, Dp, C, Hp, Wp).permute(0, 2, 1, 3, 4)

        # 7) Apply Hardtanh clamping and return
        out = self.hardtanh(unflat)
        return out

def get_inputs():
    """
    Returns example input tensors compatible with Model.forward:
    - A random volumetric tensor of shape (BATCH, CHANNELS, DEPTH, HEIGHT, WIDTH)
    """
    x = torch.randn(BATCH, CHANNELS, DEPTH, HEIGHT, WIDTH)
    return [x]

def get_init_inputs():
    """
    No special initialization inputs are required for this model instance.
    """
    return []