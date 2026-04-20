import torch
import torch.nn as nn

# Configuration
batch_size = 8
in_channels = 16
depth = 4
height = 32
width = 32

out_channels = 24
deconv_kernel_size = 3
deconv_stride = 2
deconv_padding = 1
pad_values = (1, 2, 1, 2)  # (left, right, top, bottom)
pad_constant = 0.1

class Model(nn.Module):
    """
    Complex module that combines 3D batch normalization, a 2D transposed convolution applied
    across depth slices, constant padding and a simple depth-aggregation to produce a 4D output.
    """
    def __init__(self,
                 in_ch: int = in_channels,
                 out_ch: int = out_channels,
                 k: int = deconv_kernel_size,
                 s: int = deconv_stride,
                 p: int = deconv_padding,
                 pad_vals: tuple = pad_values,
                 pad_const: float = pad_constant):
        super(Model, self).__init__()
        # Normalize over (N, C, D, H, W)
        self.bn3d = nn.BatchNorm3d(in_ch)

        # We'll apply ConvTranspose2d to each depth slice independently by reshaping:
        # (N, C, D, H, W) -> (N*D, C, H, W) -> deconv -> reshape back
        self.deconv = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False)

        # Constant padding applied after the transposed convolution on the 2D spatial dims
        self.pad2d = nn.ConstantPad2d(pad_vals, pad_const)

        # Small epsilon for numeric stability if needed in aggregation
        self.eps = 1e-6

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass:
        1. BatchNorm3d over (N, C, D, H, W)
        2. Permute to bring depth into batch dimension and reshape to 4D
        3. ConvTranspose2d on spatial dimensions
        4. ConstantPad2d to adjust spatial sizes
        5. ReLU activation
        6. Reshape back to 5D and aggregate across depth with a depth-wise mean to return a 4D tensor

        Args:
            x (torch.Tensor): Input tensor of shape (N, C_in, D, H, W)

        Returns:
            torch.Tensor: Output tensor of shape (N, C_out, H_out, W_out)
        """
        N, C, D, H, W = x.shape

        # 1) Batch normalization across 5D input
        x = self.bn3d(x)  # (N, C, D, H, W)

        # 2) Move depth dimension into the batch for 2D conv processing
        #    From (N, C, D, H, W) -> (N, D, C, H, W) -> (N*D, C, H, W)
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = x.view(N * D, C, H, W)

        # 3) 2D transposed convolution applied to each depth-slice flattened into batch
        x = self.deconv(x)  # (N*D, out_ch, H_out, W_out)

        # 4) Constant padding to introduce asymmetric padding across spatial dimensions
        x = self.pad2d(x)  # (N*D, out_ch, H_out_padded, W_out_padded)

        # 5) Activation
        x = torch.relu(x)

        # 6) Reshape back to (N, out_ch, D, H_out_padded, W_out_padded)
        _, C_out, H_out_p, W_out_p = x.shape
        x = x.view(N, D, C_out, H_out_p, W_out_p).permute(0, 2, 1, 3, 4).contiguous()

        # Aggregate across depth dimension using a numerically-stable mean
        out = x.mean(dim=2)  # (N, out_ch, H_out_p, W_out_p)

        return out

# Input generation functions to match the expected signatures used in the examples
def get_inputs():
    # Create a random 5D input (N, C, D, H, W)
    x = torch.randn(batch_size, in_channels, depth, height, width)
    return [x]

def get_init_inputs():
    # No special initialization parameters required for this model beyond configuration variables
    return []