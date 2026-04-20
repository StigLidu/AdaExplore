import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Any

class Model(nn.Module):
    """
    Complex sequence transformation module that:
    - Upsamples the 1D signal via ConvTranspose1d
    - Applies AvgPool1d pooling
    - Applies a Lazy 2D InstanceNorm by reshaping to (B, C, 1, L)
    - Performs a channel-wise gating derived from global temporal statistics
    - Adds a resized residual branch from the original input (channel-adjusted via padding/truncation)
    This combines ConvTranspose1d, AvgPool1d, and LazyInstanceNorm2d in a non-trivial processing graph.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        output_padding: int,
        pool_kernel: int,
        pool_stride: int,
        inst_eps: float = 1e-5,
        inst_affine: bool = True
    ):
        """
        Initializes the module with the specified convolution/pooling parameters.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels produced by ConvTranspose1d.
            kernel_size (int): Kernel size for ConvTranspose1d.
            stride (int): Stride for ConvTranspose1d.
            padding (int): Padding for ConvTranspose1d.
            output_padding (int): Additional size added to one side of the output shape.
            pool_kernel (int): Kernel size for AvgPool1d.
            pool_stride (int): Stride for AvgPool1d.
            inst_eps (float): Epsilon for LazyInstanceNorm2d.
            inst_affine (bool): Whether LazyInstanceNorm2d has learnable affine parameters.
        """
        super(Model, self).__init__()
        # Upsampling transposed convolution (1D)
        self.deconv = nn.ConvTranspose1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding
        )
        # Average pooling (1D)
        self.pool = nn.AvgPool1d(kernel_size=pool_kernel, stride=pool_stride)
        # Lazy InstanceNorm2d will be initialized on first forward pass when num_features is known
        self.inst_norm = nn.LazyInstanceNorm2d(eps=inst_eps, affine=inst_affine)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, seq_length)

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, L_out)
        """
        # Step 1: Upsample via ConvTranspose1d -> (B, out_channels, L1)
        y = self.deconv(x)

        # Step 2: Smooth/aggregate via AvgPool1d -> (B, out_channels, L2)
        y = self.pool(y)

        # Step 3: Non-linearity
        y = torch.relu(y)

        # Step 4: Apply LazyInstanceNorm2d by reshaping to 4D: (B, C, H=1, W=L2)
        y_4d = y.unsqueeze(2)  # (B, C, 1, L2)
        y_4d = self.inst_norm(y_4d)  # LazyInstanceNorm2d initializes here
        y = y_4d.squeeze(2)  # back to (B, C, L2)

        # Step 5: Channel-wise global temporal pooling to produce gates (Squeeze-and-Excitation-like)
        # Compute channel-wise temporal average: (B, C, 1)
        channel_avg = y.mean(dim=2, keepdim=True)

        # Derive a centered gate and squash with sigmoid: emphasize channels with above-average energy
        mean_across_channels = channel_avg.mean(dim=1, keepdim=True)  # (B, 1, 1)
        gate = torch.sigmoid(channel_avg - mean_across_channels)  # (B, C, 1)

        # Apply gate to the features
        gated = y * gate  # broadcasting over time dimension -> (B, C, L2)

        # Step 6: Residual addition from original input:
        # Resize original input temporally to match current length and adjust channels by padding/truncation
        target_len = gated.shape[2]
        orig_resized = F.interpolate(x, size=target_len, mode='nearest')  # (B, in_channels, target_len)

        # Channel adjustment: pad with zeros if fewer channels, or truncate if more
        in_ch = orig_resized.shape[1]
        out_ch = gated.shape[1]
        if in_ch < out_ch:
            pad_ch = out_ch - in_ch
            pad_tensor = orig_resized.new_zeros(orig_resized.size(0), pad_ch, orig_resized.size(2))
            orig_adjusted = torch.cat([orig_resized, pad_tensor], dim=1)
        else:
            orig_adjusted = orig_resized[:, :out_ch, :]

        # Final fused output: gated features plus a scaled residual branch
        out = gated + 0.5 * orig_adjusted

        return out

# Configuration / example parameters
batch_size = 8
in_channels = 12
out_channels = 32
seq_length = 256
kernel_size = 4
stride = 2
padding = 1
output_padding = 0
pool_kernel = 3
pool_stride = 2
inst_eps = 1e-5
inst_affine = True

def get_inputs() -> List[torch.Tensor]:
    """
    Returns example input tensors for the model's forward method.
    """
    x = torch.randn(batch_size, in_channels, seq_length)
    return [x]

def get_init_inputs() -> List[Any]:
    """
    Returns the initialization parameters to construct the Model instance.
    Order matches Model.__init__ signature.
    """
    return [
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        output_padding,
        pool_kernel,
        pool_stride,
        inst_eps,
        inst_affine,
    ]