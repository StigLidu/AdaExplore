import torch
import torch.nn as nn
import torch.nn.functional as F

# Configuration
batch_size = 8
channels = 32
height = 64
width = 64
PAD = 1  # circular padding size to preserve spatial dims with a 3x3 depthwise conv
DW_KERNEL = 3
POOL_OUTPUT = (1, 1)  # adaptive pool output size (H_out, W_out)

class Model(nn.Module):
    """
    Complex image-processing module that demonstrates:
      - Circular padding to provide wrap-around context,
      - Depthwise separable convolution (depthwise conv followed by pointwise conv),
      - GELU non-linearity,
      - Adaptive average pooling to compute global channel descriptors,
      - A learned channel gating (squeeze-and-excite style) applied as an attention mask,
      - Residual connection to the original input.

    The computation pattern intentionally mixes spatially-aware convolutions with global
    channel-wise modulation to produce an output of the same shape as the input.
    """
    def __init__(self, in_channels: int, pad: int = PAD, dw_kernel: int = DW_KERNEL, pool_output=(1,1)):
        """
        Initializes the module components.

        Args:
            in_channels (int): Number of input channels (and internal channels used).
            pad (int): Circular padding applied before the depthwise conv.
            dw_kernel (int): Kernel size for the depthwise convolution (must be odd for symmetry).
            pool_output (tuple): Output size (H_out, W_out) for AdaptiveAvgPool2d to compute descriptors.
        """
        super(Model, self).__init__()
        assert dw_kernel % 2 == 1, "Depthwise kernel should be odd for symmetric receptive field."

        self.pad = nn.CircularPad2d(pad)
        # Depthwise convolution: groups=in_channels to convolve each channel independently
        self.conv_dw = nn.Conv2d(in_channels=in_channels,
                                 out_channels=in_channels,
                                 kernel_size=dw_kernel,
                                 stride=1,
                                 padding=0,  # padding handled by CircularPad2d
                                 groups=in_channels,
                                 bias=True)
        # Pointwise convolution to mix channels after depthwise spatial filtering
        self.conv_pw = nn.Conv2d(in_channels=in_channels,
                                 out_channels=in_channels,
                                 kernel_size=1,
                                 bias=True)

        # Non-linear activation
        self.gelu = nn.GELU()

        # Adaptive pooling to produce compact channel descriptors
        self.pool = nn.AdaptiveAvgPool2d(output_size=pool_output)

        # Small MLP to produce channel gating values (squeeze-and-excite style)
        # Since pool_output is typically (1,1), the fc reduces to a linear per-channel transform.
        pooled_size = in_channels * pool_output[0] * pool_output[1]
        # Map pooled descriptor back to channel gating logits
        self.fc = nn.Linear(pooled_size, in_channels)

        # Initialize weights with a stable scheme
        nn.init.kaiming_normal_(self.conv_dw.weight, nonlinearity='linear')
        if self.conv_dw.bias is not None:
            nn.init.zeros_(self.conv_dw.bias)
        nn.init.kaiming_normal_(self.conv_pw.weight, nonlinearity='linear')
        if self.conv_pw.bias is not None:
            nn.init.zeros_(self.conv_pw.bias)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (B, C, H, W).
        """
        # Preserve a reference for the residual connection
        residual = x

        # 1) Circular pad to provide wrap-around context
        x_p = self.pad(x)

        # 2) Depthwise convolution picks up spatial patterns per channel
        x_dw = self.conv_dw(x_p)

        # 3) Non-linearity (GELU)
        x_act = self.gelu(x_dw)

        # 4) Pointwise convolution to mix channel information
        x_pw = self.conv_pw(x_act)

        # 5) Produce a compact descriptor via adaptive average pooling
        x_pool = self.pool(x_pw)  # shape: (B, C, pool_h, pool_w)

        # 6) Squeeze to vector and compute channel gating logits
        b = x_pool.shape[0]
        x_vec = x_pool.view(b, -1)  # (B, pooled_size)
        gating = self.fc(x_vec)     # (B, C)
        gating = self.gelu(gating)
        gating = torch.sigmoid(gating)  # (B, C) in (0,1)

        # 7) Expand gating to spatial dims and apply as multiplicative attention
        gating = gating.view(b, -1, 1, 1)  # (B, C, 1, 1)
        x_modulated = x_pw * gating  # (B, C, H, W)

        # 8) Residual connection with a simple scaling to aid gradient flow
        out = x_modulated + residual * 0.5

        return out

def get_inputs():
    """
    Returns a list with a single input tensor suitable for the model.
    """
    x = torch.randn(batch_size, channels, height, width)
    return [x]

def get_init_inputs():
    """
    Returns initialization parameters for constructing the Model.
    In this case, the number of channels is required.
    """
    return [channels]