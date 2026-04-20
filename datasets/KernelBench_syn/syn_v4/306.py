import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Complex module that demonstrates a small generator-like block:
    - Zero padding to increase receptive field
    - Lazy transposed convolution to learn an upsampling transform
    - ReLU activation
    - Local Response Normalization across channels
    - Spatial gating derived from the input (channel-averaged and resized)
    - Final non-linearity (tanh)

    The LazyConvTranspose2d lazily infers input channels on first forward pass.
    """
    def __init__(self, out_channels: int):
        """
        Initializes the composite module.

        Args:
            out_channels (int): Number of output channels produced by the transposed conv.
        """
        super(Model, self).__init__()
        # Zero pad (left, right, top, bottom)
        self.pad = nn.ZeroPad2d(PAD)
        # Lazy transposed convolution: in_channels inferred on first call
        self.deconv = nn.LazyConvTranspose2d(
            out_channels=out_channels,
            kernel_size=KERNEL_SIZE,
            stride=STRIDE,
            padding=CONV_PADDING
        )
        self.act = nn.ReLU()
        # Local Response Norm acts across channels
        self.lrn = nn.LocalResponseNorm(size=LRN_SIZE, alpha=LRN_ALPHA, beta=LRN_BETA, k=LRN_K)
        # A small scalar residual weight (learnable) to blend gating cleanly
        self.res_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the composite module.

        Steps:
            1. Zero-pad the input to expand spatial support.
            2. Apply a lazy ConvTranspose2d (learned upsampling).
            3. Apply ReLU activation.
            4. Normalize across channels with LocalResponseNorm.
            5. Build a spatial gating map from the input (channel-averaged -> resized).
            6. Multiply the normalized features by the gating map (broadcasted).
            7. Blend a small residual of the normalized features back in and apply tanh.

        Args:
            x (torch.Tensor): Input tensor shaped (B, C, H, W).

        Returns:
            torch.Tensor: Output tensor shaped (B, out_channels, H_out, W_out).
        """
        # 1) Zero pad
        x_padded = self.pad(x)

        # 2) Transposed convolution (upsampling-like learnable operation)
        y = self.deconv(x_padded)

        # 3) Activation
        y = self.act(y)

        # 4) Local response normalization (channel-wise competition)
        y = self.lrn(y)

        # 5) Spatial gating: derive a single-channel spatial map from input and resize
        #    - Average across input channels to produce a spatial importance map
        gate = x.mean(dim=1, keepdim=True)  # (B,1,H,W)
        # Resize the gate to match y's spatial dims
        _, _, H_out, W_out = y.shape
        gate_resized = F.interpolate(gate, size=(H_out, W_out), mode='bilinear', align_corners=False)
        # Expand gate to match y's channel dimension
        gate_expanded = gate_resized.expand(-1, y.shape[1], -1, -1)  # (B, out_channels, H_out, W_out)

        # 6) Apply gating (element-wise modulation)
        y_gated = y * torch.sigmoid(gate_expanded)  # sigmoid to keep gating between 0 and 1

        # 7) Blend a small residual of the normalized features back and final non-linearity
        output = torch.tanh(y_gated + self.res_scale * y)

        return output

# Module level configuration variables
batch_size = 8
in_channels = 3
out_channels = 16
height = 32
width = 32

# ConvTranspose parameters chosen to demonstrate upsampling behavior
KERNEL_SIZE = 4
STRIDE = 2
CONV_PADDING = 1

# ZeroPad2d expects (left, right, top, bottom)
PAD = (1, 1, 1, 1)

# LocalResponseNorm parameters
LRN_SIZE = 5
LRN_ALPHA = 1e-4
LRN_BETA = 0.75
LRN_K = 1.0

def get_inputs():
    """
    Returns a list containing a single input tensor consistent with the configuration.
    Shape: (batch_size, in_channels, height, width)
    """
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]

def get_init_inputs():
    """
    Returns initialization inputs required for Model.__init__.
    """
    return [out_channels]