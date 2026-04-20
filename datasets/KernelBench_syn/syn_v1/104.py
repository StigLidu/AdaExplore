import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Complex convolutional module combining Conv2d, LazyBatchNorm2d, SiLU activation,
    FeatureAlphaDropout and a small projection head. The model demonstrates a
    non-trivial forward path with a residual-like pattern and global pooling.

    Forward signature:
        forward(x: Tensor) -> Tensor

    Expected input shape:
        x: (batch_size, in_channels=3, height, width)
    """
    def __init__(self, out_channels: int = 64, dropout_p: float = 0.1, expansion: int = 2):
        """
        Initializes the module.

        Args:
            out_channels (int): Number of output channels for the first convolution.
            dropout_p (float): Dropout probability for FeatureAlphaDropout.
            expansion (int): Channel expansion factor for the second convolution.
        """
        super(Model, self).__init__()
        self.in_channels = 3
        self.out_channels = out_channels
        self.expanded_channels = out_channels * expansion
        self.dropout_p = dropout_p
        self.expansion = expansion

        # First conv block (no bias because BatchNorm will follow)
        self.conv1 = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=3, padding=1, bias=False)
        # LazyBatchNorm2d will infer num_features on first forward pass
        self.bn1 = nn.LazyBatchNorm2d()
        # SiLU activation (also known as Swish)
        self.act = nn.SiLU()
        # FeatureAlphaDropout (channel-wise dropout variant)
        self.fad = nn.FeatureAlphaDropout(p=self.dropout_p)

        # Second conv expands channel dimension
        self.conv2 = nn.Conv2d(self.out_channels, self.expanded_channels, kernel_size=3, padding=1, bias=False)
        # Small projection head after global pooling
        self.fc = nn.Linear(self.expanded_channels, self.out_channels, bias=True)

        # A learnable scale to mix residual and transformed paths
        self.res_scale = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass combining the layers into a composite computation.

        Computation steps:
            1. conv1 -> lazy batchnorm -> SiLU
            2. Save residual from step 1
            3. Apply FeatureAlphaDropout
            4. conv2 -> SiLU
            5. Fuse residual (channel-wise) with transformed path using learnable scale
            6. Global average pooling over spatial dims
            7. Final linear projection

        Args:
            x (torch.Tensor): Input tensor with shape (B, 3, H, W)

        Returns:
            torch.Tensor: Output tensor with shape (B, out_channels)
        """
        # Step 1: convolution, normalization, activation
        x1 = self.conv1(x)         # (B, out_channels, H, W)
        x1 = self.bn1(x1)          # LazyBatchNorm2d will be initialized here
        x1 = self.act(x1)          # SiLU activation

        # Step 2: residual from early representation
        residual = x1

        # Step 3: stochastic regularization across channels
        x2 = self.fad(x1)          # FeatureAlphaDropout

        # Step 4: expand channels and activate
        x2 = self.conv2(x2)        # (B, expanded_channels, H, W)
        x2 = self.act(x2)

        # Step 5: fuse residual into expanded channel space by simple channel-wise projection
        # Project residual to expanded channels using a grouped 1x1 convolution implemented via linear-like matmul:
        # reshape residual -> (B, out_channels, H*W), expand to match expanded_channels by tiling groups
        B, C, H, W = residual.shape
        # Simple deterministic channel expansion by repeating groups (keeps parameter-free)
        repeat_factor = self.expanded_channels // self.out_channels
        if self.expanded_channels % self.out_channels != 0:
            # Fallback: if not divisible, resize via interpolation along channel dim (rare for our init choices)
            residual_proj = F.interpolate(residual, size=(H, W), mode='nearest')
            residual_proj = residual_proj.repeat(1, (self.expanded_channels + C - 1) // C, 1, 1)[:, :self.expanded_channels, :, :]
        else:
            residual_proj = residual.repeat(1, repeat_factor, 1, 1)  # (B, expanded_channels, H, W)

        # Combine with learnable scale (broadcasting)
        combined = self.res_scale * residual_proj + (1.0 - self.res_scale) * x2  # (B, expanded_channels, H, W)

        # Step 6: global average pooling
        pooled = combined.mean(dim=(2, 3))  # (B, expanded_channels)

        # Step 7: final projection
        out = self.fc(pooled)  # (B, out_channels)
        return out

# Configuration for input generation
batch_size = 8
in_channels = 3
height = 128
width = 128

# Default initialization parameters for Model
init_out_channels = 64
init_dropout_p = 0.12
init_expansion = 2

def get_inputs():
    """
    Generates an input tensor matching the expected shape for the model.

    Returns:
        list: [x] where x is a torch.Tensor of shape (batch_size, 3, height, width)
    """
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]

def get_init_inputs():
    """
    Returns the initialization parameters for the Model constructor.

    Returns:
        list: [out_channels, dropout_p, expansion]
    """
    return [init_out_channels, init_dropout_p, init_expansion]