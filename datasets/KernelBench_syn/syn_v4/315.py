import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Complex convolutional head that demonstrates a sequence of spatial padding,
    convolution, channel-wise normalization via LayerNorm, FeatureAlphaDropout,
    global pooling and a final linear projection.

    The forward pass:
      1. Reflection pad the input spatially.
      2. 3x3 convolution to expand channels.
      3. Non-linear activation (GELU).
      4. Permute to apply LayerNorm over channels (LayerNorm expects channel last).
      5. Apply FeatureAlphaDropout for robust channel dropout.
      6. Permute back to channel-first.
      7. Global average pooling to (1,1).
      8. Flatten and final Linear projection.

    This design mixes padding, normalization, and structured channel dropout
    to produce a reusable feature extractor block.
    """
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_features: int,
        pad: int = 1,
        dropout_p: float = 0.1,
    ):
        """
        Initializes the module.

        Args:
            in_channels (int): Number of input channels.
            hidden_channels (int): Number of channels after the 3x3 convolution.
            out_features (int): Size of the final output vector per sample.
            pad (int): Reflection padding size applied on all sides (symmetric).
            dropout_p (float): Probability for FeatureAlphaDropout.
        """
        super(Model, self).__init__()
        # Reflection padding to preserve spatial dimensions when using 3x3 conv with no internal padding
        self.ref_pad = nn.ReflectionPad2d(pad)

        # 3x3 convolution expands and mixes spatial info
        self.conv = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, stride=1, padding=0, bias=True)

        # Non-linearity
        self.act = nn.GELU()

        # LayerNorm will be applied over the channel dimension after permuting to channel-last.
        # normalized_shape expects the size of the last dimension -> hidden_channels
        self.layer_norm = nn.LayerNorm(hidden_channels)

        # FeatureAlphaDropout operates channel-wise (works well with LayerNorm)
        self.feat_dropout = nn.FeatureAlphaDropout(dropout_p)

        # Global average pooling to produce a compact per-channel descriptor
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Final linear projection from hidden_channels -> out_features
        self.fc = nn.Linear(hidden_channels, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward computation.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C_in, H, W)

        Returns:
            torch.Tensor: Output tensor of shape (B, out_features)
        """
        # 1) Reflection pad
        x = self.ref_pad(x)  # shape: (B, C_in, H+2*pad, W+2*pad)

        # 2) Convolution + activation
        x = self.conv(x)     # shape: (B, hidden_channels, H', W')
        x = self.act(x)

        # 3) LayerNorm expects the channels to be the last dimension; permute accordingly
        #    From (B, C, H, W) -> (B, H, W, C)
        x = x.permute(0, 2, 3, 1)

        # 4) Layer normalization across channel dimension at each spatial location
        x = self.layer_norm(x)

        # 5) Apply FeatureAlphaDropout (keeps same shape)
        x = self.feat_dropout(x)

        # 6) Permute back to (B, C, H, W)
        x = x.permute(0, 3, 1, 2)

        # 7) Global average pooling -> (B, hidden_channels, 1, 1)
        x = self.global_pool(x)

        # 8) Flatten channels and apply final linear layer -> (B, hidden_channels)
        x = torch.flatten(x, 1)

        # 9) Final projection
        out = self.fc(x)  # (B, out_features)

        return out

# Module-level configuration variables (example sizes)
batch_size = 8
in_channels = 3
height = 128
width = 128
hidden_channels = 64
out_features = 100
pad = 1
dropout_p = 0.08

def get_inputs():
    """
    Returns a list with a single input tensor for the model forward pass.

    Shape: (batch_size, in_channels, height, width)
    """
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]

def get_init_inputs():
    """
    Returns the initialization parameters for constructing the Model:
      [in_channels, hidden_channels, out_features, pad, dropout_p]
    """
    return [in_channels, hidden_channels, out_features, pad, dropout_p]