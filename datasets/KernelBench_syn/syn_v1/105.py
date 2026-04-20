import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Volumetric feature transformer that demonstrates a combination of:
     - Reflection padding in 3D (nn.ReflectionPad3d)
     - Channel-wise linear projections using nn.LazyLinear applied per-voxel
     - Channel-wise dropout specialized for features (nn.FeatureAlphaDropout)
    The module pads the input volume, projects per-voxel channel features into a higher-dimensional
    hidden space, applies non-linearity and feature dropout, projects back to a target channel
    dimensionality, and finally adds a projected residual (skip connection).
    """
    def __init__(self, out_channels: int, dropout_p: float = 0.1, expansion: int = 2, pad: int = 1):
        """
        Args:
            out_channels (int): Target number of output channels after projection.
            dropout_p (float): Probability for FeatureAlphaDropout.
            expansion (int): Expansion factor for the hidden linear layer (hidden = in_channels * expansion).
            pad (int): Amount of symmetric reflection padding applied to D,H,W dimensions.
        """
        super(Model, self).__init__()
        self.out_channels = out_channels
        self.expansion = expansion
        self.pad_amount = pad

        # Reflection padding for volumetric (D, H, W) data.
        # padding argument for ReflectionPad3d is (pad_L, pad_R, pad_T, pad_B, pad_F, pad_Back)
        self.pad = nn.ReflectionPad3d((pad, pad, pad, pad, pad, pad))

        # LazyLinear layers infer in_features on first forward pass.
        # lin1 maps per-voxel channel vector -> expanded hidden representation
        self.lin1 = nn.LazyLinear(out_features=out_channels * expansion)
        # lin2 maps hidden back to desired out_channels
        self.lin2 = nn.LazyLinear(out_features=out_channels)
        # Residual projection: if input channels != out_channels, this will map them appropriately.
        self.res_proj = nn.LazyLinear(out_features=out_channels)

        # Activation and dropout specialized for feature maps
        self.act = nn.SELU()
        self.dropout = nn.FeatureAlphaDropout(p=dropout_p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape (N, C, D, H, W)

        Returns:
            torch.Tensor: Output tensor of shape (N, out_channels, D, H, W)
        """
        if x.dim() != 5:
            raise ValueError(f"Expected input tensor of shape (N, C, D, H, W), got {tuple(x.shape)}")

        N, C_in, D_in, H_in, W_in = x.shape

        # 1) Reflection pad spatial dims -> increases D, H, W
        x_padded = self.pad(x)  # shape (N, C_in, Dp, Hp, Wp)
        _, _, Dp, Hp, Wp = x_padded.shape

        # 2) Prepare per-voxel channel vectors for linear layers.
        #    Permute to (N, Dp, Hp, Wp, C) then flatten leading dims -> (N*Dp*Hp*Wp, C)
        x_flat = x_padded.permute(0, 2, 3, 4, 1).contiguous().view(-1, C_in)  # shape (N*Dp*Hp*Wp, C_in)

        # 3) First linear projection to expanded hidden space
        hidden = self.lin1(x_flat)  # shape (N*Dp*Hp*Wp, out_channels*expansion)
        hidden = self.act(hidden)
        hidden = self.dropout(hidden)

        # 4) Project back to target out_channels
        out_flat = self.lin2(hidden)  # shape (N*Dp*Hp*Wp, out_channels)

        # 5) Reshape back to volumetric format (N, out_channels, Dp, Hp, Wp)
        out_vol = out_flat.view(N, Dp, Hp, Wp, self.out_channels).permute(0, 4, 1, 2, 3).contiguous()

        # 6) Crop padded borders to restore original spatial dimensions (D_in, H_in, W_in)
        p = self.pad_amount
        # If pad is zero, slicing will be : which is fine.
        out_cropped = out_vol[:, :, p: p + D_in, p: p + H_in, p: p + W_in]  # shape (N, out_channels, D_in, H_in, W_in)

        # 7) Create residual projection of the original (unpadded) input if needed and add.
        #    Apply res_proj per-voxel similar to above.
        res_flat = x.permute(0, 2, 3, 4, 1).contiguous().view(-1, C_in)  # shape (N*D_in*H_in*W_in, C_in)
        res_proj_flat = self.res_proj(res_flat)  # shape (N*D_in*H_in*W_in, out_channels)
        res_proj = res_proj_flat.view(N, D_in, H_in, W_in, self.out_channels).permute(0, 4, 1, 2, 3).contiguous()

        # 8) Final combine and activation
        out = out_cropped + res_proj
        out = self.act(out)

        return out

# Configuration / default sizes
batch_size = 2
in_channels = 8
depth = 10
height = 12
width = 12

# Default initialization arguments for the model
default_out_channels = 12
default_dropout_p = 0.15
default_expansion = 2
default_pad = 1

def get_inputs():
    """
    Returns one volumetric input tensor shaped (N, C, D, H, W).
    """
    x = torch.randn(batch_size, in_channels, depth, height, width)
    return [x]

def get_init_inputs():
    """
    Returns initialization parameters to construct the Model:
      - out_channels
      - dropout probability
      - expansion factor
      - padding amount
    """
    return [default_out_channels, default_dropout_p, default_expansion, default_pad]