import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Complex example combining ReflectionPad2d, GLU gating, and LazyBatchNorm1d.
    The model:
      - Pads the input spatially using reflection padding.
      - Creates a paired channel tensor by rolling channels and concatenating.
      - Applies a GLU over the channel dimension to produce gated feature maps.
      - Performs global average pooling per channel and applies LazyBatchNorm1d
        (which is lazily initialized on first forward).
      - Uses the normalized pooled features to produce a sigmoid scale per channel,
        reweights the gated feature maps, and adds a simple residual/statistics term.
      - Returns a flattened vector with a final GELU nonlinearity.
    Input shape: (batch_size, in_channels, height, width)
    Output shape: (batch_size, in_channels * padded_height * padded_width)
    """
    def __init__(self, pad: int = 2):
        """
        Args:
            pad (int): Reflection padding size applied to all 4 sides.
        """
        super(Model, self).__init__()
        # reflection padding layer
        self.pad = nn.ReflectionPad2d(pad)
        # GLU that will split along the channel dimension (dim=1)
        self.glu = nn.GLU(dim=1)
        # Lazy BatchNorm1d: will be initialized on first forward when input size is known
        self.bn = nn.LazyBatchNorm1d()
        # small learnable scale to modulate the gating output before final flatten
        self.post_scale = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward computation.

        Steps:
          1) Reflection pad input -> pad_x
          2) Create a paired channel tensor by rolling pad_x channels -> pad_x_roll
          3) Concatenate pad_x and pad_x_roll along channel dim -> concat (2*C channels)
          4) Apply GLU along channel dim -> gated (C channels)
          5) Global average pool gated -> pooled (N, C)
          6) Apply LazyBatchNorm1d to pooled -> normed (N, C)
          7) Sigmoid(normed) to obtain channel-wise scale, unsqueeze to (N,C,1,1)
          8) Reweight gated by scale, add a simple residual channel-statistics term
          9) Multiply by learnable scalar and apply GELU, then flatten
        """
        # 1) pad
        pad_x = self.pad(x)  # shape: (N, C, H+2*pad, W+2*pad)

        # 2) roll channels to create a paired signal (permutes channel content)
        pad_x_roll = torch.roll(pad_x, shifts=1, dims=1)

        # 3) concatenate along channel dimension -> 2*C channels
        concat = torch.cat([pad_x, pad_x_roll], dim=1)

        # 4) GLU splits the channel dim in half and applies gating -> back to C channels
        gated = self.glu(concat)  # shape: (N, C, H', W')

        # 5) global average pooling per channel
        pooled = gated.mean(dim=[2, 3])  # shape: (N, C)

        # 6) Lazy BatchNorm1d will initialize num_features=C on first forward
        normed = self.bn(pooled)  # shape: (N, C)

        # 7) channel-wise scale from normalized pooled features
        scale = torch.sigmoid(normed).unsqueeze(-1).unsqueeze(-1)  # shape: (N, C, 1, 1)

        # 8) reweight gated feature maps and add a residual statistic (mean of padded input)
        channel_mean = pad_x.mean(dim=[2, 3], keepdim=True)  # (N, C, 1, 1)
        out = gated * scale + channel_mean

        # 9) final modulation and flatten
        out = out * self.post_scale
        out = F.gelu(out)
        out = out.flatten(start_dim=1)
        return out

# Configuration variables for test inputs
batch_size = 8
in_channels = 3
height = 32
width = 32
pad = 2  # default padding used in Model init

def get_inputs():
    """
    Returns a list with a single input tensor matching (batch_size, in_channels, height, width).
    """
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]

def get_init_inputs():
    """
    Returns initialization parameters for the Model constructor.
    """
    return [pad]