import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Complex 3D feature processing block that combines Instance Normalization,
    SELU activation, channel-wise learned gating, and Feature Alpha Dropout,
    with a scaled residual connection.

    Computation pipeline:
      1. InstanceNorm3d normalization across spatial dims per-instance.
      2. SELU activation for self-normalizing non-linearity.
      3. Compute per-channel spatial means and apply a learned channel-wise
         affine transform followed by a sigmoid to produce a gating map.
      4. Multiply the activated features by the gating map (per-sample, per-channel).
      5. Apply FeatureAlphaDropout to mask entire channels probabilistically.
      6. Add a scaled residual connection from the input, with a learnable
         per-channel scaling factor.

    This module is designed to operate on 5D tensors with shape
    (batch, channels, depth, height, width).
    """
    def __init__(self, channels: int, dropout_p: float = 0.1, inst_eps: float = 1e-5):
        """
        Initializes the block.

        Args:
            channels (int): Number of channels in the input tensor.
            dropout_p (float): Dropout probability for FeatureAlphaDropout.
            inst_eps (float): Epsilon value for InstanceNorm3d for numerical stability.
        """
        super(Model, self).__init__()
        self.channels = channels
        # Instance normalization without affine: we'll apply our own channel transforms.
        self.instnorm = nn.InstanceNorm3d(num_features=channels, eps=inst_eps, affine=False, track_running_stats=False)
        # SELU activation for self-normalizing behavior
        self.selu = nn.SELU(inplace=False)
        # FeatureAlphaDropout works well with SELU
        self.dropout = nn.FeatureAlphaDropout(p=dropout_p)
        # Learned parameters for channel-wise gating: scale and bias (applied to spatial mean)
        # and a residual rescaling factor for the input skip connection.
        self.gate_scale = nn.Parameter(torch.ones(channels))
        self.gate_bias = nn.Parameter(torch.zeros(channels))
        self.rescale = nn.Parameter(torch.ones(channels) * 0.5)  # start with a moderate residual weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, D, H, W).

        Returns:
            torch.Tensor: Output tensor of the same shape as input.
        """
        # 1) Instance normalization per-instance
        y = self.instnorm(x)

        # 2) Non-linear activation
        y = self.selu(y)

        # 3) Compute per-sample, per-channel spatial means -> shape (B, C, 1, 1, 1)
        spatial_mean = y.mean(dim=(2, 3, 4), keepdim=True)

        # 4) Channel-wise affine transform on the means, then sigmoid to obtain gating [0,1]
        # Reshape learned parameters to broadcast over batch and spatial dims
        scale = self.gate_scale.view(1, -1, 1, 1, 1)
        bias = self.gate_bias.view(1, -1, 1, 1, 1)
        gate = torch.sigmoid(scale * spatial_mean + bias)

        # 5) Apply gating (modulates features per-sample and per-channel)
        y = y * gate

        # 6) Apply Feature Alpha Dropout (channel-wise dropout compatible with SELU)
        y = self.dropout(y)

        # 7) Add scaled residual connection (input scaled per-channel)
        rescale = self.rescale.view(1, -1, 1, 1, 1)
        out = y + x * rescale

        return out

# Configuration variables for input generation
batch_size = 4
channels = 32
depth = 8
height = 16
width = 16
dropout_p = 0.12
inst_eps = 1e-5

def get_inputs():
    """
    Returns a list containing a single input tensor matching the expected shape:
    (batch_size, channels, depth, height, width).
    """
    x = torch.randn(batch_size, channels, depth, height, width)
    return [x]

def get_init_inputs():
    """
    Returns a list of initialization parameters for the Model constructor:
    [channels, dropout_p, inst_eps]
    """
    return [channels, dropout_p, inst_eps]