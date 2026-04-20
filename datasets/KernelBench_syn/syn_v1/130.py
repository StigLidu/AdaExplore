import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    1D upsampling and gating module.

    This model performs the following high-level steps:
    - Uses a LazyConvTranspose1d to upsample the input sequence by a factor (stride=2).
      The lazy transpose conv infers its in_channels from the input tensor.
    - Applies SyncBatchNorm across the channel dimension for stabilization.
    - Applies a channel-wise PReLU nonlinearity.
    - Computes a lightweight channel attention (global pooling + two linear layers) which
      produces channel-wise gates that modulate the activations in a residual manner.

    The combination results in an upsampling block with learned channel-wise gating.
    """
    def __init__(self, out_channels: int, attention_hidden: int = None):
        """
        Args:
            out_channels: Number of output channels produced by the transposed conv.
            attention_hidden: Hidden dimensionality for the channel attention MLP.
                              If None, defaults to out_channels // 2.
        """
        super(Model, self).__init__()
        self.out_channels = out_channels
        if attention_hidden is None:
            attention_hidden = max(1, out_channels // 2)

        # LazyConvTranspose1d: in_channels will be inferred on first forward pass.
        # We provide out_channels explicitly via keyword to avoid confusion.
        self.deconv = nn.LazyConvTranspose1d(out_channels=out_channels,
                                             kernel_size=4,
                                             stride=2,
                                             padding=1,
                                             bias=True)

        # SyncBatchNorm operates per-channel. In non-distributed mode this behaves
        # like a normal BatchNorm but keeps the same API.
        self.bn = nn.SyncBatchNorm(num_features=out_channels)

        # Channel-wise parametric ReLU
        self.prelu = nn.PReLU(num_parameters=out_channels)

        # Small MLP for channel attention: reduces channels, nonlinearity, then restores
        self.att_fc1 = nn.Linear(out_channels, attention_hidden, bias=True)
        self.att_fc2 = nn.Linear(attention_hidden, out_channels, bias=True)
        self.att_sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, in_channels, length).

        Returns:
            Tensor of shape (batch_size, out_channels, length * 2).
        """
        # 1) Upsample via transposed convolution (in_channels inferred lazily)
        y = self.deconv(x)                       # (B, out_channels, L_up)

        # 2) Normalize across batch & channels
        y = self.bn(y)                           # (B, out_channels, L_up)

        # 3) Channel-wise parametric activation
        y = self.prelu(y)                       # (B, out_channels, L_up)

        # 4) Channel attention: global pooling -> MLP -> sigmoid gates
        # Global average pooling over the temporal dimension
        z = y.mean(dim=2)                       # (B, out_channels)

        # 5) Bottleneck MLP with non-linearity
        z = F.relu(self.att_fc1(z))             # (B, attention_hidden)
        z = self.att_sigmoid(self.att_fc2(z))   # (B, out_channels)

        # 6) Rescale activations channel-wise and add residual-style skip
        z = z.unsqueeze(2)                      # (B, out_channels, 1)
        out = y * z + y                         # (B, out_channels, L_up)

        return out

# Configuration variables
BATCH_SIZE = 8
IN_CHANNELS = 12
INPUT_LENGTH = 64
OUT_CHANNELS = 64  # number of channels produced by the transposed convolution

def get_inputs():
    """
    Returns inputs for the model:
    - A single 1D tensor of shape (BATCH_SIZE, IN_CHANNELS, INPUT_LENGTH)
    """
    x = torch.randn(BATCH_SIZE, IN_CHANNELS, INPUT_LENGTH)
    return [x]

def get_init_inputs():
    """
    Returns initialization parameters required to construct the Model.
    The testing harness can use these to instantiate the model consistently.
    """
    return [OUT_CHANNELS]