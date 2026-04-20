import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Complex 1D processing module that demonstrates a small CNN-like pipeline
    using circular padding, convolution, elementwise nonlinearity transforms,
    and gating via hard-sigmoid. The dataflow:

        Input (N, C_in, L) ->
        CircularPad1d ->
        Conv1d (temporal receptive field) ->
        Square nonlinearity (energy) ->
        RReLU (randomized leaky ReLU) ->
        Pointwise Conv1d (1x1) ->
        Hardsigmoid gating ->
        Global average pooling over temporal dim ->
        Output (N, C_out)

    This combines nn.CircularPad1d, nn.RReLU, and nn.Hardsigmoid from the
    provided layer list, along with standard convolutional layers.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        rrelu_lower: float = 0.125,
        rrelu_upper: float = 0.333,
    ):
        """
        Initializes the model.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels after the first conv.
            kernel_size: Kernel size for the temporal convolution.
            rrelu_lower: Lower bound for the randomized negative slope in RReLU.
            rrelu_upper: Upper bound for the randomized negative slope in RReLU.
        """
        super(Model, self).__init__()
        # Amount of circular padding on each side to preserve length
        pad_amount = kernel_size // 2

        # Layers
        self.pad = nn.CircularPad1d(pad_amount)
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, bias=True)
        # RReLU introduces a randomized negative slope during training
        self.rrelu = nn.RReLU(lower=rrelu_lower, upper=rrelu_upper, inplace=False)
        # Pointwise convolution acts as a channel-wise linear mixing (1x1 conv)
        self.pointwise = nn.Conv1d(out_channels, out_channels, kernel_size=1, bias=True)
        # Hard sigmoid to produce gating in range [0,1]
        self.hardsigmoid = nn.Hardsigmoid()

        # Small weight initialization to encourage stable activations
        nn.init.kaiming_uniform_(self.conv.weight, a=0.0)
        if self.conv.bias is not None:
            nn.init.constant_(self.conv.bias, 0.0)
        nn.init.kaiming_uniform_(self.pointwise.weight, a=0.0)
        if self.pointwise.bias is not None:
            nn.init.constant_(self.pointwise.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, in_channels, length).

        Returns:
            Tensor of shape (batch_size, out_channels) after temporal pooling.
        """
        # 1) Circular padding to give conv a wrap-around receptive field
        x = self.pad(x)  # (N, C_in, L + 2*pad) but pad is symmetric

        # 2) Temporal convolution -> (N, out_channels, L)
        x = self.conv(x)

        # 3) Energy-style transform (element-wise square)
        x = x * x

        # 4) Randomized leaky rectified linear unit
        x = self.rrelu(x)

        # 5) Pointwise linear mixing across channels
        x = self.pointwise(x)

        # 6) Hardsigmoid gating (elementwise) to squash into [0,1]
        x = self.hardsigmoid(x)

        # 7) Global average pooling over temporal dimension to produce (N, C_out)
        out = x.mean(dim=2)

        return out

# Configuration for generating inputs
batch_size = 32
in_channels = 16
out_channels = 32
length = 1024
kernel_size = 15

def get_inputs():
    """
    Returns example input tensors for the model's forward pass.
    """
    # Typical float32 inputs for convolutional processing
    x = torch.randn(batch_size, in_channels, length, dtype=torch.float32)
    return [x]

def get_init_inputs():
    """
    Returns initialization parameters for the Model constructor.
    """
    return [in_channels, out_channels, kernel_size]