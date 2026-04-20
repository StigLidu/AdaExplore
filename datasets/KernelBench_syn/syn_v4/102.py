import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Complex 1D patch-aggregation model that:
    - Pads the input with replication padding
    - Extracts sliding local blocks using nn.Unfold (treating 1D as 2D with height=1)
    - Applies LogSigmoid non-linearity to each extracted patch element
    - Performs a learned weighted aggregation across the patch dimension per channel
    - Adds a per-channel bias and final non-linearity

    The module is intended for inputs of shape (batch_size, in_channels, length).
    The output has shape (batch_size, in_channels, output_length) where
    output_length depends on padding, kernel_size and stride.
    """
    def __init__(self, in_channels: int, kernel_size: int, stride: int = 1, padding: int = 0):
        """
        Initializes the model components.

        Args:
            in_channels (int): Number of channels in the input tensor.
            kernel_size (int): Size of the 1D sliding window (patch length).
            stride (int, optional): Stride for sliding windows. Defaults to 1.
            padding (int, optional): Replication padding applied to left/right. Defaults to 0.
        """
        super(Model, self).__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # ReplicationPad1d works on (N, C, L)
        self.pad = nn.ReplicationPad1d(padding)

        # nn.Unfold expects 4D input (N, C, H, W). For 1D signal, treat H=1 and W=length.
        # Kernel and stride are (1, kernel_size) and (1, stride) respectively.
        self.unfold = nn.Unfold(kernel_size=(1, kernel_size), stride=(1, stride))

        # Elementwise nonlinearity applied to each patch element
        self.logsigmoid = nn.LogSigmoid()

        # Learned weights to aggregate across the kernel dimension for each channel
        # Shape: (in_channels, kernel_size)
        self.register_parameter(
            "agg_weights",
            nn.Parameter(torch.randn(in_channels, kernel_size) * (1.0 / (kernel_size ** 0.5)))
        )

        # Per-channel bias after aggregation
        self.register_parameter(
            "bias",
            nn.Parameter(torch.zeros(in_channels))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward computation.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, length).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, in_channels, output_length).
        """
        # Step 1: replication pad
        x_padded = self.pad(x)  # (N, C, L_padded)

        # Step 2: convert to 4D (N, C, 1, L_padded) for nn.Unfold and extract patches
        x4 = x_padded.unsqueeze(2)  # (N, C, 1, L_padded)
        patches = self.unfold(x4)  # (N, C * kernel_size, L_out)

        N = patches.size(0)
        L_out = patches.size(-1)
        # Reshape to (N, C, kernel_size, L_out)
        patches = patches.view(N, self.in_channels, self.kernel_size, L_out)

        # Step 3: apply elementwise LogSigmoid to each patch element
        patches = self.logsigmoid(patches)  # (N, C, kernel_size, L_out)

        # Step 4: weighted aggregation across kernel dimension using learned weights
        # agg_weights: (C, kernel_size) -> (1, C, kernel_size, 1) for broadcasting
        weights = self.agg_weights.view(1, self.in_channels, self.kernel_size, 1)
        aggregated = (patches * weights).sum(dim=2)  # (N, C, L_out)

        # Step 5: add bias per channel and apply final non-linearity (tanh)
        aggregated = aggregated + self.bias.view(1, self.in_channels, 1)
        output = torch.tanh(aggregated)  # (N, C, L_out)

        return output

# Configuration / example initialization values
batch_size = 8
in_channels = 16
input_length = 128
kernel_size = 5
stride = 2
padding = 2  # replication padding on both sides

def get_inputs():
    """
    Returns example runtime inputs for the model.
    """
    x = torch.randn(batch_size, in_channels, input_length)
    return [x]

def get_init_inputs():
    """
    Returns the initialization parameters for the Model constructor:
    [in_channels, kernel_size, stride, padding]
    """
    return [in_channels, kernel_size, stride, padding]