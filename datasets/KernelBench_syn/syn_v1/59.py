import math
import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Complex model that:
    - Applies Fractional Max Pooling to reduce spatial dimensions.
    - Uses GELU non-linearity on pooled features.
    - Treats one spatial dimension as a time axis and processes it with an RNNCell across time steps.
    - Aggregates RNN outputs and modulates them with a global spatial average.

    This creates a hybrid spatial-temporal processing pattern combining pooling, activation,
    and recurrent cell operations.
    """
    def __init__(
        self,
        in_channels: int,
        height: int,
        width: int,
        output_ratio: tuple = (0.5, 0.5),
        kernel_size: int = 2,
        rnn_hidden_size: int = 128,
        return_indices: bool = False
    ):
        """
        Args:
            in_channels (int): Number of input channels.
            height (int): Input height (spatial).
            width (int): Input width (spatial).
            output_ratio (tuple): Fractional output ratios (height_ratio, width_ratio).
            kernel_size (int): Kernel size argument for FractionalMaxPool2d.
            rnn_hidden_size (int): Hidden size for the RNNCell.
            return_indices (bool): Whether FractionalMaxPool2d should return indices (unused here).
        """
        super(Model, self).__init__()

        # Fractional pooling reduces spatial dimensions approximately by output_ratio.
        self.fpool = nn.FractionalMaxPool2d(
            kernel_size=kernel_size,
            output_ratio=output_ratio,
            return_indices=return_indices
        )

        # Non-linear activation after pooling.
        self.gelu = nn.GELU()

        # Given we know input height/width at initialization, compute pooled spatial dims
        # for constructing the RNN input size.
        hout = max(1, int(math.floor(height * output_ratio[0])))
        wout = max(1, int(math.floor(width * output_ratio[1])))

        # RNN will run across the width dimension (timestep = wout). Each timestep
        # input size = channels * pooled_height.
        rnn_input_size = in_channels * hout
        self.rnn_cell = nn.RNNCell(input_size=rnn_input_size, hidden_size=rnn_hidden_size)

        # Save configuration for reference
        self._cfg = {
            "in_channels": in_channels,
            "height": height,
            "width": width,
            "hout": hout,
            "wout": wout,
            "rnn_input_size": rnn_input_size,
            "rnn_hidden_size": rnn_hidden_size
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward computation:
        1. Fractional max pooling => (B, C, Hout, Wout)
        2. GELU activation
        3. Rearrange to sequence across width: (B, Wout, C * Hout)
        4. Iterate RNNCell across time (width) dimension producing hidden states
        5. Aggregate RNN outputs (mean over time) and apply tanh
        6. Modulate by global spatial average (from pooled tensor)
        Returns:
            Tensor of shape (B, rnn_hidden_size)
        """
        # 1. Fractional pooling
        pooled = self.fpool(x)  # (B, C, Hout, Wout)

        # 2. Non-linearity
        activated = self.gelu(pooled)

        # 3. Prepare sequence across width: shape -> (B, Wout, C * Hout)
        B, C, Hout, Wout = activated.shape
        seq = activated.permute(0, 3, 1, 2).contiguous()  # (B, Wout, C, Hout)
        seq = seq.view(B, Wout, -1)  # (B, Wout, C * Hout)

        # 4. RNNCell loop across time (width)
        h = torch.zeros(B, self.rnn_cell.hidden_size, device=x.device, dtype=x.dtype)
        outputs = []
        for t in range(Wout):
            input_t = seq[:, t, :]  # (B, input_size)
            h = self.rnn_cell(input_t, h)  # (B, hidden_size)
            outputs.append(h.unsqueeze(1))
        out_seq = torch.cat(outputs, dim=1)  # (B, Wout, hidden_size)

        # 5. Temporal aggregation and non-linearity
        temporal_mean = out_seq.mean(dim=1)  # (B, hidden_size)
        activated_out = torch.tanh(temporal_mean)  # (B, hidden_size)

        # 6. Modulate by a scalar global spatial average from pooled features
        # pooled.mean over (channels, height, width) -> (B,)
        spatial_mean = pooled.mean(dim=(1, 2, 3), keepdim=True)  # (B, 1)
        # Broadcast multiply to modulate each hidden feature
        final = activated_out * spatial_mean  # (B, hidden_size)

        return final

# Module-level configuration variables
batch_size = 8
in_channels = 3
height = 64
width = 32
output_ratio = (0.6, 0.5)
kernel_size = 2
rnn_hidden_size = 128
return_indices = False

def get_inputs():
    """
    Returns a list containing the main input tensor with shape:
    (batch_size, in_channels, height, width)
    """
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]

def get_init_inputs():
    """
    Returns initialization arguments for the Model constructor in the order:
    [in_channels, height, width, output_ratio, kernel_size, rnn_hidden_size, return_indices]
    """
    return [in_channels, height, width, output_ratio, kernel_size, rnn_hidden_size, return_indices]