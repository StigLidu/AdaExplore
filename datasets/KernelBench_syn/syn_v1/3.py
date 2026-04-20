import torch
import torch.nn as nn

# Configuration variables
batch_size = 8
in_channels = 12
height = 16
width = 16
rnn_input_size = 32
hidden_size = 64
out_channels = 10  # number of output channels after sequence processing

class Model(nn.Module):
    """
    A composite model that:
    - Applies BatchNorm2d to a 4D input (B, C, H, W)
    - Converts spatial locations (H*W) into a sequence
    - Maps per-location channels into an RNN input space via a Linear layer
    - Runs an nn.RNNCell across the spatial sequence (treating flattened H*W as time steps)
    - Uses Tanhshrink non-linearity between Linear -> RNNCell
    - Projects RNN hidden states back to output channels and reshapes to (B, out_channels, H, W)
    """
    def __init__(self,
                 num_channels: int,
                 rnn_in_size: int,
                 rnn_hidden: int,
                 output_channels: int):
        super(Model, self).__init__()
        # Normalize input feature maps per-channel
        self.bn = nn.BatchNorm2d(num_features=num_channels)

        # Map per-spatial-location channels -> rnn input dimension
        self.input_proj = nn.Linear(num_channels, rnn_in_size)

        # RNN cell processes one time-step (spatial location) at a time
        self.rnn_cell = nn.RNNCell(input_size=rnn_in_size, hidden_size=rnn_hidden, nonlinearity='tanh')

        # Non-linearity between projection and RNN cell
        self.tanhshrink = nn.Tanhshrink()

        # Project hidden state back to desired output channels per spatial location
        self.hidden_proj = nn.Linear(rnn_hidden, output_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (B, C, H, W)

        Returns:
            out: Tensor of shape (B, out_channels, H, W)
        """
        B, C, H, W = x.shape
        # 1) Batch normalization
        x = self.bn(x)  # (B, C, H, W)

        # 2) Prepare a sequence across spatial dimensions: (B, L, C) where L = H*W
        # Permute to (B, H, W, C) then flatten spatial dims
        seq = x.permute(0, 2, 3, 1).contiguous().view(B, H * W, C)  # (B, L, C)
        L = H * W

        # 3) Iterate through sequence with RNNCell. Use a buffer to store per-step hidden states.
        device = seq.device
        dtype = seq.dtype
        h = torch.zeros(B, self.rnn_cell.hidden_size, device=device, dtype=dtype)
        outputs = torch.zeros(B, L, self.rnn_cell.hidden_size, device=device, dtype=dtype)

        # Process each spatial location as a time-step
        for t in range(L):
            # linear projection from channels -> rnn_input_size
            inp = self.input_proj(seq[:, t, :])  # (B, rnn_input_size)

            # non-linearity
            inp = self.tanhshrink(inp)

            # RNN cell update
            h = self.rnn_cell(inp, h)  # (B, hidden_size)

            # store hidden state
            outputs[:, t, :] = h

        # 4) Project hidden states back to output channels per spatial location
        out_seq = self.hidden_proj(outputs)  # (B, L, out_channels)

        # 5) Reshape back to (B, out_channels, H, W)
        out = out_seq.view(B, H, W, -1).permute(0, 3, 1, 2).contiguous()  # (B, out_channels, H, W)
        return out

def get_inputs():
    """
    Returns a list of input tensors for the model.
    Single tensor shaped (batch_size, in_channels, height, width).
    """
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]

def get_init_inputs():
    """
    Returns initialization parameters for Model:
    [num_channels, rnn_input_size, hidden_size, out_channels]
    """
    return [in_channels, rnn_input_size, hidden_size, out_channels]