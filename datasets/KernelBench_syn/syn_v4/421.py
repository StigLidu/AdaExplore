import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Sequence encoder that:
      - Zero-pads the temporal dimension using nn.ZeroPad1d
      - Processes the padded sequence step-by-step with an nn.RNNCell
      - Treats the sequence of hidden states as a volumetric channel input and applies nn.Dropout3d
      - Aggregates over time and projects to an output vector

    This creates a hybrid of padding, recurrent, and volumetric dropout operations.
    """
    def __init__(
        self,
        input_size: int,
        pad_left: int,
        pad_right: int,
        hidden_size: int,
        nonlinearity: str,
        dropout_p: float,
        output_size: int,
    ):
        """
        Args:
            input_size (int): Dimensionality of input features per time step.
            pad_left (int): Amount of zero padding to add on the left of the temporal axis.
            pad_right (int): Amount of zero padding to add on the right of the temporal axis.
            hidden_size (int): Size of the hidden state in the RNNCell.
            nonlinearity (str): Non-linearity for the RNNCell ('tanh' or 'relu').
            dropout_p (float): Dropout probability for Dropout3d.
            output_size (int): Dimensionality of the final output vector.
        """
        super(Model, self).__init__()
        # Pad along the temporal length dimension (last dim when input is (N, C, L))
        self.pad = nn.ZeroPad1d((pad_left, pad_right))
        # Single RNN cell that will be iterated over time
        self.rnncell = nn.RNNCell(input_size=input_size, hidden_size=hidden_size, nonlinearity=nonlinearity)
        # Volumetric dropout that will zero out entire channels (treat hidden as channels)
        self.dropout3d = nn.Dropout3d(p=dropout_p)
        # Final projection from aggregated hidden to desired output size
        self.fc = nn.Linear(hidden_size, output_size)
        # Save configuration for potential inspection
        self.input_size = input_size
        self.pad_left = pad_left
        self.pad_right = pad_right
        self.hidden_size = hidden_size
        self.nonlinearity = nonlinearity
        self.dropout_p = dropout_p
        self.output_size = output_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_size).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_size).
        """
        # x: (N, T, F)
        if x.ndim != 3:
            raise ValueError(f"Expected input of shape (batch, seq_len, input_size), got {x.shape}")

        batch_size, seq_len, feature_dim = x.shape
        if feature_dim != self.input_size:
            raise ValueError(f"Input feature dim ({feature_dim}) does not match model input_size ({self.input_size})")

        # Prepare for ZeroPad1d which pads the last (temporal) dimension when input is (N, C, L)
        x_for_pad = x.permute(0, 2, 1)  # (N, F, T)
        x_padded = self.pad(x_for_pad)  # (N, F, T_padded)
        x_padded = x_padded.permute(0, 2, 1)  # (N, T_padded, F)
        T_padded = x_padded.size(1)

        # Initialize hidden state (zeros), matching dtype and device of input
        h = x.new_zeros(batch_size, self.hidden_size)

        # Process sequence step-by-step through the RNNCell
        outputs = []
        for t in range(T_padded):
            input_t = x_padded[:, t, :]  # (N, F)
            h = self.rnncell(input_t, h)  # (N, hidden_size)
            outputs.append(h.unsqueeze(1))  # keep time dim for stacking

        # Stack outputs -> (N, T_padded, hidden_size)
        outputs_seq = torch.cat(outputs, dim=1)
        # Rearrange to treat hidden_size as channels for Dropout3d: (N, C, D) -> expand to 5D
        outputs_ch_time = outputs_seq.permute(0, 2, 1)  # (N, hidden_size, T_padded)

        # Expand to 5D volumetric tensor expected by Dropout3d: (N, C, D, H, W)
        vol = outputs_ch_time.unsqueeze(-1).unsqueeze(-1)  # (N, C, T_padded, 1, 1)
        vol_dropped = self.dropout3d(vol)  # dropout across channels
        vol_dropped = vol_dropped.squeeze(-1).squeeze(-1)  # (N, C, T_padded)

        # Aggregate over time (temporal mean) -> (N, hidden_size)
        aggregated = vol_dropped.mean(dim=2)

        # Final projection and non-linearity
        projected = self.fc(aggregated)  # (N, output_size)
        output = torch.relu(projected)

        return output

# Module-level configuration variables
batch_size = 8
seq_len = 50
input_size = 32
pad_left = 2
pad_right = 3
hidden_size = 64
nonlinearity = "tanh"  # 'tanh' or 'relu'
dropout_p = 0.2
output_size = 16

def get_inputs():
    """
    Returns:
        list: [x] where x has shape (batch_size, seq_len, input_size)
    """
    x = torch.randn(batch_size, seq_len, input_size)
    return [x]

def get_init_inputs():
    """
    Returns initialization arguments for Model.__init__ in the same order:
      input_size, pad_left, pad_right, hidden_size, nonlinearity, dropout_p, output_size
    """
    return [input_size, pad_left, pad_right, hidden_size, nonlinearity, dropout_p, output_size]