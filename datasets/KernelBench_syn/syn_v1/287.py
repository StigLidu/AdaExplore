import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Complex sequence-to-feature model that:
    - Encodes an input sequence with an LSTM (batch_first=True).
    - Reshapes the LSTM last-time-step output into a 4D tensor (N, C, H, W).
    - Applies LazyBatchNorm2d (lazy initialization of channel dimension).
    - Applies SELU nonlinearity.
    - Performs spatial global average pooling to produce a compact feature vector.
    - Projects to an output embedding via a fully connected layer with SELU activation.
    The design demonstrates mixing recurrent, normalization, and activation layers with
    tensor shape transformations.
    """
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        bidirectional: bool,
        H: int,
        W: int,
        output_dim: int,
        dropout: float = 0.0
    ):
        """
        Initializes the model.

        Args:
            input_size (int): Dimensionality of input features per time step.
            hidden_size (int): LSTM hidden size (must equal C * H * W).
            num_layers (int): Number of stacked LSTM layers.
            bidirectional (bool): Whether the LSTM is bidirectional (must be False to keep reshape consistent).
            H (int): Spatial height for reshaping LSTM hidden vector.
            W (int): Spatial width for reshaping LSTM hidden vector.
            output_dim (int): Output embedding dimensionality.
            dropout (float): Dropout probability (applied inside LSTM when num_layers > 1).
        """
        super(Model, self).__init__()

        if bidirectional:
            raise ValueError("This model assumes a unidirectional LSTM to keep hidden_size == C*H*W.")

        if hidden_size % (H * W) != 0:
            raise ValueError("hidden_size must be divisible by H * W to form (C, H, W).")

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.H = H
        self.W = W
        self.output_dim = output_dim

        # LSTM encoder (batch_first for easier handling)
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=False
        )

        # LazyBatchNorm2d will initialize num_features on first forward pass from the input tensor
        self.bn2d = nn.LazyBatchNorm2d()

        # SELU activation (self-normalizing activation)
        self.selu = nn.SELU()

        # Final projection: from pooled channels to output_dim
        # After pooling we'll have a vector of size C where C = hidden_size / (H*W)
        self.C = self.hidden_size // (self.H * self.W)
        self.fc = nn.Linear(self.C, self.output_dim)

        # Small residual projection from the raw LSTM last output to output_dim for richer gradients
        self.res_proj = nn.Linear(self.hidden_size, self.output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_size).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_dim).
        """
        # x -> (batch, seq_len, input_size)
        batch_size = x.size(0)

        # 1) LSTM encoding
        # out: (batch, seq_len, hidden_size)
        out, (hn, cn) = self.lstm(x)

        # 2) Take the last time-step's output as sequence summary
        last = out[:, -1, :]  # (batch, hidden_size)

        # 3) Reshape to 4D tensor for 2D batch norm: (batch, C, H, W)
        feat_4d = last.view(batch_size, self.C, self.H, self.W)

        # 4) Apply lazy BatchNorm2d (initializes num_features = C on first run)
        feat_bn = self.bn2d(feat_4d)

        # 5) Apply SELU nonlinearity
        feat_act = self.selu(feat_bn)

        # 6) Global average pooling over spatial dims -> (batch, C, 1, 1) then squeeze to (batch, C)
        pooled = F.adaptive_avg_pool2d(feat_act, (1, 1)).view(batch_size, self.C)

        # 7) Project pooled features to output_dim
        out_main = self.fc(pooled)  # (batch, output_dim)
        out_main = self.selu(out_main)

        # 8) Residual projection from raw last LSTM vector
        out_res = self.res_proj(last)  # (batch, output_dim)

        # 9) Combine main and residual pathways
        output = out_main + out_res
        return output

# Configuration / default values
batch_size = 12
seq_len = 20
input_size = 64
hidden_size = 512  # must equal C * H * W
num_layers = 2
bidirectional = False
H = 4
W = 4
output_dim = 128
dropout = 0.1

def get_inputs():
    """
    Returns a list containing the input tensor for the model:
    - x: shape (batch_size, seq_len, input_size)
    """
    x = torch.randn(batch_size, seq_len, input_size)
    return [x]

def get_init_inputs():
    """
    Returns the initialization arguments for the Model constructor in order:
    [input_size, hidden_size, num_layers, bidirectional, H, W, output_dim, dropout]
    """
    return [input_size, hidden_size, num_layers, bidirectional, H, W, output_dim, dropout]