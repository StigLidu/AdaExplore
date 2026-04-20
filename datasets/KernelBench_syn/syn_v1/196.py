import torch
import torch.nn as nn

# Configuration
batch_size = 8
seq_len = 20
input_size = 64
rnn_hidden = 128
rnn_layers = 2
proj_dim = 256  # Dimension of final projection output

class Model(nn.Module):
    """
    Composite model that:
    - Runs a multi-layer Elman RNN over an input sequence (batch_first).
    - Reshapes the sequence outputs into a 4D tensor and applies LazyBatchNorm2d.
    - Applies SiLU activation, spatial pooling to get a summary feature.
    - Combines the pooled summary with the final RNN hidden state via a learned gating mechanism.
    - Projects the combined representation to a final output dimension.
    """
    def __init__(self, out_dim: int):
        """
        Initializes the model.

        Args:
            out_dim (int): Output projection dimension.
        """
        super(Model, self).__init__()
        # Recurrent layer
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=rnn_hidden,
            num_layers=rnn_layers,
            nonlinearity='tanh',
            batch_first=True,
            bias=True
        )

        # LazyBatchNorm2d: will infer num_features (channels) at first forward
        self.bn2d = nn.LazyBatchNorm2d()

        # SiLU activation
        self.silu = nn.SiLU()

        # Gating parameters (learnable per feature dimension)
        self.scale = nn.Parameter(torch.ones(rnn_hidden))
        self.bias = nn.Parameter(torch.zeros(rnn_hidden))

        # Final projection
        self.fc = nn.Linear(rnn_hidden, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input of shape (batch, seq_len, input_size).

        Returns:
            torch.Tensor: Output of shape (batch, out_dim).
        """
        # x -> RNN
        # rnn_out: (batch, seq_len, rnn_hidden)
        # h_n: (num_layers, batch, rnn_hidden)
        rnn_out, h_n = self.rnn(x)

        # Prepare a 4D tensor for BatchNorm2d:
        # permute to (batch, channels=rnn_hidden, height=seq_len, width=1)
        feat_map = rnn_out.permute(0, 2, 1).unsqueeze(-1)  # (batch, rnn_hidden, seq_len, 1)

        # Apply LazyBatchNorm2d (will initialize num_features on first run)
        normalized = self.bn2d(feat_map)

        # Non-linear activation
        activated = self.silu(normalized)

        # Spatial pooling to summarize sequence -> (batch, rnn_hidden)
        pooled = activated.mean(dim=(2, 3))  # average over seq_len and width

        # Extract last layer hidden state from RNN and reshape to (batch, rnn_hidden)
        last_hidden = h_n[-1]  # (batch, rnn_hidden)

        # Gating: combine pooled summary and last hidden with learned scale and bias
        # scale and bias shape: (rnn_hidden,)
        gate_input = pooled * self.scale + self.bias + last_hidden
        gate = torch.sigmoid(gate_input)  # (batch, rnn_hidden)

        # Combine representations using gate
        combined = pooled * gate + last_hidden * (1.0 - gate)  # (batch, rnn_hidden)

        # Final linear projection
        out = self.fc(combined)  # (batch, out_dim)
        return out

def get_inputs():
    """
    Returns example runtime inputs for the model.
    """
    x = torch.randn(batch_size, seq_len, input_size)
    return [x]

def get_init_inputs():
    """
    Returns initialization inputs required by Model.__init__.
    """
    return [proj_dim]