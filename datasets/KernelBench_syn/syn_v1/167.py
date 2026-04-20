import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Sequence encoder that runs a multi-layer (optionally bidirectional) LSTM,
    applies GELU to the sequence outputs, computes a dot-product attention
    between the final layer hidden state and every timestep, gates the attention
    scores with LogSigmoid, and produces a compact normalized representation
    by weighted-summing the activated sequence outputs and fusing with the
    final hidden state.
    """
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 2, bidirectional: bool = True):
        """
        Initializes the Model.

        Args:
            input_size (int): Number of expected features in the input.
            hidden_size (int): Hidden state size for the LSTM (per direction).
            num_layers (int, optional): Number of recurrent layers. Default: 2.
            bidirectional (bool, optional): If True, use a bidirectional LSTM. Default: True.
        """
        super(Model, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        # Core recurrent encoder
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=self.bidirectional
        )

        # Non-linearities used in the computation graph
        self.gelu = nn.GELU()
        self.logsigmoid = nn.LogSigmoid()

    def forward(self, x: torch.Tensor, h0: torch.Tensor = None, c0: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, seq_len, input_size).
            h0 (torch.Tensor, optional): Initial hidden state of shape
                (num_layers * num_directions, batch, hidden_size). If None, zeros are used.
            c0 (torch.Tensor, optional): Initial cell state of same shape as h0. If None, zeros are used.

        Returns:
            torch.Tensor: Normalized fused representation of shape (batch, hidden_size * num_directions).
        """
        batch = x.size(0)

        # Prepare initial states if not provided
        if h0 is None or c0 is None:
            h0 = x.new_zeros(self.num_layers * self.num_directions, batch, self.hidden_size)
            c0 = x.new_zeros(self.num_layers * self.num_directions, batch, self.hidden_size)

        # 1) LSTM encode the sequence -> outputs: (batch, seq, hidden*dirs)
        outputs, (hn, cn) = self.lstm(x, (h0, c0))

        # 2) Non-linearity on per-timestep outputs
        activated = self.gelu(outputs)  # shape: (batch, seq, hidden * num_directions)

        # 3) Derive a single query vector from the final layer hidden state.
        #    If bidirectional, concatenate the last forward and backward states for the top layer.
        if self.bidirectional:
            # hn layout: (num_layers * 2, batch, hidden); top-layer forward is -2, backward is -1
            final_hidden = torch.cat((hn[-2], hn[-1]), dim=-1)  # shape: (batch, hidden * 2)
        else:
            final_hidden = hn[-1]  # shape: (batch, hidden)

        # 4) Compute attention-like scores via dot product between query and every timestep
        #    scores shape: (batch, seq)
        scores = torch.einsum('bsd,bd->bs', activated, final_hidden)

        # 5) Gate the scores with LogSigmoid to introduce a multiplicative attenuation,
        #    then exponentiate to get positive weights (equivalent to sigmoid(scores))
        gated = self.logsigmoid(scores)    # shape: (batch, seq)
        weights = torch.exp(gated)         # shape: (batch, seq) approx sigmoid(scores)

        # 6) Weighted sum of activated outputs to produce a context vector
        weighted = activated * weights.unsqueeze(-1)  # broadcast over features: (batch, seq, d)
        context = weighted.sum(dim=1)                  # shape: (batch, d)

        # 7) Fuse context with final_hidden and apply a GELU non-linearity
        fused = self.gelu(context * final_hidden)      # element-wise fusion

        # 8) Normalize each example to unit norm to produce a stable representation
        norm = torch.norm(fused, p=2, dim=1, keepdim=True).clamp(min=1e-6)
        out = fused / norm                              # shape: (batch, d)

        return out

# Configuration / hyperparameters
batch_size = 12
seq_len = 128
input_size = 64
hidden_size = 128
num_layers = 3
bidirectional = True

def get_inputs():
    """
    Returns:
        List containing:
          - x: tensor (batch, seq_len, input_size)
          - h0: initial hidden state (num_layers * num_directions, batch, hidden_size)
          - c0: initial cell state (num_layers * num_directions, batch, hidden_size)
    """
    num_directions = 2 if bidirectional else 1
    x = torch.randn(batch_size, seq_len, input_size)
    h0 = torch.randn(num_layers * num_directions, batch_size, hidden_size)
    c0 = torch.randn(num_layers * num_directions, batch_size, hidden_size)
    return [x, h0, c0]

def get_init_inputs():
    """
    Returns initialization parameters that describe the model configuration.
    """
    return [input_size, hidden_size, num_layers, bidirectional]