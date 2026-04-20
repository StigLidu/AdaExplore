import torch
import torch.nn as nn

# Configuration
batch_size = 8
seq_len = 512
dim = 1024
pad = 3  # number of elements to pad on each side of the sequence

class Model(nn.Module):
    """
    Complex sequence transform that:
    - Pads the sequence with zeros (ZeroPad1d) by treating feature dim as channels
    - Applies Layer Normalization across the embedding dimension
    - Projects each position with a learned linear projection
    - Applies a gated non-linearity (tanh * linear output)
    - Adds a scaled residual connection from the original (un-padded) input
    - Returns log-probabilities across the feature dimension via LogSoftmax

    Input shape: (batch, seq_len, dim)
    Output shape: (batch, seq_len, dim) -- log-probabilities across the last dim
    """
    def __init__(self, dim: int, pad: int):
        super(Model, self).__init__()
        self.dim = dim
        self.pad = pad

        # ZeroPad1d expects input of shape (N, C, L) and pads L,
        # so we will permute (B, L, D) -> (B, D, L), apply padding, then permute back.
        # Pad tuple is (left, right)
        self.pad_layer = nn.ZeroPad1d((pad, pad))

        # Normalize across the embedding dimension
        self.layer_norm = nn.LayerNorm(dim)

        # Position-wise linear projection
        self.proj = nn.Linear(dim, dim, bias=True)

        # log-softmax across feature dimension for final output
        self.log_softmax = nn.LogSoftmax(dim=-1)

        # Learnable scalar for residual scaling
        self.res_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Steps:
        1. Save residual (original, unpadded) for later skip connection.
        2. Permute to (B, D, L) and apply ZeroPad1d to pad sequence length.
        3. Permute back to (B, L_padded, D) and apply LayerNorm over D.
        4. Project with a linear layer and apply a gated non-linearity: tanh(projected) * projected.
        5. Slice out the center (unpadded) portion to restore original sequence length.
        6. Add the scaled residual connection.
        7. Apply LogSoftmax across the embedding dimension and return.

        Args:
            x: Tensor of shape (B, L, D)

        Returns:
            Tensor of shape (B, L, D) with log-probabilities across D
        """
        assert x.dim() == 3 and x.size(2) == self.dim, "Input must be (B, L, D)"

        # 1. Residual
        residual = x  # (B, L, D)

        # 2. Pad: permute to (B, D, L)
        x_perm = x.permute(0, 2, 1)  # (B, D, L)
        x_padded = self.pad_layer(x_perm)  # (B, D, L + 2*pad)

        # 3. Permute back to (B, L_padded, D) and LayerNorm over D
        x_padded = x_padded.permute(0, 2, 1)  # (B, L_padded, D)
        x_norm = self.layer_norm(x_padded)  # (B, L_padded, D)

        # 4. Position-wise projection + gated non-linearity
        proj = self.proj(x_norm)  # (B, L_padded, D)
        gated = torch.tanh(proj) * proj  # (B, L_padded, D)  (gated activation)

        # 5. Slice out center to original length
        start = self.pad
        end = start + residual.size(1)
        center = gated[:, start:end, :]  # (B, L, D)

        # 6. Scaled residual addition
        out = center + residual * self.res_scale  # (B, L, D)

        # 7. LogSoftmax across feature dimension
        out = self.log_softmax(out)  # (B, L, D)

        return out

def get_inputs():
    """
    Returns a list containing the main input tensor for the model.
    Input shape: (batch_size, seq_len, dim)
    """
    x = torch.randn(batch_size, seq_len, dim)
    return [x]

def get_init_inputs():
    """
    Returns initialization parameters required to construct the Model.
    """
    return [dim, pad]