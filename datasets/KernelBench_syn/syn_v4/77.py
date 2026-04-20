import torch
import torch.nn as nn

# Configuration variables
batch_size = 8
in1_features = 128
in2_features = 64
bilinear_out = 256
hidden_dim = 512
use_bias = True

class Model(nn.Module):
    """
    A composite module that fuses two input streams with a bilinear layer,
    then applies nonlinear transforms, a projection bottleneck, gating-like
    residual path, and a final L2 normalization.

    Computation pattern:
      1. y = Bilinear(x1, x2)                        # (batch, bilinear_out)
      2. y = GELU(y)
      3. h = Linear(y)                                # project to hidden_dim
      4. h = ReLU6(h)
      5. s = Linear(h)                                # project back to bilinear_out
      6. out = y + s                                  # residual fusion
      7. out = out / (norm(out) + eps)                # L2 normalize across feature dim

    This combines nn.Bilinear, nn.GELU, and nn.ReLU6 in a small fusion block.
    """
    def __init__(self, in1: int, in2: int, bilinear_out: int, hidden: int, bias: bool = True):
        """
        Initialize the Model.

        Args:
            in1 (int): Size of the first input feature vector.
            in2 (int): Size of the second input feature vector.
            bilinear_out (int): Output feature size of the bilinear layer.
            hidden (int): Hidden dimension used for the bottleneck projection.
            bias (bool): Whether to use bias in linear/bilinear layers.
        """
        super(Model, self).__init__()
        self.bilinear = nn.Bilinear(in1, in2, bilinear_out, bias=bias)
        self.gelu = nn.GELU()
        # Projection to a wider hidden space, then nonlinearity and projection back
        self.proj_up = nn.Linear(bilinear_out, hidden, bias=bias)
        self.relu6 = nn.ReLU6()
        self.proj_down = nn.Linear(hidden, bilinear_out, bias=bias)
        # Small epsilon for numerical stability in normalization
        self.eps = 1e-6

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        Forward pass that fuses x1 and x2.

        Args:
            x1 (torch.Tensor): Tensor of shape (batch_size, in1).
            x2 (torch.Tensor): Tensor of shape (batch_size, in2).

        Returns:
            torch.Tensor: Normalized fused output of shape (batch_size, bilinear_out).
        """
        # 1) Bilinear fusion
        fused = self.bilinear(x1, x2)                      # (batch, bilinear_out)

        # 2) Nonlinear activation
        activated = self.gelu(fused)                       # (batch, bilinear_out)

        # 3) Bottleneck projection up
        up = self.proj_up(activated)                       # (batch, hidden_dim)

        # 4) Bounded nonlinearity (helps keep activations in a stable range)
        up_activated = self.relu6(up)                      # (batch, hidden_dim)

        # 5) Project back to match bilinear output size
        down = self.proj_down(up_activated)                # (batch, bilinear_out)

        # 6) Residual fusion and final normalization
        out = activated + down                             # (batch, bilinear_out)
        norm = torch.norm(out, p=2, dim=1, keepdim=True)   # (batch, 1)
        out = out / (norm + self.eps)

        return out

def get_inputs():
    """
    Returns a list of input tensors for the model: [x1, x2]
    x1: (batch_size, in1_features)
    x2: (batch_size, in2_features)
    """
    x1 = torch.randn(batch_size, in1_features)
    x2 = torch.randn(batch_size, in2_features)
    return [x1, x2]

def get_init_inputs():
    """
    Returns initialization parameters for the Model constructor:
    [in1_features, in2_features, bilinear_out, hidden_dim, use_bias]
    """
    return [in1_features, in2_features, bilinear_out, hidden_dim, use_bias]