import torch
import torch.nn as nn

# Module-level configuration
batch_size = 8
in_channels = 16
height = 64
width = 64
cond_dim = 32
hidden_dim = 128
output_dim = 10

# Pooling and dropout hyperparameters
pool_p = 2          # Lp norm degree for LPPool2d
pool_kernel = 3
pool_stride = 2
dropout_p = 0.1

class Model(nn.Module):
    """
    Complex model that fuses a spatial input tensor with a conditioning vector,
    applies non-linear activations, Lp pooling, convolutional mixing, and
    AlphaDropout before producing a final classification/regression vector.

    Computation graph (high-level):
      1. Project conditioning vector -> channel bias and add to spatial input
      2. Softplus activation
      3. Lp pooling (LPPool2d)
      4. 2D convolutional mixing (Conv2d)
      5. Softplus activation
      6. AlphaDropout
      7. Global adaptive average pooling -> flatten
      8. Linear projection to output_dim
    """
    def __init__(
        self,
        in_channels: int,
        cond_dim: int,
        hidden_dim: int,
        output_dim: int,
        pool_p: int = 2,
        pool_kernel: int = 3,
        pool_stride: int = 2,
        dropout_p: float = 0.1,
    ):
        """
        Initializes the Model.

        Args:
            in_channels (int): Number of channels in the spatial input.
            cond_dim (int): Dimension of the conditioning vector.
            hidden_dim (int): Hidden dimensionality for intermediate linear layer.
            output_dim (int): Output dimensionality.
            pool_p (int): Norm degree for LPPool2d.
            pool_kernel (int): Kernel size for LPPool2d.
            pool_stride (int): Stride for LPPool2d.
            dropout_p (float): Dropout probability for AlphaDropout.
        """
        super(Model, self).__init__()
        self.in_channels = in_channels
        self.cond_dim = cond_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Project conditioning vector into channel-wise bias that can be broadcast and added to x
        self.cond_proj = nn.Sequential(
            nn.Linear(cond_dim, in_channels),
            nn.Softplus()  # ensure non-negative bias if desired
        )

        # Lp pooling
        self.lppool = nn.LPPool2d(norm_type=pool_p, kernel_size=pool_kernel, stride=pool_stride)

        # Convolutional mixing after pooling
        self.conv = nn.Conv2d(in_channels, hidden_dim // 2, kernel_size=3, padding=1, bias=True)
        self.act = nn.Softplus()

        # AlphaDropout to preserve self-normalizing properties when needed
        self.alpha_dropout = nn.AlphaDropout(p=dropout_p)

        # Global pooling and final projection
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(hidden_dim // 2, output_dim)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Spatial input tensor with shape (B, C, H, W).
            cond (torch.Tensor): Conditioning tensor with shape (B, cond_dim).

        Returns:
            torch.Tensor: Output tensor with shape (B, output_dim).
        """
        # Project conditioning to channel biases and add to spatial tensor (broadcast)
        # cond_proj -> (B, C) -> (B, C, 1, 1) -> broadcast add to x
        bias = self.cond_proj(cond).unsqueeze(-1).unsqueeze(-1)
        x = x + bias

        # Non-linear gating
        x = self.act(x)

        # Lp pooling reduces spatial resolution and aggregates information
        x = self.lppool(x)

        # Convolutional mixing to increase channel interactions
        x = self.conv(x)
        x = self.act(x)

        # Regularization that preserves mean/variance properties (AlphaDropout)
        x = self.alpha_dropout(x)

        # Global pooling -> flatten -> final linear
        x = self.global_pool(x)  # (B, C', 1, 1)
        x = torch.flatten(x, 1)  # (B, C')
        out = self.fc(x)         # (B, output_dim)
        return out

def get_inputs():
    """
    Returns a list of input tensors for the forward method:
      - x: spatial input tensor of shape (batch_size, in_channels, height, width)
      - cond: conditioning vector of shape (batch_size, cond_dim)
    """
    x = torch.randn(batch_size, in_channels, height, width)
    cond = torch.randn(batch_size, cond_dim)
    return [x, cond]

def get_init_inputs():
    """
    Returns initialization parameters for constructing the Model:
      [in_channels, cond_dim, hidden_dim, output_dim, pool_p, pool_kernel, pool_stride, dropout_p]
    """
    return [in_channels, cond_dim, hidden_dim, output_dim, pool_p, pool_kernel, pool_stride, dropout_p]