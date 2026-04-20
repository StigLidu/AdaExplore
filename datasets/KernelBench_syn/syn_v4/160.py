import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Sequence processing model that combines a lazily-initialized 1D convolution
    for feature extraction, RMS layer normalization across feature dimension,
    and a gated projection using LogSigmoid for non-linear gating. The model
    reduces sequence dimension with mean pooling and produces class logits.

    Input shape: (batch_size, seq_len, in_channels)
    Output shape: (batch_size, num_classes)
    """
    def __init__(self, hidden_dim: int, num_classes: int, kernel_size: int = 3):
        """
        Initializes the model components.

        Args:
            hidden_dim (int): Number of output channels for convolution and hidden size.
            num_classes (int): Number of output classes for final logits.
            kernel_size (int, optional): Convolution kernel size. Defaults to 3.
        """
        super(Model, self).__init__()
        # Lazily initialize Conv1d so in_channels can be inferred at first forward pass.
        # Use padding to preserve sequence length.
        self.conv = nn.LazyConv1d(out_channels=hidden_dim, kernel_size=kernel_size, padding=kernel_size // 2)
        # RMS normalization over the feature dimension (last dimension after permute)
        self.rmsnorm = nn.RMSNorm(hidden_dim)
        # LogSigmoid activation used as a numerically-stable log-domain gating function
        self.logsigmoid = nn.LogSigmoid()

        # Parameterized linear projection used to compute gating logits.
        # These parameters are small and learned jointly with conv.
        self.proj_weight = nn.Parameter(torch.randn(hidden_dim, hidden_dim) * 0.02)
        self.proj_bias = nn.Parameter(torch.zeros(hidden_dim))

        # Final classifier projection from pooled hidden vector to logits.
        self.out_weight = nn.Parameter(torch.randn(hidden_dim, num_classes) * 0.02)
        self.out_bias = nn.Parameter(torch.zeros(num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Steps:
        1. Permute input to (batch, channels, seq_len) for Conv1d.
        2. Apply LazyConv1d -> (batch, hidden_dim, seq_len).
        3. Permute to (batch, seq_len, hidden_dim) and apply RMSNorm.
        4. Compute a linear projection; apply LogSigmoid then exp to obtain a sigmoid gate.
        5. Elementwise multiply (gating) with normalized features.
        6. Mean-pool across sequence dimension and apply final linear projection to get logits.

        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, seq_len, in_channels).

        Returns:
            torch.Tensor: Logits with shape (batch_size, num_classes).
        """
        # (batch, seq_len, in_channels) -> (batch, in_channels, seq_len)
        x_perm = x.permute(0, 2, 1)
        # Convolutional feature extraction: (batch, hidden_dim, seq_len)
        conv_out = self.conv(x_perm)
        # (batch, hidden_dim, seq_len) -> (batch, seq_len, hidden_dim)
        seq_feat = conv_out.permute(0, 2, 1)
        # Normalize across feature dimension
        normed = self.rmsnorm(seq_feat)  # (batch, seq_len, hidden_dim)

        # Linear projection to compute gating logits: (batch, seq_len, hidden_dim)
        gating_logits = torch.matmul(normed, self.proj_weight) + self.proj_bias

        # Use LogSigmoid (log-domain sigmoid), then exponentiate to recover sigmoid values.
        # This sequence is numerically stable and demonstrates use of nn.LogSigmoid.
        gate = torch.exp(self.logsigmoid(gating_logits))

        # Apply gating
        gated = normed * gate

        # Pool across sequence dimension -> (batch, hidden_dim)
        pooled = gated.mean(dim=1)

        # Final linear projection to logits
        logits = torch.matmul(pooled, self.out_weight) + self.out_bias

        return logits

# Configuration variables
batch_size = 32
seq_len = 512
in_channels = 64
hidden_dim = 128
num_classes = 10
kernel_size = 5

def get_inputs():
    """
    Returns example input tensors for the model.

    Input tensor shape: (batch_size, seq_len, in_channels)
    """
    x = torch.randn(batch_size, seq_len, in_channels, dtype=torch.float32)
    return [x]

def get_init_inputs():
    """
    Returns initialization arguments for the Model constructor:
    [hidden_dim, num_classes, kernel_size]
    """
    return [hidden_dim, num_classes, kernel_size]