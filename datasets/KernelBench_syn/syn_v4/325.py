import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Complex 1D sequence processing module combining convolutional residual blocks,
    FeatureAlphaDropout, Dropout1d, and Tanhshrink non-linearity. The block includes:
      - Initial Conv1d + BatchNorm1d + ReLU
      - FeatureAlphaDropout (channel-wise alpha dropout)
      - Second Conv1d + BatchNorm1d + Tanhshrink
      - Dropout1d (channel dropout)
      - Residual connection (with 1x1 conv if channel dims differ)
      - Squeeze-and-excitation style channel gating (AdaptiveAvgPool1d -> Linear -> sigmoid)
      - Final 1x1 projection back to input channels and global residual addition
    """
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        kernel_size: int = 3,
        dropout_p: float = 0.1,
        feature_dropout_p: float = 0.05
    ):
        """
        Args:
            in_channels (int): Number of channels in the input sequence.
            hidden_channels (int): Number of channels in the hidden convolutional layers.
            kernel_size (int): Kernel size for the main conv layer (will use padding to preserve length).
            dropout_p (float): Probability for Dropout1d.
            feature_dropout_p (float): Probability for FeatureAlphaDropout.
        """
        super(Model, self).__init__()
        padding = kernel_size // 2

        # First convolutional projection
        self.conv1 = nn.Conv1d(in_channels, hidden_channels, kernel_size=kernel_size, stride=1, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm1d(hidden_channels)

        # Channel-wise alpha dropout to randomly drop entire features with alpha preservation
        self.feat_dropout = nn.FeatureAlphaDropout(p=feature_dropout_p)

        # Second convolution that preserves sequence length
        self.conv2 = nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(hidden_channels)

        # Non-linearity from provided list
        self.act = nn.Tanhshrink()

        # Channel-wise dropout (Dropout1d) for regularization after non-linearity
        self.drop1d = nn.Dropout1d(p=dropout_p)

        # Residual adjustment if input/output channel dims differ
        if in_channels != hidden_channels:
            self.res_conv = nn.Conv1d(in_channels, hidden_channels, kernel_size=1, bias=False)
        else:
            self.res_conv = None

        # Squeeze-and-excitation style gating: reduce to channels and compute gating scalar per channel
        self.global_pool = nn.AdaptiveAvgPool1d(1)  # output shape (B, C, 1)
        self.fc = nn.Linear(hidden_channels, hidden_channels // 4 if hidden_channels >= 4 else 1)
        self.fc_expand = nn.Linear(hidden_channels // 4 if hidden_channels >= 4 else 1, hidden_channels)

        # Final projection back to input channels and output residual addition
        self.project_back = nn.Conv1d(hidden_channels, in_channels, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, in_channels, seq_len)

        Returns:
            torch.Tensor: Output tensor of same shape as input (batch, in_channels, seq_len)
        """
        # Save residual
        residual = x

        # First conv -> BN -> ReLU
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)

        # FeatureAlphaDropout (channel-wise alpha dropout)
        out = self.feat_dropout(out)

        # Second conv -> BN -> Tanhshrink
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.act(out)  # Tanhshrink non-linearity

        # Dropout1d (channel-wise dropout)
        out = self.drop1d(out)

        # Residual addition (project if necessary)
        if self.res_conv is not None:
            res = self.res_conv(residual)
        else:
            res = residual
        out = out + res

        # Squeeze-and-excitation style gating
        # Global pooling to (B, C, 1) -> (B, C)
        gap = self.global_pool(out).squeeze(-1)
        # Bottleneck -> ReLU -> expand -> sigmoid
        gating = self.fc(gap)
        gating = F.relu(gating, inplace=True)
        gating = self.fc_expand(gating)
        gating = torch.sigmoid(gating).unsqueeze(-1)  # (B, C, 1)
        out = out * gating  # Scale channels

        # Project back to original input channels and add global residual
        out = self.project_back(out)
        out = out + residual

        return out


# Configuration variables
batch_size = 8
in_channels = 64
hidden_channels = 128
seq_len = 512
kernel_size = 5
dropout_p = 0.1
feature_dropout_p = 0.05

def get_inputs():
    """
    Returns a list containing a single input tensor with shape (batch_size, in_channels, seq_len).
    """
    x = torch.randn(batch_size, in_channels, seq_len)
    return [x]

def get_init_inputs():
    """
    Returns the initialization parameters for the Model constructor in order:
      [in_channels, hidden_channels, kernel_size, dropout_p, feature_dropout_p]
    """
    return [in_channels, hidden_channels, kernel_size, dropout_p, feature_dropout_p]