import torch
import torch.nn as nn

class Model(nn.Module):
    """
    A moderately complex convolutional classification head that combines:
    - two Conv2d layers with BatchNorm and ReLU
    - a residual connection
    - FeatureAlphaDropout (channel-wise Alpha dropout)
    - AlphaDropout (element-wise Alpha dropout)
    - Hardshrink non-linearity for sparse activation
    - AdaptiveAvgPool2d + Linear classifier

    The combination demonstrates channel-wise and element-wise stochastic regularization
    together with a shrinkage non-linearity to produce sparse, robust features.
    """
    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        out_features: int,
        kernel_size: int,
        feature_alpha_p: float,
        alpha_p: float,
        hardshrink_lambda: float
    ):
        """
        Initializes the model.

        Args:
            in_channels (int): Number of input channels.
            mid_channels (int): Number of channels used in the intermediate convolutions.
            out_features (int): Dimension of the final linear output.
            kernel_size (int): Kernel size for convolutional layers.
            feature_alpha_p (float): Probability for FeatureAlphaDropout (channel-wise).
            alpha_p (float): Probability for AlphaDropout (element-wise).
            hardshrink_lambda (float): Lambda threshold for Hardshrink activation.
        """
        super(Model, self).__init__()
        padding = kernel_size // 2

        # Initial conv block
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.relu = nn.ReLU(inplace=True)

        # Second conv transforms features; kept same channels to allow residual add
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)

        # Regularizers and non-linearity
        self.feature_alpha_dropout = nn.FeatureAlphaDropout(feature_alpha_p)
        self.alpha_dropout = nn.AlphaDropout(alpha_p)
        self.hardshrink = nn.Hardshrink(lambd=hardshrink_lambda)

        # Global pooling + classifier
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(mid_channels, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass combining convs, residual add, dropout variants, and shrinkage.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C_in, H, W)

        Returns:
            torch.Tensor: Output tensor of shape (B, out_features)
        """
        # First conv -> BN -> ReLU
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # Save residual (post-activation)
        residual = out

        # Second conv -> BN
        out = self.conv2(out)
        out = self.bn2(out)

        # Channel-wise stochastic regularization (FeatureAlphaDropout)
        out = self.feature_alpha_dropout(out)

        # Residual connection (post-dropout)
        out = out + residual

        # Element-wise AlphaDropout to maintain self-normalizing properties with SELU-like behavior
        out = self.alpha_dropout(out)

        # Hard shrinkage to enforce sparsity in the activations
        out = self.hardshrink(out)

        # Global pooling and final linear layer
        out = self.global_pool(out)       # shape (B, C, 1, 1)
        out = out.view(out.size(0), -1)   # shape (B, C)
        out = self.fc(out)                # shape (B, out_features)
        return out

# Configuration / default inputs
batch_size = 8
in_channels = 3
mid_channels = 64
height = 128
width = 128
kernel_size = 3
feature_alpha_p = 0.1
alpha_p = 0.05
hardshrink_lambda = 0.7
out_features = 1000

def get_inputs():
    """
    Returns a list with a single input tensor for the forward pass.
    """
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]

def get_init_inputs():
    """
    Returns initialization parameters for the Model constructor in the same order.
    """
    return [in_channels, mid_channels, out_features, kernel_size, feature_alpha_p, alpha_p, hardshrink_lambda]