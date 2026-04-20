import torch
import torch.nn as nn

# Configuration
BATCH = 8
IN_CHANNELS = 64
PROJ_OUT_CHANNELS = 128
H = 32
W = 32
SCALE_FACTOR = 2
DROPOUT_P = 0.1

class Model(nn.Module):
    """
    Complex model that combines upsampling, batch normalization, channel dropout (FeatureAlphaDropout),
    spatial pooling and a learned channel projection. The model is designed to accept an input tensor X
    of shape (B, C, H, W) and optionally an external projection matrix P of shape (out_c, C) to
    override the internal projection weights.

    Computation pattern (forward):
    1. Upsample spatial dimensions using nearest-neighbor upsampling.
    2. Apply BatchNorm2d over channels.
    3. Apply ReLU nonlinearity.
    4. Apply FeatureAlphaDropout to randomly mask channels.
    5. Global average pool spatial dimensions to produce a (B, C) tensor.
    6. Apply a linear channel projection (either internal learned weight or external P).
    7. Apply sigmoid activation to produce final (B, out_c) outputs.
    """
    def __init__(self, in_channels: int, out_channels: int, scale_factor: int = 2, dropout_p: float = 0.1):
        super(Model, self).__init__()
        self.scale_factor = scale_factor
        # Upsampling layer (nearest neighbor)
        self.upsample = nn.UpsamplingNearest2d(scale_factor=scale_factor)
        # Batch normalization over channels
        self.bn = nn.BatchNorm2d(in_channels)
        # Channel-wise dropout (FeatureAlphaDropout)
        self.dropout = nn.FeatureAlphaDropout(dropout_p)
        # Internal projection: learnable weight and bias for channel projection
        # weight shape: (out_channels, in_channels)
        self.proj_weight = nn.Parameter(torch.randn(out_channels, in_channels) * (1.0 / (in_channels ** 0.5)))
        self.proj_bias = nn.Parameter(torch.zeros(out_channels))
        # Adaptive pooling to collapse spatial dimensions
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, X: torch.Tensor, P: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass.

        Args:
            X (torch.Tensor): Input tensor of shape (B, C, H, W).
            P (torch.Tensor, optional): External projection matrix of shape (out_c, C).
                                        If provided, it overrides the internal projection weight.

        Returns:
            torch.Tensor: Output tensor of shape (B, out_c).
        """
        # 1. Upsample spatially
        x = self.upsample(X)  # (B, C, H * scale, W * scale)

        # 2. Batch normalization
        x = self.bn(x)

        # 3. Nonlinearity
        x = torch.relu(x)

        # 4. Channel dropout
        x = self.dropout(x)

        # 5. Global average pooling to (B, C)
        x = self.pool(x)               # (B, C, 1, 1)
        x = x.view(x.size(0), -1)      # (B, C)

        # 6. Channel projection (matmul with weight^T to go from (B, C) -> (B, out_c))
        weight = P if P is not None else self.proj_weight  # (out_c, C)
        out = torch.matmul(x, weight.t()) + self.proj_bias  # (B, out_c)

        # 7. Final activation
        return torch.sigmoid(out)


def get_inputs():
    """
    Create example input tensors for the model.

    Returns:
        list: [X, P] where:
            - X is a random input tensor of shape (BATCH, IN_CHANNELS, H, W)
            - P is an optional projection matrix of shape (PROJ_OUT_CHANNELS, IN_CHANNELS)
    """
    X = torch.randn(BATCH, IN_CHANNELS, H, W)
    # Provide an external projection matrix to demonstrate override capability
    P = torch.randn(PROJ_OUT_CHANNELS, IN_CHANNELS) * (1.0 / (IN_CHANNELS ** 0.5))
    return [X, P]


def get_init_inputs():
    """
    Return initialization parameters for constructing the Model instance.

    Returns:
        list: [in_channels, out_channels, scale_factor, dropout_p]
    """
    return [IN_CHANNELS, PROJ_OUT_CHANNELS, SCALE_FACTOR, DROPOUT_P]