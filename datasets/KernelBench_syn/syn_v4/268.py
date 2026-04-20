import torch
import torch.nn as nn

"""
Complex model that combines 3D Lp-pooling, 2D batch normalization, and 1D dropout.
The model ingests a 5D tensor (N, C, D, H, W), reduces its spatial footprint with LPPool3d,
collapses dimensions to apply BatchNorm2d, applies channel dropout (Dropout1d), and finally
projects to a smaller feature vector per sample.
"""

# Configuration / shape parameters
BATCH = 8
C = 64
D = 16
H = 32
W = 32
OUT_FEATURES = 128
LPP_NORM = 2  # p value for LPPool3d
LPP_KERNEL = (3, 3, 3)
LPP_STRIDE = (2, 2, 2)
DROPOUT_P = 0.3

class Model(nn.Module):
    """
    Model pipeline:
    1) LPPool3d (power-average pooling) over (D, H, W)
    2) Reshape pooled 5D tensor -> 4D tensor by merging D and H into a single spatial dim for BatchNorm2d
    3) BatchNorm2d across channels
    4) ReLU activation
    5) Collapse spatial dims to produce a 3D tensor for Dropout1d (channel-wise dropout)
    6) Global mean over sequence length -> per-channel summary
    7) Linear projection to OUT_FEATURES
    """
    def __init__(self):
        super(Model, self).__init__()
        # 3D power-average pooling
        self.pool3d = nn.LPPool3d(norm_type=LPP_NORM, kernel_size=LPP_KERNEL, stride=LPP_STRIDE)
        # BatchNorm2d needs number of channels C
        self.bn2d = nn.BatchNorm2d(num_features=C)
        # Dropout1d performs channel-wise dropout on 3D inputs (N, C, L)
        self.dropout1d = nn.Dropout1d(p=DROPOUT_P)
        # Final projection
        self.fc = nn.Linear(C, OUT_FEATURES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (N, C, D, H, W)

        Returns:
            Tensor of shape (N, OUT_FEATURES)
        """
        # 1) LPPool3d -> (N, C, D_p, H_p, W_p)
        x = self.pool3d(x)

        # 2) Merge depth and height dims into a single spatial dim to make 4D input for BatchNorm2d
        # x.shape -> (N, C, D_p * H_p, W_p)
        N, C_, Dp, Hp, Wp = x.shape
        x = x.view(N, C_, Dp * Hp, Wp)

        # 3) Batch normalization over (N, C, H, W)
        x = self.bn2d(x)

        # 4) Non-linearity
        x = torch.relu(x)

        # 5) Collapse last two spatial dimensions to create a sequence length for Dropout1d
        #    new shape -> (N, C, L) where L = (D_p * H_p * W_p)
        N, C_, Hp2, Wp2 = x.shape
        x = x.view(N, C_, Hp2 * Wp2)

        # 6) Channel-wise dropout
        x = self.dropout1d(x)

        # 7) Global mean across the sequence dimension to get (N, C)
        x = x.mean(dim=2)

        # 8) Final linear projection to OUT_FEATURES
        out = self.fc(x)
        return out

def get_inputs():
    """
    Returns a list with a single 5D input tensor matching the configured shapes:
    (BATCH, C, D, H, W)
    """
    x = torch.randn(BATCH, C, D, H, W)
    return [x]

def get_init_inputs():
    """
    No external initialization parameters are required for this module since
    layer parameters are created in the Model.__init__.
    """
    return []