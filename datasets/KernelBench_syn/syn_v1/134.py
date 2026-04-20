import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Complex module combining LocalResponseNorm, Dropout3d, and Hardshrink with
    channel-wise linear projections and residual connections.

    Computation outline:
    1. Apply Local Response Normalization over channels.
    2. Flatten spatial dims and project channels -> proj_channels.
    3. Reshape to image form, add a singleton "depth" dim and apply Dropout3d.
    4. Apply Hardshrink nonlinearity element-wise.
    5. Project back to original channel dimension and add residual connection.
    6. Global spatial average pooling followed by a final linear projection to produce outputs.
    """
    def __init__(
        self,
        in_channels: int,
        proj_channels: int,
        out_features: int,
        lrn_size: int = 5,
        lrn_alpha: float = 1e-4,
        lrn_beta: float = 0.75,
        dropout_p: float = 0.5,
        hardshrink_lambda: float = 0.5,
    ):
        super(Model, self).__init__()
        # Normalization across channels
        self.lrn = nn.LocalResponseNorm(size=lrn_size, alpha=lrn_alpha, beta=lrn_beta)
        # Drop entire channels in a 3D sense (we'll add a singleton depth dim)
        self.dropout = nn.Dropout3d(p=dropout_p)
        # Element-wise shrinkage non-linearity
        self.hardshrink = nn.Hardshrink(lambd=hardshrink_lambda)

        # Learnable linear projections implemented as parameters (channel-wise)
        # proj: maps in_channels -> proj_channels
        self.proj = nn.Parameter(torch.randn(in_channels, proj_channels) * (1.0 / max(1, in_channels)**0.5))
        # res_proj: maps proj_channels -> in_channels (for residual path)
        self.res_proj = nn.Parameter(torch.randn(proj_channels, in_channels) * (1.0 / max(1, proj_channels)**0.5))
        # final fully-connected mapping from channels to out_features
        self.fc = nn.Parameter(torch.randn(in_channels, out_features) * (1.0 / max(1, in_channels)**0.5))
        self.fc_bias = nn.Parameter(torch.zeros(out_features))

        # Keep dims for sanity checks or potential use
        self.in_channels = in_channels
        self.proj_channels = proj_channels
        self.out_features = out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, in_channels, height, width)

        Returns:
            Tensor of shape (batch_size, out_features)
        """
        B, C, H, W = x.shape
        assert C == self.in_channels, f"Expected input with {self.in_channels} channels, got {C}"

        # 1) Local response normalization (across channels)
        x_norm = self.lrn(x)  # (B, C, H, W)

        # 2) Flatten spatial dims and prepare for channel projection
        S = H * W
        x_flat = x_norm.view(B, C, S)  # (B, C, S)
        x_flat_t = x_flat.transpose(1, 2)  # (B, S, C)

        # 3) Project channels -> proj_channels using batch matmul with learnable parameter
        #    (B, S, C) @ (C, P) -> (B, S, P)
        y = torch.matmul(x_flat_t, self.proj)  # (B, S, P)

        # 4) Reshape back to image grid with proj_channels
        y = y.transpose(1, 2).contiguous().view(B, self.proj_channels, H, W)  # (B, P, H, W)

        # 5) Add a singleton depth dimension to apply Dropout3d which expects 5D input
        y_depth = y.unsqueeze(2)  # (B, P, 1, H, W)
        y_dropped = self.dropout(y_depth)  # (B, P, 1, H, W)
        y_dropped = y_dropped.squeeze(2)  # (B, P, H, W)

        # 6) Hardshrink non-linearity (element-wise)
        y_shrunk = self.hardshrink(y_dropped)  # (B, P, H, W)

        # 7) Project back to original channel dimensionality to form a residual
        y_shrunk_flat = y_shrunk.view(B, self.proj_channels, S)  # (B, P, S)
        y_shrunk_t = y_shrunk_flat.transpose(1, 2)  # (B, S, P)
        # (B, S, P) @ (P, C) -> (B, S, C)
        res = torch.matmul(y_shrunk_t, self.res_proj)  # (B, S, C)
        res = res.transpose(1, 2).contiguous().view(B, C, H, W)  # (B, C, H, W)

        # 8) Residual connection with the normalized input
        out = x_norm + res  # (B, C, H, W)

        # 9) Global average pooling over spatial dims -> (B, C)
        out_pool = out.mean(dim=(2, 3))  # (B, C)

        # 10) Final linear projection to out_features: (B, C) @ (C, out_features) + bias -> (B, out_features)
        out_final = torch.matmul(out_pool, self.fc) + self.fc_bias  # (B, out_features)

        return out_final

# Configuration variables (module-level)
BATCH_SIZE = 8
IN_CHANNELS = 32
HEIGHT = 16
WIDTH = 16

PROJ_CHANNELS = 16   # intermediate channel projection size
OUT_FEATURES = 64    # final output dimensionality

LRN_SIZE = 5
LRN_ALPHA = 1e-4
LRN_BETA = 0.75

DROPOUT_P = 0.3
HARDSHINK_LAMBDA = 0.2

def get_inputs():
    """
    Returns a list with a single input tensor of shape (BATCH_SIZE, IN_CHANNELS, HEIGHT, WIDTH).
    """
    x = torch.randn(BATCH_SIZE, IN_CHANNELS, HEIGHT, WIDTH)
    return [x]

def get_init_inputs():
    """
    Returns initialization parameters for the Model constructor in the order:
    (in_channels, proj_channels, out_features, lrn_size, lrn_alpha, lrn_beta, dropout_p, hardshrink_lambda)
    """
    return [IN_CHANNELS, PROJ_CHANNELS, OUT_FEATURES, LRN_SIZE, LRN_ALPHA, LRN_BETA, DROPOUT_P, HARDSHINK_LAMBDA]