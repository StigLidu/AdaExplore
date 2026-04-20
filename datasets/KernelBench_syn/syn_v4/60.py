import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Complex 3D context-aware channel-covariance projector.

    Pipeline:
    1. Reflection-pad the input volume to gather local boundary context.
    2. Compute per-channel RMS energy from the padded volume and normalize the original input channels by this energy.
    3. Apply Hardswish activation to introduce non-linearity.
    4. Flatten spatial dimensions and compute a per-sample channel covariance matrix.
    5. Apply LogSigmoid to the covariance, project it with a learnable matrix, and aggregate channels to produce final features.

    Input:
        x: Tensor of shape (N, C, D, H, W)

    Output:
        Tensor of shape (N, out_features)
    """
    def __init__(self, in_channels: int, out_features: int, pad: int = 1, eps: float = 1e-6):
        super(Model, self).__init__()
        self.in_channels = in_channels
        self.out_features = out_features
        self.pad = pad
        self.eps = eps

        # Reflection padding to provide local spatial context at boundaries
        self.pad_layer = nn.ReflectionPad3d(self.pad)

        # Non-linearities
        self.hswish = nn.Hardswish()
        self.logsigmoid = nn.LogSigmoid()

        # Learnable projection from channel-covariance space to output features
        # Shape: (C, out_features)
        self.proj = nn.Parameter(torch.randn(self.in_channels, self.out_features) * (1.0 / (self.in_channels ** 0.5)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: (N, C, D, H, W) input tensor

        Returns:
            out: (N, out_features) feature tensor
        """
        N, C, D, H, W = x.shape

        # 1) Pad the input to gather context for energy estimation
        padded = self.pad_layer(x)  # (N, C, D+2p, H+2p, W+2p)

        # 2) Compute per-channel RMS energy from padded context and normalize original input channels
        #    energy shape -> (N, C, 1, 1, 1)
        energy = torch.mean(padded * padded, dim=(2, 3, 4), keepdim=True) + self.eps
        scale = 1.0 / torch.sqrt(energy)  # (N, C, 1, 1, 1)
        x_scaled = x * scale  # broadcast to (N, C, D, H, W)

        # 3) Apply non-linearity
        x_act = self.hswish(x_scaled)  # (N, C, D, H, W)

        # 4) Flatten spatial dimensions to compute channel covariance per sample
        S = D * H * W
        x_flat = x_act.view(N, C, S)  # (N, C, S)

        # Compute a channel covariance-like matrix: (N, C, C)
        # Note: dividing by S to get scale-invariant covariance estimate
        cov = torch.matmul(x_flat, x_flat.transpose(1, 2)) / float(S)  # (N, C, C)

        # 5) Apply LogSigmoid to covariance to introduce another non-linear transformation
        cov_nl = self.logsigmoid(cov)  # (N, C, C)

        # 6) Project covariance via learnable matrix: (N, C, Out)
        projected = torch.matmul(cov_nl, self.proj)  # (N, C, out_features)

        # 7) Aggregate channel dimension to produce final feature vector per sample
        out = torch.mean(projected, dim=1)  # (N, out_features)

        # Final Hardswish to add mild non-linearity on outputs
        out = self.hswish(out)

        return out

# Module-level configuration variables
batch_size = 8
in_channels = 32
D = 8
H = 16
W = 16
out_features = 128
pad = 1
eps = 1e-6

def get_inputs():
    """
    Returns a list with one input tensor of shape (batch_size, in_channels, D, H, W).
    """
    x = torch.randn(batch_size, in_channels, D, H, W)
    return [x]

def get_init_inputs():
    """
    Returns initialization parameters to construct the Model:
    [in_channels, out_features, pad, eps]
    """
    return [in_channels, out_features, pad, eps]