import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class Model(nn.Module):
    """
    Complex image-to-vector module that combines replication padding,
    adaptive max pooling, instance normalization (1d), and channel-wise
    learned scaling to produce a compact representation.

    Pipeline:
    1. ReplicationPad2d to expand spatial context.
    2. AdaptiveMaxPool2d to produce a small fixed spatial grid.
    3. Reshape to sequence and apply InstanceNorm1d (per-instance, per-channel).
    4. Learnable channel-wise scale and bias applied across the sequence.
    5. Non-linearity (LeakyReLU).
    6. Two spatial summarizations (mean and max) across the sequence, concatenated
       to produce the final feature vector of size 2 * channels.
    """
    def __init__(
        self,
        channels: int,
        pad: Tuple[int, int, int, int] = (1, 1, 1, 1),
        out_pool: Tuple[int, int] = (4, 4),
        negative_slope: float = 0.02,
        inst_eps: float = 1e-5,
        inst_affine: bool = True,
    ):
        """
        Args:
            channels (int): Number of input channels.
            pad (tuple): ReplicationPad2d pad in form (left, right, top, bottom).
            out_pool (tuple): Output spatial size (H_out, W_out) for AdaptiveMaxPool2d.
            negative_slope (float): Negative slope for LeakyReLU.
            inst_eps (float): Epsilon for InstanceNorm1d.
            inst_affine (bool): Whether InstanceNorm1d has affine parameters.
        """
        super(Model, self).__init__()
        self.channels = channels
        self.pad = nn.ReplicationPad2d(pad)
        self.pool = nn.AdaptiveMaxPool2d(out_pool)
        # InstanceNorm1d expects input of shape (N, C, L) where C == num_features
        self.inst_norm = nn.InstanceNorm1d(num_features=channels, eps=inst_eps, affine=inst_affine)
        # Additional learnable per-channel scale and bias applied after instance norm
        self.scale = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.negative_slope = negative_slope

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W)

        Returns:
            torch.Tensor: Output tensor of shape (B, 2 * C) which concatenates
                          the mean and max pooled summaries over the spatial grid.
        """
        # 1) Pad spatially to bring more boundary information into pooling windows
        x_padded = self.pad(x)  # (B, C, H_pad, W_pad)

        # 2) Adaptive max pool to fixed small grid (H_out, W_out)
        x_pooled = self.pool(x_padded)  # (B, C, H_out, W_out)

        # 3) Reshape to sequence for InstanceNorm1d: (B, C, L)
        B, C, H_out, W_out = x_pooled.shape
        L = H_out * W_out
        x_seq = x_pooled.view(B, C, L)

        # 4) Instance normalization across the sequence dimension for each instance and channel
        x_norm = self.inst_norm(x_seq)  # (B, C, L)

        # 5) Channel-wise learned affine transform (broadcast across sequence)
        x_scaled = x_norm * self.scale.view(1, C, 1) + self.bias.view(1, C, 1)

        # 6) Non-linearity
        x_act = F.leaky_relu(x_scaled, negative_slope=self.negative_slope)

        # 7) Two complementary spatial summaries across the sequence (mean and max)
        x_mean = x_act.mean(dim=2)  # (B, C)
        x_max, _ = x_act.max(dim=2)  # (B, C)

        # 8) Concatenate summaries to produce final feature vector
        out = torch.cat([x_mean, x_max], dim=1)  # (B, 2*C)
        return out

# Configuration / default sizes for tests
batch_size = 8
channels = 32
height = 64
width = 48
pad = (2, 2, 3, 1)         # left, right, top, bottom
out_pool = (7, 5)          # produce a 7x5 grid after pooling
negative_slope = 0.02

def get_inputs():
    """
    Returns the runtime inputs for the model:
    - A 4D tensor shaped (batch_size, channels, height, width)
    """
    x = torch.randn(batch_size, channels, height, width)
    return [x]

def get_init_inputs():
    """
    Returns the initialization arguments to construct the Model:
    [channels, pad, out_pool, negative_slope]
    """
    return [channels, pad, out_pool, negative_slope]