import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Complex model that combines 3D batch normalization, 2D convolution (applied
    across depth slices), and 1D channel-wise dropout with a small depth-attention
    aggregation. 

    Pipeline:
      - Input x: (B, C, D, H, W)
      - BatchNorm3d over C -> same shape
      - Reshape to (B*D, C, H, W) to apply a shared Conv2d across depth slices
      - ReLU activation and spatial global average pooling -> (B, C_out, D)
      - Dropout1d operates on (B, C_out, D) to randomly zero channels
      - A sigmoid gating and depth-wise weighted aggregation produces (B, C_out)
      - Final linear projection returns (B, C_out)
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        dropout_prob: float = 0.2,
        bn_eps: float = 1e-5,
    ):
        """
        Initializes the submodules.

        Args:
            in_channels (int): Number of channels in the input (C).
            out_channels (int): Number of output channels from the Conv2d.
            kernel_size (int, optional): Kernel size for the Conv2d. Defaults to 3.
            stride (int, optional): Stride for the Conv2d. Defaults to 1.
            padding (int, optional): Padding for the Conv2d. Defaults to 1.
            dropout_prob (float, optional): Dropout probability for Dropout1d. Defaults to 0.2.
            bn_eps (float, optional): Epsilon for BatchNorm3d. Defaults to 1e-5.
        """
        super(Model, self).__init__()
        # BatchNorm over 3D inputs: expects num_features == in_channels
        self.bn3d = nn.BatchNorm3d(num_features=in_channels, eps=bn_eps)

        # Conv2d operates on (N, C, H, W). We will reshape (B, C, D, H, W) -> (B*D, C, H, W)
        self.conv2d = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

        # Channel-wise dropout for sequences: expects input shape (N, C, L)
        self.dropout1d = nn.Dropout1d(p=dropout_prob)

        # Final projection from per-channel aggregated features
        self.fc = nn.Linear(out_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, D, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (B, out_channels).
        """
        # Validate input dims
        if x.dim() != 5:
            raise ValueError(f"Expected 5D input (B, C, D, H, W), got shape {tuple(x.shape)}")

        B, C, D, H, W = x.shape

        # 1) Batch-normalize across channels for 3D volumes
        x_bn = self.bn3d(x)  # shape: (B, C, D, H, W)

        # 2) Reshape to apply the same 2D convolution to each depth slice independently
        # Move depth to batch dimension: (B, C, D, H, W) -> (B, D, C, H, W) -> (B*D, C, H, W)
        x_slices = x_bn.permute(0, 2, 1, 3, 4).contiguous().view(B * D, C, H, W)

        # 3) Shared 2D convolution applied to every depth slice
        conv_out = self.conv2d(x_slices)  # shape: (B*D, out_channels, H_out, W_out)
        conv_act = F.relu(conv_out, inplace=True)

        # 4) Global spatial average pooling to summarize each slice -> (B*D, out_channels)
        pooled = conv_act.mean(dim=(2, 3))  # spatial mean

        # 5) Restore depth dimension and reorder to (B, out_channels, D) for Dropout1d
        pooled = pooled.view(B, D, -1).permute(0, 2, 1).contiguous()  # (B, out_channels, D)

        # 6) Channel-wise dropout across the depth sequence
        dropped = self.dropout1d(pooled)  # (B, out_channels, D)

        # 7) Simple depth-attention: compute soft weights across depth and aggregate
        # Use a sigmoid gating to produce per-element gates, then softmax across depth
        gates = torch.sigmoid(dropped)  # (B, out_channels, D)
        attn_logits = dropped * gates  # elementwise modulation
        attn_weights = F.softmax(attn_logits, dim=2)  # softmax over depth dimension

        # Weighted sum across depth -> (B, out_channels)
        aggregated = (pooled * attn_weights).sum(dim=2)

        # 8) Final linear projection
        out = self.fc(aggregated)  # (B, out_channels)

        return out

# Configuration / default inputs
batch_size = 8
in_channels = 16
out_channels = 32
depth = 4
height = 64
width = 64

kernel_size = 3
stride = 1
padding = 1
dropout_prob = 0.25
bn_eps = 1e-5

def get_inputs():
    """
    Returns the runtime inputs to the model:
      - x: Tensor of shape (batch_size, in_channels, depth, height, width)
    """
    x = torch.randn(batch_size, in_channels, depth, height, width)
    return [x]

def get_init_inputs():
    """
    Returns the initialization parameters for the Model constructor in the same order:
      in_channels, out_channels, kernel_size, stride, padding, dropout_prob, bn_eps
    """
    return [in_channels, out_channels, kernel_size, stride, padding, dropout_prob, bn_eps]