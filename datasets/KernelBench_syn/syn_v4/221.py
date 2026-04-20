import torch
import torch.nn as nn
import torch.nn.functional as F

# Configuration
batch_size = 8
input_channels = 3
depth = 4
height = 32
width = 32

hidden_size = 128
num_layers = 2

class Model(nn.Module):
    """
    Complex model combining 3D replication padding, circular 2D padding across spatial slices,
    per-slice spatial pooling, and a temporal GRU to aggregate across the depth dimension.

    Pipeline:
    1. ReplicationPad3d to expand the 3D input (depth, height, width).
    2. Reinterpret depth slices as a batch of 2D images and apply CircularPad2d to each slice.
    3. Adaptive average pool each slice down to a channel vector.
    4. Form a sequence across the (padded) depth dimension and process with an nn.GRU.
    5. Project the final GRU hidden output back to channel dimension and fuse with a global pooled residual.
    6. L2-normalize the resulting channel vectors per sample.
    """
    def __init__(self, input_channels: int = input_channels, hidden_size: int = hidden_size, num_layers: int = num_layers):
        super(Model, self).__init__()
        self.input_channels = input_channels
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Pad 3D input by 1 on each side (left,right,top,bottom,front,back)
        self.rep_pad3d = nn.ReplicationPad3d((1, 1, 1, 1, 1, 1))
        # Circular pad 2D for spatial wrapping behavior on each slice
        self.circ_pad2d = nn.CircularPad2d(1)
        # GRU to aggregate across the (padded) depth dimension; expects input size == channels
        self.gru = nn.GRU(input_size=self.input_channels, hidden_size=self.hidden_size,
                          num_layers=self.num_layers, batch_first=False, bidirectional=False)
        # Project hidden representation back to channel dimension for fusion
        self.project = nn.Linear(self.hidden_size, self.input_channels)
        # small epsilon for numeric stability in normalization
        self.eps = 1e-6

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, D, H, W)

        Returns:
            torch.Tensor: L2-normalized fused channel vectors per batch element of shape (B, C)
        """
        # Validate input dimensionality
        if x.dim() != 5:
            raise ValueError(f"Expected 5D input (B, C, D, H, W), got {x.dim()}D tensor")

        B, C, D, H, W = x.shape

        # 1) Replication 3D padding: shape -> (B, C, D+2, H+2, W+2)
        x_padded = self.rep_pad3d(x)

        # 2) Prepare slices for circular 2D padding:
        # Rearrange to (B, D_padded, C, H_padded, W_padded)
        Dp, Hp, Wp = x_padded.shape[2], x_padded.shape[3], x_padded.shape[4]
        x_slices = x_padded.permute(0, 2, 1, 3, 4).contiguous()  # (B, Dp, C, Hp, Wp)
        # Collapse batch and depth to treat every depth-slice as an independent image
        x_slices = x_slices.view(B * Dp, C, Hp, Wp)  # (B*Dp, C, Hp, Wp)

        # 3) Circular pad each slice on spatial dims
        x_circ = self.circ_pad2d(x_slices)  # (B*Dp, C, Hp+2, Wp+2)

        # 4) Spatial pooling per slice to get channel vectors: adaptive avg pool to (1,1)
        pooled = F.adaptive_avg_pool2d(x_circ, output_size=(1, 1)).view(B, Dp, C)  # (B, Dp, C)

        # 5) Form sequence across depth: GRU expects (seq_len, batch, input_size)
        seq = pooled.permute(1, 0, 2).contiguous()  # (Dp, B, C)

        # Pass through GRU
        gru_out, _h_n = self.gru(seq)  # gru_out: (Dp, B, hidden_size)

        # Take the last time-step's output as aggregated temporal descriptor
        last_out = gru_out[-1]  # (B, hidden_size)

        # 6) Project back to channel dimension and fuse with a global residual
        projected = self.project(last_out)  # (B, C)

        # Compute a global pooled residual from the (replication-padded) input
        global_residual = x_padded.mean(dim=(2, 3, 4))  # (B, C)

        # Fuse (additive) and apply a non-linearity
        fused = torch.tanh(projected + global_residual)  # (B, C)

        # 7) L2-normalize per sample across channel dimension
        norm = torch.norm(fused, p=2, dim=1, keepdim=True)  # (B, 1)
        normalized = fused / (norm + self.eps)  # (B, C)

        return normalized

def get_inputs():
    """
    Returns example input tensors matching the configured shapes.
    """
    x = torch.randn(batch_size, input_channels, depth, height, width)
    return [x]

def get_init_inputs():
    """
    Returns initialization arguments for the Model constructor:
    (input_channels, hidden_size, num_layers)
    """
    return [input_channels, hidden_size, num_layers]