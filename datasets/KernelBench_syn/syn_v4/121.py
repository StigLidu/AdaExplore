import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Complex model that combines a 3D convolution (lazy-initialized), spatial pooling,
    a multi-layer bidirectional GRU across the temporal/depth dimension, and PReLU activations.
    
    Input shape: (batch, in_channels, time, height, width)
    Pipeline:
      - LazyConv3d (3D conv across time+spatial dims)
      - Channel-wise PReLU
      - Adaptive average pool to collapse spatial HxW -> 1x1 while keeping time dimension
      - Permute to (time, batch, features) and feed through a multi-layer bidirectional GRU
      - Take the final GRU output (last time step), project with a Linear layer and a final PReLU
    """
    def __init__(
        self,
        conv_out_channels: int = 32,
        gru_hidden_size: int = 64,
        gru_num_layers: int = 2,
        gru_bidirectional: bool = True,
        fc_out_features: int = 128,
    ):
        super(Model, self).__init__()
        # LazyConv3d will infer in_channels on first forward
        self.conv = nn.LazyConv3d(
            out_channels=conv_out_channels,
            kernel_size=(3, 3, 3),
            padding=(1, 1, 1),
            bias=False
        )
        # PReLU applied per-channel after convolution
        self.prelu1 = nn.PReLU(num_parameters=conv_out_channels)
        
        # GRU processes the temporal/depth dimension: input_size will be conv_out_channels
        self.gru = nn.GRU(
            input_size=conv_out_channels,
            hidden_size=gru_hidden_size,
            num_layers=gru_num_layers,
            bidirectional=gru_bidirectional,
            dropout=0.0,
        )
        
        # Determine the feature size coming out of the GRU (accounts for bidirectionality)
        self._gru_output_size = gru_hidden_size * (2 if gru_bidirectional else 1)
        
        # Final projection
        self.fc = nn.Linear(self._gru_output_size, fc_out_features)
        self.prelu2 = nn.PReLU(num_parameters=fc_out_features)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input tensor with shape (B, C_in, T, H, W)
        
        Returns:
            torch.Tensor: Output tensor of shape (B, fc_out_features)
        """
        # 1) 3D convolution across (T,H,W)
        x = self.conv(x)  # -> (B, C_out, T, H, W)
        
        # 2) Channel-wise nonlinearity
        x = self.prelu1(x)
        
        # 3) Collapse spatial HxW to 1x1 while preserving depth/time dimension T
        # Use adaptive_avg_pool3d with D_out = current T to preserve time steps
        # Note: x.size(2) is the current depth/time dimension
        x = F.adaptive_avg_pool3d(x, output_size=(x.size(2), 1, 1))  # -> (B, C_out, T, 1, 1)
        x = x.squeeze(-1).squeeze(-1)  # -> (B, C_out, T)
        
        # 4) Permute to GRU expected shape: (seq_len=T, batch=B, input_size=C_out)
        x = x.permute(2, 0, 1)  # -> (T, B, C_out)
        
        # 5) GRU across the temporal dimension
        gru_out, _ = self.gru(x)  # -> (T, B, num_directions * hidden_size)
        
        # 6) Take the last time-step's output as a summary representation
        last = gru_out[-1]  # -> (B, num_directions * hidden_size)
        
        # 7) Final projection + activation
        out = self.fc(last)  # -> (B, fc_out_features)
        out = self.prelu2(out)
        
        return out

# Module-level configuration variables
batch_size = 8
in_channels = 3
time = 16
height = 32
width = 32
conv_out_channels = 32
gru_hidden_size = 64
gru_num_layers = 2
gru_bidirectional = True
fc_out_features = 128

def get_inputs():
    """
    Generates a single input tensor matching the expected shape:
    (batch_size, in_channels, time, height, width)
    """
    x = torch.randn(batch_size, in_channels, time, height, width)
    return [x]

def get_init_inputs():
    """
    No special initialization inputs required (LazyConv3d initializes on first forward).
    """
    return []