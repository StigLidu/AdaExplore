import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Complex model combining AdaptiveAvgPool3d, AdaptiveAvgPool2d, and GRUCell.

    The model ingests a 5D video-like tensor of shape (B, C, D, H, W).
    - First, an AdaptiveAvgPool3d reduces (D, H, W) to a coarser 3D grid (D_out, H3, W3).
    - For each temporal slice along D_out we apply an AdaptiveAvgPool2d to further reduce spatial resolution to (H2, W2).
    - Each pooled slice is flattened and fed into a GRUCell sequentially across the D_out dimension.
    - The final GRU hidden state is projected to an output vector.

    This creates a spatio-temporal pooling + recurrent processing pipeline.
    """
    def __init__(
        self,
        in_channels: int,
        hidden_size: int,
        d_out: int,
        h3: int,
        w3: int,
        h2: int,
        w2: int,
        output_size: int
    ):
        super(Model, self).__init__()
        # 3D pooling to reduce depth and coarse spatial dims
        self.pool3d = nn.AdaptiveAvgPool3d((d_out, h3, w3))
        # 2D pooling applied per temporal slice to reduce spatial dims further
        self.pool2d = nn.AdaptiveAvgPool2d((h2, w2))
        # GRUCell processes each temporal slice (after pooling & flatten)
        self.input_size = in_channels * h2 * w2
        self.hidden_size = hidden_size
        self.gru_cell = nn.GRUCell(self.input_size, hidden_size)
        # Final projection from hidden state to desired output dimension
        self.out_proj = nn.Linear(hidden_size, output_size)

        # Save configuration for potential external use
        self.d_out = d_out
        self.h3 = h3
        self.w3 = w3
        self.h2 = h2
        self.w2 = w2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Tensor of shape (B, C, D, H, W)

        Returns:
            Tensor of shape (B, output_size) - projection of the final GRU hidden state.
        """
        # x -> (B, C, D_out, H3, W3)
        x_pooled3d = self.pool3d(x)

        # Reorder to sequence-first: (D_out, B, C, H3, W3)
        seq = x_pooled3d.permute(2, 0, 1, 3, 4)

        batch_size = x.size(0)
        device = x.device
        # Initialize hidden state (B, hidden_size)
        h = torch.zeros(batch_size, self.hidden_size, device=device, dtype=x.dtype)

        # Iterate over temporal dimension D_out
        for t in range(self.d_out):
            # slice -> (B, C, H3, W3)
            slice_t = seq[t]
            # further spatial reduction -> (B, C, H2, W2)
            slice_t_pooled2d = self.pool2d(slice_t)
            # flatten to (B, input_size)
            inp = slice_t_pooled2d.reshape(batch_size, -1)
            # GRUCell step
            h = self.gru_cell(inp, h)

        # Project final hidden state to output
        out = self.out_proj(h)  # (B, output_size)
        return out

# Configuration / default sizes for test inputs
BATCH = 4
IN_CHANNELS = 32
DEPTH = 16
HEIGHT = 64
WIDTH = 64

# Pooling and recurrent sizes
D_OUT = 8    # temporal pooling target
H3 = 8       # coarse spatial H after 3D pool
W3 = 8       # coarse spatial W after 3D pool
H2 = 4       # fine spatial H after 2D pool
W2 = 4       # fine spatial W after 2D pool

HIDDEN_SIZE = 512
OUTPUT_SIZE = 128

def get_inputs():
    """
    Generates a sample 5D input tensor representing a batch of video-like data.

    Returns:
        list: [x] where x has shape (BATCH, IN_CHANNELS, DEPTH, HEIGHT, WIDTH)
    """
    x = torch.randn(BATCH, IN_CHANNELS, DEPTH, HEIGHT, WIDTH)
    return [x]

def get_init_inputs():
    """
    Returns initialization parameters required to construct the Model.

    Returns:
        list: [in_channels, hidden_size, d_out, h3, w3, h2, w2, output_size]
    """
    return [IN_CHANNELS, HIDDEN_SIZE, D_OUT, H3, W3, H2, W2, OUTPUT_SIZE]