import torch
import torch.nn as nn

# Configuration
BATCH_SIZE = 8
INPUT_LENGTH = 16384
OUT_CHANNELS = 64
CONV_KERNEL_SIZE = 15
PAD_TOP_BOTTOM = 3  # number of rows to circularly pad on H dimension after reshaping
FC_HIDDEN = 32

class Model(nn.Module):
    """
    A composite module that demonstrates a mixed 1D/2D processing pattern using:
      - nn.LazyConv1d: lazily initialized 1D convolution (infers in_channels on first forward)
      - nn.CircularPad2d: circular padding on a reshaped 4D view (treating time as H)
      - nn.LazyBatchNorm2d: lazily initialized 2D batchnorm (infers num_features on first forward)

    Computation pipeline:
      1. Accepts a 1D temporal signal (N, L) or (N, C_in, L). If needed, unsqueezes to (N, 1, L).
      2. Applies a LazyConv1d -> ReLU.
      3. Reshapes to 4D (N, C, H=L_out, W=1) to apply circular 2D padding along H.
      4. Applies LazyBatchNorm2d on the padded 4D tensor.
      5. Squeezes back to 3D and performs global average pooling over the time dimension.
      6. Two-layer MLP (Linear -> ReLU -> Linear) to produce final per-sample outputs of shape (N, 1).
    """
    def __init__(self):
        super(Model, self).__init__()
        # LazyConv1d will infer in_channels at first forward
        self.conv = nn.LazyConv1d(out_channels=OUT_CHANNELS, kernel_size=CONV_KERNEL_SIZE, bias=True)
        # CircularPad2d pads on (left, right, top, bottom); we only pad top/bottom (time dimension)
        # After conv output will be reshaped to (N, C, H, W=1), so top/bottom refer to H.
        self.pad2d = nn.CircularPad2d((0, 0, PAD_TOP_BOTTOM, PAD_TOP_BOTTOM))
        # LazyBatchNorm2d will infer num_features on first forward from C
        self.bn2d = nn.LazyBatchNorm2d()
        self.relu = nn.ReLU(inplace=True)
        # Simple two-layer MLP after global pooling
        self.fc1 = nn.Linear(OUT_CHANNELS, FC_HIDDEN)
        self.fc2 = nn.Linear(FC_HIDDEN, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor, either of shape (N, L) or (N, C_in, L). If 2D, assumes single channel.

        Returns:
            Tensor of shape (N, 1): one scalar per sample after the composed pipeline.
        """
        # Normalize input shape to (N, C_in, L)
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (N, 1, L)

        # 1) LazyConv1d -> ReLU
        x = self.conv(x)  # (N, C_out, L_out)
        x = self.relu(x)

        # 2) Reshape to 4D to apply circular padding along the temporal axis
        x = x.unsqueeze(-1)  # (N, C_out, H=L_out, W=1)
        x = self.pad2d(x)    # circularly pad top/bottom on H
        x = self.bn2d(x)     # (N, C_out, H_padded, W=1)
        x = x.squeeze(-1)    # (N, C_out, H_padded)

        # 3) Global average pooling across the temporal dimension
        x = x.mean(dim=2)    # (N, C_out)

        # 4) Two-layer MLP for final prediction
        x = self.fc1(x)      # (N, FC_HIDDEN)
        x = self.relu(x)
        x = self.fc2(x)      # (N, 1)

        return x

def get_inputs():
    """
    Generate example inputs for the model.

    Returns:
        A list containing a single input tensor of shape (BATCH_SIZE, INPUT_LENGTH).
    """
    x = torch.randn(BATCH_SIZE, INPUT_LENGTH)
    return [x]

def get_init_inputs():
    """
    No special initialization inputs are required (lazy layers infer shapes on first forward).
    """
    return []