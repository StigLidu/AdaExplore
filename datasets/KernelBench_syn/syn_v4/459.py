import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Complex 1D processing module that demonstrates a combination of:
      - Circular padding in 1D (nn.CircularPad1d)
      - Extraction of sliding blocks via nn.Unfold (on a 2D view)
      - Learned mixing of patch features with 1x1 Conv1d layers
      - Reconstruction with nn.Fold
      - Spatial replication padding (nn.ReplicationPad2d) and a simple height-wise aggregation

    The model takes an input of shape (batch, channels, length) and returns a tensor of the same shape.
    The internal pipeline:
      1. CircularPad1d to wrap boundaries
      2. Unsqueeze to 2D-like (height=1) and nn.Unfold to get sliding patches along length
      3. Two 1x1 Conv1d layers with a ReLU in between to transform patch features
      4. nn.Fold to stitch patches back to a padded signal
      5. ReplicationPad2d to expand height, then height-wise mean to merge spatial context
      6. Crop the padded signal back to the original length
    """
    def __init__(
        self,
        in_channels: int,
        kernel_size: int,
        stride: int,
        pad: int,
        intermediate_channels: int,
        input_length: int,
    ):
        """
        Args:
            in_channels (int): Number of input channels.
            kernel_size (int): Size of 1D sliding window.
            stride (int): Stride for sliding windows.
            pad (int): Amount of circular padding to add on both sides.
            intermediate_channels (int): Number of channels in the intermediate 1x1 Conv1d.
            input_length (int): Original length of the input signal (before padding).
        """
        super(Model, self).__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad = pad
        self.input_length = input_length

        # Padders
        self.circ_pad = nn.CircularPad1d(pad)
        # We'll pad height by replication after folding; left/right top/bottom ordering:
        # (pad_left, pad_right, pad_top, pad_bottom)
        # We only want to replicate in height dimension (top and bottom), keep left/right zero.
        self.rep_pad = nn.ReplicationPad2d((0, 0, 1, 1))

        # Convert to 2D -> use Unfold/Fold with kernel_size (1, kernel_size)
        self.unfold = nn.Unfold(kernel_size=(1, kernel_size), stride=(1, stride))

        # Output size after padding (height=1, width = input_length + 2*pad)
        self.padded_length = input_length + 2 * pad
        self.fold = nn.Fold(output_size=(1, self.padded_length),
                            kernel_size=(1, kernel_size), stride=(1, stride))

        # 1x1 Conv1d layers to mix patch features across channels.
        # Unfold produces channels = in_channels * kernel_size
        in_patch_channels = in_channels * kernel_size
        self.conv1 = nn.Conv1d(in_patch_channels, intermediate_channels, kernel_size=1)
        self.conv2 = nn.Conv1d(intermediate_channels, in_patch_channels, kernel_size=1)

        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, channels, length)

        Returns:
            torch.Tensor: Output tensor of shape (batch, channels, length)
        """
        # x: (B, C, L)
        if x.dim() != 3:
            raise ValueError("Input must be 3D tensor (B, C, L)")

        # 1) Circular pad along length -> (B, C, L + 2*pad)
        x_padded = self.circ_pad(x)

        # 2) View as 2D (height=1, width=padded_length) to use Unfold
        x_2d = x_padded.unsqueeze(2)  # (B, C, 1, W)
        patches = self.unfold(x_2d)   # (B, C * kernel_size, L_out)

        # 3) Mix patch features with small MLP implemented as 1x1 Conv1d
        y = self.conv1(patches)       # (B, intermediate, L_out)
        y = self.act(y)
        y = self.conv2(y)             # (B, C * kernel_size, L_out)

        # 4) Fold back to padded 2D image then squeeze height
        folded = self.fold(y)         # (B, C, 1, padded_length)
        folded = folded.squeeze(2)    # (B, C, padded_length)

        # 5) Use ReplicationPad2d to create a small height dimension, then aggregate
        folded_2d = folded.unsqueeze(2)  # (B, C, 1, padded_length)
        rep_padded = self.rep_pad(folded_2d)  # (B, C, 3, padded_length) if top/bottom=1
        # Aggregate height dimension (simple mean) to fuse replicated context
        rep_agg = rep_padded.mean(dim=2)  # (B, C, padded_length)

        # 6) Crop back to original length (remove circular padding)
        start = self.pad
        end = start + self.input_length
        out = rep_agg[:, :, start:end]  # (B, C, input_length)

        return out


# Module-level configuration variables
batch_size = 8
channels = 3
length = 512
kernel_size = 7
stride = 3
pad = 3
intermediate_channels = 64

def get_inputs():
    """
    Returns a list with the primary input tensor(s) for the model.
    Single input: a random tensor shaped (batch_size, channels, length).
    """
    x = torch.randn(batch_size, channels, length)
    return [x]

def get_init_inputs():
    """
    Returns initialization parameters for the Model __init__:
      [in_channels, kernel_size, stride, pad, intermediate_channels, input_length]
    """
    return [channels, kernel_size, stride, pad, intermediate_channels, length]