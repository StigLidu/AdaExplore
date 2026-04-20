import torch
import torch.nn as nn

# Configuration variables
batch_size = 8
in_channels = 3
mid_channels = 16
out_channels = 3
length = 4096
pad = 2  # circular padding on each side
kernel_size_conv1 = 3  # for Conv1d

class Model(nn.Module):
    """
    Complex 1D-to-1D module that demonstrates use of CircularPad1d, PReLU,
    MaxPool2d and MaxUnpool2d by reshaping a sequence into a 2D tensor for pooling.
    
    Pipeline:
    - Circularly pad the input sequence (CircularPad1d)
    - Apply a 1D convolution
    - Apply PReLU activation
    - Temporarily view features as (N, C, H=1, W) and apply MaxPool2d (with indices)
    - Restore (unpool) using MaxUnpool2d
    - Add a residual connection from the pre-pooled features
    - Final pointwise Conv1d projection to output channels
    """
    def __init__(
        self,
        in_ch: int = in_channels,
        mid_ch: int = mid_channels,
        out_ch: int = out_channels,
        pad_amt: int = pad,
        ksize: int = kernel_size_conv1,
    ):
        super(Model, self).__init__()
        self.pad = nn.CircularPad1d(pad_amt)         # pads last dim circularly
        # Conv1d after circular padding (no additional padding)
        self.conv1 = nn.Conv1d(in_ch, mid_ch, kernel_size=ksize, stride=1, padding=0, bias=True)
        # Parametric ReLU with one parameter per channel
        self.prelu = nn.PReLU(num_parameters=mid_ch)
        # Use 2D pooling across width by reshaping to (N, C, 1, L)
        self.pool = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2), return_indices=True)
        # Unpool to reverse the MaxPool2d operation
        self.unpool = nn.MaxUnpool2d(kernel_size=(1, 2), stride=(1, 2))
        # Final pointwise convolution to project back to desired output channels
        self.conv_proj = nn.Conv1d(mid_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (N, C_in, L)

        Returns:
            torch.Tensor: Output tensor of shape (N, C_out, L_out)
        """
        # Circular pad along the sequence dimension
        x_padded = self.pad(x)  # (N, C_in, L + 2*pad)
        # Convolution
        conv_feat = self.conv1(x_padded)  # (N, mid_ch, L')
        # Non-linearity
        activated = self.prelu(conv_feat)  # (N, mid_ch, L')
        # Prepare for 2D pooling by adding a height dimension of 1
        feat_2d = activated.unsqueeze(2)  # (N, mid_ch, 1, L')
        # MaxPool2d returns pooled features and indices needed for unpool
        pooled, indices = self.pool(feat_2d)  # pooled: (N, mid_ch, 1, L''), indices same shape
        # Unpool back to original 2D shape (should match feat_2d's spatial dims)
        unpooled = self.unpool(pooled, indices, output_size=feat_2d.size())  # (N, mid_ch, 1, L')
        # Remove the temporary height dim
        unpooled_squeezed = unpooled.squeeze(2)  # (N, mid_ch, L')
        # Residual connection: combine unpooled features with the original conv features
        combined = unpooled_squeezed + conv_feat  # (N, mid_ch, L')
        # Final projection to output channels
        out = self.conv_proj(combined)  # (N, out_ch, L')
        return out

def get_inputs():
    """
    Generates a random input tensor that fits the model's expected shape.

    Returns:
        list: [x] where x is a torch.Tensor of shape (batch_size, in_channels, length)
    """
    x = torch.randn(batch_size, in_channels, length)
    return [x]

def get_init_inputs():
    """
    Returns initialization parameters for constructing the model externally if needed.
    """
    return [in_channels, mid_channels, out_channels, pad, kernel_size_conv1]