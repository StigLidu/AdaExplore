import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Complex module that transforms a 1D sequence into a 2D feature map, applies instance normalization,
    nonlinearities and spatial pooling, then fuses global statistics from the original sequence and
    pooled features to produce a compact output vector.

    Computation steps:
    1. 1D convolution over the input sequence to produce channel-wise features.
    2. InstanceNorm1d over the conv output and Softsign activation.
    3. Reshape the sequence into a 2D feature map (H x W).
    4. MaxPool2d spatial pooling over the 2D map.
    5. Global averaging of pooled spatial features and global averaging of the original sequence.
    6. Concatenate these statistics and apply a final linear layer followed by Softsign.
    """
    def __init__(
        self,
        in_channels: int,
        conv_out_channels: int,
        conv_kernel_size: int,
        H: int,
        W: int,
        pool_kernel: int,
        pool_stride: int,
        pool_padding: int,
        fc_out: int,
    ):
        """
        Initializes layers and stores shape parameters.

        Args:
            in_channels (int): Number of channels in the input sequence.
            conv_out_channels (int): Number of output channels for the Conv1d layer.
            conv_kernel_size (int): Kernel size for the Conv1d.
            H (int): Height for reshaping the sequence into 2D feature maps.
            W (int): Width for reshaping the sequence into 2D feature maps.
            pool_kernel (int): Kernel size for MaxPool2d.
            pool_stride (int): Stride for MaxPool2d.
            pool_padding (int): Padding for MaxPool2d.
            fc_out (int): Output feature dimension from the final fully-connected layer.
        """
        super(Model, self).__init__()

        assert H * W > 0, "H and W must be positive integers"
        self.in_channels = in_channels
        self.conv_out_channels = conv_out_channels
        self.H = H
        self.W = W

        # 1D convolution: keep sequence length by using symmetric padding
        conv_padding = conv_kernel_size // 2
        self.conv1d = nn.Conv1d(in_channels, conv_out_channels, kernel_size=conv_kernel_size, padding=conv_padding)

        # Instance normalization over channel dimension for 1D features
        self.inst_norm = nn.InstanceNorm1d(conv_out_channels, affine=False)

        # Softsign nonlinearity as a module
        self.softsign = nn.Softsign()

        # 2D max pooling operating on the reshaped 2D feature maps
        self.maxpool2d = nn.MaxPool2d(kernel_size=pool_kernel, stride=pool_stride, padding=pool_padding)

        # Final fully connected layer to fuse pooled spatial features and original sequence statistics
        # Input size = conv_out_channels (pooled spatial averages) + in_channels (original sequence channel averages)
        self.fc = nn.Linear(conv_out_channels + in_channels, fc_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, seq_len),
                              where seq_len must equal H * W.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, fc_out).
        """
        # Validate input shape
        batch_size, c, seq_len = x.shape
        expected_seq = self.H * self.W
        if seq_len != expected_seq:
            raise ValueError(f"Input sequence length {seq_len} does not match expected H*W = {expected_seq}")

        # 1) Conv1d -> (B, conv_out_channels, seq_len)
        conv_out = self.conv1d(x)

        # 2) InstanceNorm1d and Softsign nonlinearity
        normed = self.inst_norm(conv_out)
        activated = self.softsign(normed)

        # 3) Reshape into 2D feature maps (B, conv_out_channels, H, W)
        feat2d = activated.view(batch_size, self.conv_out_channels, self.H, self.W)

        # 4) MaxPool2d over spatial dims
        pooled = self.maxpool2d(feat2d)

        # 5) Global spatial average of pooled features -> (B, conv_out_channels)
        pooled_mean = pooled.mean(dim=(2, 3))

        # 6) Global average of original sequence across the temporal dimension -> (B, in_channels)
        seq_mean = x.mean(dim=2)

        # 7) Concatenate pooled spatial statistics with original sequence statistics
        fused = torch.cat([pooled_mean, seq_mean], dim=1)

        # 8) Final linear projection and Softsign activation
        out = self.fc(fused)
        out = self.softsign(out)

        return out

# Module-level configuration
batch_size = 8
in_channels = 32
conv_out_channels = 64
H = 16
W = 16
seq_len = H * W  # required sequence length to reshape into H x W
conv_kernel_size = 3
pool_kernel = 2
pool_stride = 2
pool_padding = 0
fc_out = 128

def get_inputs():
    """
    Returns a list containing a single input tensor of shape (batch_size, in_channels, seq_len).
    The sequence length is H * W to enable reshaping into 2D feature maps.
    """
    x = torch.randn(batch_size, in_channels, seq_len)
    return [x]

def get_init_inputs():
    """
    Returns initialization parameters to construct the Model:
    [in_channels, conv_out_channels, conv_kernel_size, H, W, pool_kernel, pool_stride, pool_padding, fc_out]
    """
    return [in_channels, conv_out_channels, conv_kernel_size, H, W, pool_kernel, pool_stride, pool_padding, fc_out]