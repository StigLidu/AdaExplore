import torch
import torch.nn as nn
import torch

# Configuration variables
batch_size = 8
in_channels = 32
height = 16
width = 8
upsample_scale = 2  # integer scale factor for nearest neighbor upsampling
dropout_prob = 0.25

# Lazy ConvTranspose1d configuration
deconv_out_channels = 64
deconv_kernel = 5
deconv_stride = 2
deconv_padding = 2
deconv_output_padding = 1
deconv_dilation = 1

class Model(nn.Module):
    """
    Composite model that demonstrates a mixed 2D -> 1D -> 2D processing pipeline:
      1. Nearest-neighbor upsampling (2D)
      2. Channel-wise dropout (Dropout2d)
      3. Collapse spatial dims into a 1D sequence and apply LazyConvTranspose1d (learns in_channels on first call)
      4. Non-linearity and attempt to reshape back to a 2D grid (trimming if necessary)
    This pattern exercises interplay between spatial and sequential representations and uses a lazy module
    to permit flexible input channel sizes discovered at runtime.
    """
    def __init__(self,
                 upsample_scale: int = upsample_scale,
                 dropout_p: float = dropout_prob,
                 deconv_out_ch: int = deconv_out_channels,
                 deconv_kernel_size: int = deconv_kernel,
                 deconv_stride: int = deconv_stride,
                 deconv_padding: int = deconv_padding,
                 deconv_output_padding: int = deconv_output_padding,
                 deconv_dilation: int = deconv_dilation):
        super(Model, self).__init__()
        # 1) 2D nearest neighbor upsampling
        self.upsample = nn.UpsamplingNearest2d(scale_factor=upsample_scale)
        # 2) Drop entire channels with Dropout2d
        self.dropout2d = nn.Dropout2d(p=dropout_p)
        # 3) Lazy ConvTranspose1d - will infer in_channels at first forward pass
        self.deconv1d = nn.LazyConvTranspose1d(
            out_channels=deconv_out_ch,
            kernel_size=deconv_kernel_size,
            stride=deconv_stride,
            padding=deconv_padding,
            output_padding=deconv_output_padding,
            dilation=deconv_dilation,
            bias=True
        )
        # 4) Non-linearity
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass:
        x: (B, C, H, W)
        Steps:
          - Upsample to (B, C, H', W')
          - Apply Dropout2d (channel dropout)
          - Collapse spatial dims into a 1D sequence: (B, C, H'*W')
          - Apply ConvTranspose1d -> (B, out_ch, L_out)
          - ReLU
          - If possible, reshape back to 4D (B, out_ch, H', W_out) by trimming any remainder on the sequence length.
            If trimming would remove all elements, return the 3D (B, out_ch, L_out) tensor instead.
        """
        B, C, H, W = x.shape
        # 1) Upsample spatially
        x_up = self.upsample(x)  # (B, C, H*scale, W*scale)
        _, _, H_up, W_up = x_up.shape

        # 2) Channel-dropout
        x_dropped = self.dropout2d(x_up)  # same shape as x_up

        # 3) Collapse spatial dims into a 1D width for ConvTranspose1d
        # result shape: (B, C, H_up * W_up)
        seq_len = H_up * W_up
        x_seq = x_dropped.contiguous().view(B, C, seq_len)

        # 4) Apply Lazy ConvTranspose1d (in_channels inferred on first call)
        y = self.deconv1d(x_seq)  # (B, out_channels, L_out)
        y = self.act(y)

        # 5) Attempt to reshape back to 2D grid using H_up as one spatial dimension.
        #    Compute how many full columns we can form: new_W = floor(L_out / H_up)
        L_out = y.size(2)
        if H_up > 0:
            new_W = L_out // H_up
        else:
            new_W = 0

        if new_W >= 1:
            new_L = new_W * H_up
            # Trim the sequence to a multiple of H_up and reshape to (B, out_ch, H_up, new_W)
            y_trimmed = y[:, :, :new_L]
            y_reshaped = y_trimmed.contiguous().view(B, y_trimmed.size(1), H_up, new_W)
            return y_reshaped
        else:
            # Cannot reshape sensibly, return the 3D result (B, out_channels, L_out)
            return y

# Input dimensions for test/example generation
batch_size = batch_size
channels = in_channels
H = height
W = width

def get_inputs():
    """
    Returns a single input tensor shaped (batch_size, channels, H, W)
    """
    x = torch.randn(batch_size, channels, H, W)
    return [x]

def get_init_inputs():
    """
    Returns initialization parameters for the Model constructor to allow recreation:
    [upsample_scale, dropout_prob, deconv_out_channels, deconv_kernel, deconv_stride, deconv_padding, deconv_output_padding, deconv_dilation]
    """
    return [upsample_scale, dropout_prob, deconv_out_channels, deconv_kernel, deconv_stride, deconv_padding, deconv_output_padding, deconv_dilation]