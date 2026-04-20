import torch
import torch.nn as nn

# Configuration
batch_size = 8
in_channels = 3
mid_channels = 64      # number of channels after ConvTranspose2d
out_features = 512     # final feature dimension per patch
input_H = 16
input_W = 16
deconv_kernel = 3
deconv_stride = 2
deconv_padding = 1
deconv_output_padding = 1
pool_output_size = (4, 4)   # fixed pooled spatial size -> 16 patches
softshrink_lambda = 0.5

class Model(nn.Module):
    """
    Complex model combining a 2D transposed convolution (upsample),
    Softshrink nonlinearity, spatial pooling, a linear projection across
    channels for each spatial patch, and a LogSoftmax over output features.

    Forward pass summary:
      x: (N, C_in, H, W)
      -> deconv -> (N, C_mid, H*2, W*2)
      -> softshrink -> (N, C_mid, H*2, W*2)
      -> adaptive avg pool to (4,4) -> (N, C_mid, 4, 4)
      -> reshape to patches (N, P, C_mid) where P=16
      -> linear across channels -> (N, P, out_features)
      -> log_softmax over out_features -> (N, P, out_features)
      -> mean over patches -> (N, out_features)
    """
    def __init__(
        self,
        in_ch: int = in_channels,
        mid_ch: int = mid_channels,
        out_feats: int = out_features,
        kernel_size: int = deconv_kernel,
        stride: int = deconv_stride,
        padding: int = deconv_padding,
        output_padding: int = deconv_output_padding,
        pool_size: tuple = pool_output_size,
        softshrink_lambda_val: float = softshrink_lambda
    ):
        super(Model, self).__init__()
        # Transposed convolution to increase spatial resolution and map channels
        self.deconv = nn.ConvTranspose2d(
            in_channels=in_ch,
            out_channels=mid_ch,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            bias=True
        )
        # Non-linear shrinkage
        self.softshrink = nn.Softshrink(lambd=softshrink_lambda_val)
        # Reduce/regularize spatial dimensions to a fixed small grid
        self.pool = nn.AdaptiveAvgPool2d(pool_size)
        # Linear projection from channels -> out_features applied per patch
        self.linear = nn.Linear(mid_ch, out_feats, bias=True)
        # Normalize into log-probabilities over output features
        self.logsoftmax = nn.LogSoftmax(dim=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape (N, C_in, H, W)

        Returns:
            torch.Tensor: Output tensor of shape (N, out_features).
                          This is obtained by aggregating log-softmaxed features
                          across spatial patches (mean over patches).
        """
        # Upsample & change channel dimension
        y = self.deconv(x)                         # (N, mid_ch, H*2, W*2)
        # Apply sparse-inducing nonlinearity
        y = self.softshrink(y)                     # (N, mid_ch, H*2, W*2)
        # Fixed-size spatial grid for stable parameter shapes
        y = self.pool(y)                           # (N, mid_ch, P_H, P_W), P_H*P_W = P
        N, C, PH, PW = y.shape
        P = PH * PW
        # Reshape to (N, P, C) so linear projects channels -> out_features per patch
        y_patches = y.view(N, C, P).permute(0, 2, 1)   # (N, P, C)
        # Linear projection per patch
        feats = self.linear(y_patches)              # (N, P, out_features)
        # Convert to log-probabilities across the feature dimension
        logp = self.logsoftmax(feats)               # (N, P, out_features)
        # Aggregate across patches (mean of log-probabilities)
        out = logp.mean(dim=1)                      # (N, out_features)
        return out

def get_inputs():
    """
    Returns:
        List containing a single input tensor x with shape (batch_size, in_channels, input_H, input_W)
    """
    x = torch.randn(batch_size, in_channels, input_H, input_W)
    return [x]

def get_init_inputs():
    """
    Returns initialization parameters for constructing the Model.
    Order corresponds to the Model __init__ signature defaults used above.
    """
    return [
        in_channels,         # in_ch
        mid_channels,        # mid_ch
        out_features,        # out_feats
        deconv_kernel,       # kernel_size
        deconv_stride,       # stride
        deconv_padding,      # padding
        deconv_output_padding,# output_padding
        pool_output_size,    # pool_size
        softshrink_lambda    # softshrink lambda
    ]