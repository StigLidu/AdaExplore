import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Any, Tuple

# Configuration
BATCH_SIZE = 8
IN_CHANNELS = 3
OUT_CHANNELS = 16
DEPTH = 10
HEIGHT = 32
WIDTH = 32
KERNEL_SIZE = 3  # cubic kernel for Conv3d
DROPOUT_P = 0.1
# ZeroPad3d expects (pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back)
PAD = (1, 1, 1, 1, 1, 1)
EPS = 1e-6


class Model(nn.Module):
    """
    Complex 3D feature extractor that combines ZeroPad3d, Conv3d, AlphaDropout,
    a residual projection and global pooling with normalization.

    Forward computation pattern:
      1. Zero-pad the input to control spatial dimensions.
      2. Apply a 3D convolution.
      3. Apply GELU activation.
      4. Apply Alpha Dropout for regularization.
      5. Add a residual connection (with 1x1x1 projection if channel dims differ).
      6. Global average pool over depth/height/width.
      7. L2-normalize per batch sample and return the feature matrix.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        pad: Tuple[int, int, int, int, int, int] = (0, 0, 0, 0, 0, 0),
        dropout_p: float = 0.1,
    ):
        super(Model, self).__init__()
        # Padding layer to explicitly control boundary behavior
        self.pad = nn.ZeroPad3d(pad)

        # Primary 3D convolution (we pad explicitly so set conv padding=0)
        self.conv = nn.Conv3d(
            in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=0, bias=True
        )

        # 1x1x1 projection for residual path when channel dims differ
        if in_channels != out_channels:
            self.res_proj = nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False)
        else:
            self.res_proj = nn.Identity()

        # AlphaDropout to preserve self-normalizing properties for SELU-like behavior
        self.dropout = nn.AlphaDropout(p=dropout_p)

        # Small learnable scale applied after normalization (optional)
        self.post_scale = nn.Parameter(torch.ones(out_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, channels, D, H, W)
        Returns:
            Tensor of shape (batch, out_channels) which is L2-normalized per sample.
        """
        # 1. Explicit zero padding to adjust receptive field
        x_padded = self.pad(x)

        # 2. 3D convolution
        conv_out = self.conv(x_padded)

        # 3. Non-linear activation
        act = F.gelu(conv_out)

        # 4. Alpha Dropout (keeps mean/variance for SELU-compatible networks)
        dropped = self.dropout(act)

        # 5. Residual connection (project input to out_channels if necessary)
        res = self.res_proj(x)
        # Note: res may have different spatial dimensions due to padding/conv; center-crop or adapt
        # To keep things simple and robust, slice or pad res to match dropped spatial dims.
        # Compute spatial dims
        _, _, d_r, h_r, w_r = res.shape
        _, _, d_o, h_o, w_o = dropped.shape

        # If shapes differ, center-crop or pad res to match dropped
        if (d_r, h_r, w_r) != (d_o, h_o, w_o):
            # Center-crop or pad along each spatial dimension
            def match_dim(t: torch.Tensor, target_size: Tuple[int, int, int]) -> torch.Tensor:
                _, _, D1, H1, W1 = t.shape
                Dt, Ht, Wt = target_size
                # Crop if larger
                sd = max((D1 - Dt) // 2, 0)
                sh = max((H1 - Ht) // 2, 0)
                sw = max((W1 - Wt) // 2, 0)
                ed = sd + Dt
                eh = sh + Ht
                ew = sw + Wt
                t_cropped = t[:, :, sd:ed, sh:eh, sw:ew]
                # If smaller, pad evenly
                pd = Dt - t_cropped.shape[2]
                ph = Ht - t_cropped.shape[3]
                pw = Wt - t_cropped.shape[4]
                if pd > 0 or ph > 0 or pw > 0:
                    pad_left = pw // 2
                    pad_right = pw - pad_left
                    pad_top = ph // 2
                    pad_bottom = ph - pad_top
                    pad_front = pd // 2
                    pad_back = pd - pad_front
                    t_cropped = F.pad(t_cropped, (pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back))
                return t_cropped

            res_matched = match_dim(res, (d_o, h_o, w_o))
        else:
            res_matched = res

        out = dropped + res_matched

        # 6. Global average pooling over spatial dims (D, H, W)
        pooled = out.mean(dim=(2, 3, 4))  # shape: (batch, out_channels)

        # 7. L2-normalize per sample and apply small learnable scale
        norm = torch.norm(pooled, p=2, dim=1, keepdim=True)
        normalized = pooled / (norm + EPS)
        scaled = normalized * self.post_scale

        return scaled


def get_inputs() -> List[torch.Tensor]:
    """
    Returns a list with a single input tensor matching the model expectations:
      shape = (BATCH_SIZE, IN_CHANNELS, DEPTH, HEIGHT, WIDTH)
    """
    x = torch.randn(BATCH_SIZE, IN_CHANNELS, DEPTH, HEIGHT, WIDTH)
    return [x]


def get_init_inputs() -> List[Any]:
    """
    Returns the initialization parameters required to construct the Model.
    """
    return [IN_CHANNELS, OUT_CHANNELS, KERNEL_SIZE, PAD, DROPOUT_P]