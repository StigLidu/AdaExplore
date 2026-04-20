import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Complex 3D processing module that:
    - Applies ReflectionPad3d to prepare for a 3D convolution.
    - Performs a learned 3D convolution (provided weights/bias at init).
    - Applies LocalResponseNorm for channel-wise normalization.
    - Reshapes spatial width into a 1D signal and applies CircularPad1d + 1D convolution
      to perform a wrapped (circular) smoothing along the width dimension.
    - Uses a small gating mechanism (global average -> sigmoid) to reweight channels.
    - Final nonlinearity: Swish (x * sigmoid(x)).

    Forward signature:
        x: Tensor[B, C_in, D, H, W]

    Initialization expects explicit tensors for the learned conv3d weights and for the 1D kernel.
    """
    def __init__(
        self,
        conv3d_weight: torch.Tensor,
        conv3d_bias: torch.Tensor,
        conv1d_weight: torch.Tensor,
        lrn_size: int = 5,
        lrn_alpha: float = 1e-4,
        lrn_beta: float = 0.75,
        lrn_k: float = 1.0,
    ):
        super(Model, self).__init__()

        # Validate shapes
        assert conv3d_weight.ndim == 5, "conv3d_weight must be (out_ch, in_ch, kd, kh, kw)"
        assert conv3d_bias.ndim == 1 and conv3d_bias.shape[0] == conv3d_weight.shape[0]
        assert conv1d_weight.ndim == 3 and conv1d_weight.shape[0] == 1 and conv1d_weight.shape[1] == 1

        # Register conv3d weight/bias as parameters
        out_ch, in_ch, kd, kh, kw = conv3d_weight.shape
        self.conv3d_weight = nn.Parameter(conv3d_weight.clone())
        self.conv3d_bias = nn.Parameter(conv3d_bias.clone())

        # Reflection padding to preserve spatial dims after conv3d (same conv)
        pad_w = kw // 2
        pad_h = kh // 2
        pad_d = kd // 2
        # ReflectionPad3d expects padding in the order (padW_left, padW_right, padH_top, padH_bottom, padD_front, padD_back)
        self.reflect_pad = nn.ReflectionPad3d((pad_w, pad_w, pad_h, pad_h, pad_d, pad_d))

        # Local response normalization (applied after conv3d)
        self.lrn = nn.LocalResponseNorm(size=lrn_size, alpha=lrn_alpha, beta=lrn_beta, k=lrn_k)

        # CircularPad1d to operate along the width (W) axis with padding that matches 1D conv kernel
        # conv1d_weight shape: (out_channels=1, in_channels=1, k1)
        k1 = conv1d_weight.shape[2]
        assert k1 % 2 == 1, "conv1d kernel size must be odd to preserve width after circular padding as configured"
        pad1 = (k1 - 1) // 2
        self.circ_pad = nn.CircularPad1d(pad1)

        # Register conv1d weight as parameter (we perform functional conv1d)
        self.conv1d_weight = nn.Parameter(conv1d_weight.clone())

        # Store dims for later checks/logic (not strictly necessary but helpful)
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.k3d = (kd, kh, kw)
        self.k1 = k1
        self.pad1 = pad1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: Tensor[B, C_in, D, H, W]
        returns: Tensor[B, C_out, D, H, W] after the composed operations
        """
        # Basic checks
        assert x.ndim == 5 and x.shape[1] == self.in_ch, "Input must be 5D (B, C_in, D, H, W) with the configured in_ch"

        B, C_in, D, H, W = x.shape

        # 1) Reflection pad to preserve sizes for conv3d
        xp = self.reflect_pad(x)  # shape -> (B, C_in, D + 2*pd, H + 2*ph, W + 2*pw)

        # 2) 3D convolution using supplied weight and bias (same padding effect via reflection pad)
        conv3d_out = F.conv3d(xp, self.conv3d_weight, bias=self.conv3d_bias, stride=1, padding=0, dilation=1)
        # conv3d_out shape: (B, out_ch, D, H, W)

        # 3) Local Response Normalization (channel-wise competition normalization)
        lrn_out = self.lrn(conv3d_out)  # same shape

        # 4) Prepare for 1D circular smoothing along width:
        #    reshape to treat each (B, out_ch, D, H) element as an independent signal over W
        #    we want shape (N, C=1, L=W) where N = B * out_ch * D * H
        N = B * self.out_ch * D * H
        y = lrn_out.permute(0, 1, 2, 3, 4).contiguous().view(N, 1, W)  # (N,1,W)

        # 5) Circular pad and 1D conv (acts as a wrapped smoothing filter along width)
        y_padded = self.circ_pad(y)  # (N,1,W + 2*pad1)
        y_conv1d = F.conv1d(y_padded, self.conv1d_weight, bias=None, stride=1, padding=0, groups=1)  # (N,1,W)
        y_conv1d = y_conv1d.view(B, self.out_ch, D, H, W)  # back to (B, out_ch, D, H, W)

        # 6) Gating: compute a channel-wise gate from global spatial average, then broadcast multiply
        #    squeeze per-channel by averaging over D,H,W -> (B, out_ch)
        channel_squeeze = y_conv1d.mean(dim=(2, 3, 4))  # (B, out_ch)
        gate = torch.sigmoid(channel_squeeze)  # (B, out_ch)
        # reshape for broadcasting
        gate = gate.view(B, self.out_ch, 1, 1, 1)
        gated = y_conv1d * gate  # (B, out_ch, D, H, W)

        # 7) Final Swish non-linearity
        out = gated * torch.sigmoid(gated)

        return out


# Configuration variables (example sizes)
batch_size = 4
C_in = 8
C_out = 16
D = 5
H = 32
W = 64
kd, kh, kw = 3, 3, 3  # 3D conv kernel dims (odd to simplify padding)
k1 = 3  # 1D conv kernel (odd so circular padding yields same width)

def get_inputs():
    """
    Returns the runtime inputs for forward:
      - x: Tensor[B, C_in, D, H, W]
    """
    x = torch.randn(batch_size, C_in, D, H, W, dtype=torch.float32)
    return [x]

def get_init_inputs():
    """
    Returns initialization tensors required to construct Model:
      - conv3d_weight: (C_out, C_in, kd, kh, kw)
      - conv3d_bias: (C_out,)
      - conv1d_weight: (1, 1, k1)
    These are provided so the Model can be instantiated with deterministic shapes/values.
    """
    conv3d_weight = torch.randn(C_out, C_in, kd, kh, kw, dtype=torch.float32)
    conv3d_bias = torch.randn(C_out, dtype=torch.float32)
    conv1d_weight = torch.randn(1, 1, k1, dtype=torch.float32) / (k1)  # small smoothing kernel
    return [conv3d_weight, conv3d_bias, conv1d_weight]