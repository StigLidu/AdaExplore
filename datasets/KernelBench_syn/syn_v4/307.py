import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Model(nn.Module):
    """
    Custom convolution-like module implemented with unfold + matmul,
    followed by Hardswish activation, a channel gating using Hardtanh,
    and a lightweight channel projection residual. Uses ZeroPad2d for framing.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True,
    ):
        """
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int): Square kernel height/width.
            stride (int): Stride for the unfolding (convolution).
            padding (int): Zero padding to apply around the input.
            bias (bool): Whether to include a bias term in the convolution.
        """
        super(Model, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # Zero padding layer applied before unfolding
        self.pad = nn.ZeroPad2d(padding)

        # Convolution weights implemented as a flat matrix for matmul:
        # shape: (out_channels, in_channels * kernel_size * kernel_size)
        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels * kernel_size * kernel_size)
        )

        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.bias = None

        # Small channel projection for residual path: projects pooled input channels -> out_channels
        # shape: (out_channels, in_channels)
        self.res_weight = nn.Parameter(torch.randn(out_channels, in_channels))

        # Non-linearities
        self.hardswish = nn.Hardswish()
        # Clip gating values to a bounded range
        self.hardtanh = nn.Hardtanh(min_val=-0.5, max_val=0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass:
         1. Zero-pad the input.
         2. Extract patches with unfold (acting like a convolution sliding window).
         3. Perform matmul with weight to produce out_channels responses.
         4. Reshape to (B, out_channels, H_out, W_out).
         5. Apply Hardswish activation.
         6. Compute a channel-wise gating value (spatial mean), clip it with Hardtanh,
            and use it to scale the activated features.
         7. Add a lightweight residual projected from the input's global spatial mean.
         8. Final Hardtanh clamp for numeric stability.
        """
        B, C, H, W = x.shape
        K = self.kernel_size
        S = self.stride
        P = self.padding

        # 1) Pad
        x_padded = self.pad(x)  # shape: (B, C, H + 2P, W + 2P)

        # 2) Unfold to get sliding local blocks
        # patches shape: (B, C * K * K, L) where L = H_out * W_out
        patches = F.unfold(x_padded, kernel_size=K, stride=S)

        # 3) Convolution via matrix multiplication:
        # weight: (out_channels, in_channels * K * K)
        # patches: (B, in_channels * K * K, L)
        # -> conv_out_flat: (B, out_channels, L)
        conv_out_flat = torch.matmul(self.weight.unsqueeze(0).expand(B, -1, -1), patches)
        if self.bias is not None:
            conv_out_flat = conv_out_flat + self.bias.view(1, -1, 1)

        # 4) Reshape to image grid
        L = conv_out_flat.size(-1)
        # Compute output spatial dims using input dims and conv parameters
        H_out = (H + 2 * P - K) // S + 1
        W_out = (W + 2 * P - K) // S + 1
        # Safety check: if L != H_out * W_out, infer grid by sqrt if square
        if H_out * W_out != L:
            # fallback: try square assumption
            side = int(math.sqrt(L))
            H_out = side
            W_out = side
        conv_out = conv_out_flat.view(B, self.out_channels, H_out, W_out)

        # 5) Non-linearity
        activated = self.hardswish(conv_out)

        # 6) Channel-wise gating computed from spatial means
        gating = activated.mean(dim=(2, 3), keepdim=True)  # (B, out_channels, 1, 1)
        gating = self.hardtanh(gating)  # clamp in [-0.5, 0.5]

        # Scale features by (1 + gating) so small positive/negative adjustments occur
        scaled = activated * (1.0 + gating)

        # 7) Lightweight residual: project input's global mean (per-channel) into out_channels
        # global pooled input: (B, in_channels)
        pooled_in = x.mean(dim=(2, 3))  # (B, in_channels)
        # project: (B, in_channels) x (in_channels, out_channels) -> (B, out_channels)
        res_proj = torch.matmul(pooled_in, self.res_weight.t())  # (B, out_channels)
        res_proj = res_proj.view(B, self.out_channels, 1, 1)  # expand to spatial dims
        out = scaled + res_proj  # broadcast addition

        # 8) Final stabilization clamp
        out = self.hardtanh(out)

        return out

# Module-level configuration and example sizes
batch_size = 8
in_channels = 12
out_channels = 16
height = 64
width = 64
kernel_size = 5
stride = 2
padding = 2
use_bias = True

def get_inputs():
    """
    Returns a list containing a single input tensor shaped (batch_size, in_channels, height, width).
    """
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]

def get_init_inputs():
    """
    Returns initialization parameters for the Model constructor in the order:
    (in_channels, out_channels, kernel_size, stride, padding, bias)
    """
    return [in_channels, out_channels, kernel_size, stride, padding, use_bias]