import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Complex vision-ish module that:
    - Pads the input with a constant border
    - Applies a learned convolution (provided as an input tensor)
    - Applies a SiLU non-linearity
    - Computes a channel-wise gating vector via global average pooling, a projection matrix, and a Sigmoid
    - Re-weights the feature maps by the gate
    - Computes per-example channel correlation matrices via batched matrix multiplication
    - Applies a final SiLU and returns the correlation tensors

    This structure combines padding, convolution, SiLU, Sigmoid, pooling, matrix multiplications, and batched GEMM.
    """
    def __init__(self, pad_value: float = 0.1):
        super(Model, self).__init__()
        # instantiate reusable layers/ops
        # ConstantPad2d accepts an int or tuple; using int=1 pads 1 pixel on all sides
        self.pad = nn.ConstantPad2d(1, pad_value)
        self.silu = nn.SiLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, X: torch.Tensor, conv_weight: torch.Tensor, proj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            X (torch.Tensor): Input feature maps, shape (B, C_in, H, W).
            conv_weight (torch.Tensor): Convolution weights, shape (C_out, C_in, kH, kW).
            proj (torch.Tensor): Projection matrix for channel gating, shape (C_out, C_out).

        Returns:
            torch.Tensor: A per-example channel correlation tensor of shape (B, C_out, C_out) after non-linearity.
        """
        # 1) Pad spatial boundaries with a constant value
        X_padded = self.pad(X)  # (B, C_in, H+2, W+2)

        # 2) Apply convolution using the provided conv_weight (no bias)
        #    Use stride=1 and padding=0 because padding has been handled explicitly
        Y = F.conv2d(X_padded, conv_weight, bias=None, stride=1, padding=0)  # (B, C_out, H, W)

        # 3) Non-linearity
        Y = self.silu(Y)  # (B, C_out, H, W)

        # 4) Squeeze spatial dimensions with global average pooling to get channel descriptors
        z = Y.mean(dim=(2, 3))  # (B, C_out)

        # 5) Linear projection across channels using the provided projection matrix, then sigmoid gating
        #    We compute z @ proj to produce a transformed descriptor per channel
        gating_logits = torch.matmul(z, proj)  # (B, C_out)
        gate = self.sigmoid(gating_logits).unsqueeze(-1).unsqueeze(-1)  # (B, C_out, 1, 1)

        # 6) Apply channel-wise gating
        Y_gated = Y * gate  # (B, C_out, H, W)

        # 7) Flatten spatial dims and compute channel correlation via batched matmul
        B, C_out, H, W = Y_gated.shape
        Y_flat = Y_gated.view(B, C_out, H * W)  # (B, C_out, S) where S = H*W
        # Correlation matrix per example: (B, C_out, C_out)
        corr = torch.bmm(Y_flat, Y_flat.transpose(1, 2))

        # 8) Final non-linearity to stabilize/introduce non-linearity in correlations
        out = self.silu(corr)  # (B, C_out, C_out)

        return out

# Configuration variables
batch_size = 8
C_in = 3
C_out = 16
H = 32
W = 32
kernel_size = 3

def get_inputs():
    """
    Returns:
        List of input tensors compatible with Model.forward:
            - X: (batch_size, C_in, H, W)
            - conv_weight: (C_out, C_in, kernel_size, kernel_size)
            - proj: (C_out, C_out)
    """
    X = torch.randn(batch_size, C_in, H, W)
    conv_weight = torch.randn(C_out, C_in, kernel_size, kernel_size)
    proj = torch.randn(C_out, C_out)
    return [X, conv_weight, proj]

def get_init_inputs():
    # No special initialization parameters required for construction of Model()
    return []