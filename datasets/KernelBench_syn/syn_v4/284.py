import torch
import torch.nn as nn

# Configuration parameters for the model inputs and convolutions
BATCH = 8
IN_CH = 3
MID_CH = 32
OUT_CH = 16
H = 128
W = 128
KERNEL = 3
STRIDE = 1
PADDING = 1

class Model(nn.Module):
    """
    Convolutional gating block with residual connection and shrinkage.

    The computation performed in forward is:
      1) conv1 = Conv2d(IN_CH -> MID_CH, kernel=3, padding=1)
      2) conv2 = Conv2d(MID_CH -> OUT_CH, kernel=3, padding=1)
      3) skip = Conv2d(IN_CH -> OUT_CH, kernel=1)   # channel-matching residual
      4) gate = Sigmoid(conv1)                      # spatial gating from conv1
      5) gated = conv2 * gate                       # elementwise gating
      6) res = skip + gated                         # residual addition
      7) out = Hardshrink(res, lambd=0.6)           # sparsify small activations

    This creates a small, expressive block that mixes convolutions, multiplicative gating,
    residual paths and a hard shrinkage nonlinearity.
    """
    def __init__(self,
                 in_channels: int = IN_CH,
                 mid_channels: int = MID_CH,
                 out_channels: int = OUT_CH,
                 kernel_size: int = KERNEL,
                 stride: int = STRIDE,
                 padding: int = PADDING):
        super(Model, self).__init__()
        # Main convolutional pathway
        self.conv1 = nn.Conv2d(in_channels, mid_channels,
                               kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv2 = nn.Conv2d(mid_channels, out_channels,
                               kernel_size=kernel_size, stride=1, padding=padding)
        # 1x1 conv for residual / skip connection to match channels
        self.skip_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        # Non-linearities
        self.gate = nn.Sigmoid()
        self.shrink = nn.Hardshrink(lambd=0.6)

        # Initialize weights for stability (Kaiming for convs)
        for m in [self.conv1, self.conv2, self.skip_conv]:
            nn.init.kaiming_uniform_(m.weight, a=0.2)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the block.

        Args:
            x (torch.Tensor): Input tensor of shape (BATCH, IN_CH, H, W)

        Returns:
            torch.Tensor: Output tensor of shape (BATCH, OUT_CH, H, W)
        """
        # Primary path
        conv1_out = self.conv1(x)          # (B, MID_CH, H, W)
        conv2_out = self.conv2(conv1_out) # (B, OUT_CH, H, W)

        # Gating mechanism: use sigmoid of conv1 to gate conv2 spatially & channel-wise
        gate_mask = self.gate(conv1_out)  # broadcastable to conv2_out via channel reduction; conv1_out channels may differ
        # If channels differ, reduce gate to match conv2_out channels by a 1x1 conv-like projection:
        if gate_mask.shape[1] != conv2_out.shape[1]:
            # Simple linear projection along channel dim using mean pooling across channel groups
            # This keeps the gating operation parameter-free and deterministic.
            groups = gate_mask.shape[1] // conv2_out.shape[1] if gate_mask.shape[1] >= conv2_out.shape[1] else 1
            if groups > 1:
                # reshape and mean across groups to reduce channels
                B, Cg, Ht, Wt = gate_mask.shape
                gate_mask = gate_mask.view(B, conv2_out.shape[1], groups, Ht, Wt).mean(dim=2)
            else:
                # if fewer gate channels than conv2, repeat channels
                repeat = (conv2_out.shape[1] + gate_mask.shape[1] - 1) // gate_mask.shape[1]
                gate_mask = gate_mask.repeat(1, repeat, 1, 1)[:, :conv2_out.shape[1], :, :]

        gated = conv2_out * gate_mask      # elementwise gating

        # Residual / skip connection to preserve input information
        skip = self.skip_conv(x)

        res = skip + gated

        # Apply hard shrinkage to encourage sparsity of small activations
        out = self.shrink(res)

        return out

def get_inputs():
    """
    Create and return input tensors for the model.

    Returns:
        list: [x] where x is a tensor shaped (BATCH, IN_CH, H, W)
    """
    x = torch.randn(BATCH, IN_CH, H, W)
    return [x]

def get_init_inputs():
    """
    Return any special initialization parameters required to construct the model.
    For this module we don't require additional runtime initialization parameters,
    so return an empty list.
    """
    return []