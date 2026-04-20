import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Complex 3D-to-2D convolutional processing block that demonstrates:
      - 3D constant padding (nn.ConstantPad3d)
      - Channel-wise Layer Normalization applied after permuting channels to the last dimension (nn.LayerNorm)
      - Lazy 2D convolution that infers input channels at first forward (nn.LazyConv2d)
      - Channel gating using a small fully-connected bottleneck with sigmoid gating

    The forward pass expects a 5D tensor x with shape (N, C, D, H, W) and performs:
      1. ConstantPad3d to expand D,H,W dims
      2. Permute to put channels last: (N, D, H, W, C)
      3. LayerNorm over the channel dimension
      4. Permute back and collapse D and H into a single spatial dimension to form a 4D tensor (N, C, D*H, W)
      5. Apply LazyConv2d -> ReLU
      6. Global average pooling + Linear + sigmoid to form channel gates
      7. Multiply gates across channels (channel-wise modulation)
    """
    def __init__(self, in_channels: int, out_channels: int, conv_kernel_size=(3,3), pad_tuple=(1,1,1,1,1,1), pad_value: float = 0.0):
        """
        Args:
            in_channels (int): Number of channels in the input (C).
            out_channels (int): Number of output channels for the LazyConv2d.
            conv_kernel_size (int or tuple): Kernel size for the 2D convolution applied after collapsing spatial dims.
            pad_tuple (tuple): 6-tuple for ConstantPad3d (pad_w_left, pad_w_right, pad_h_top, pad_h_bottom, pad_d_front, pad_d_back).
            pad_value (float): Constant value used for padding.
        """
        super(Model, self).__init__()
        # 3D constant padding layer; pad_tuple must be length 6
        self.pad3d = nn.ConstantPad3d(pad_tuple, pad_value)
        # LayerNorm applied over the channel dimension after permuting channels to last position
        # normalized_shape is the size of the last dimension (channels)
        self.ln = nn.LayerNorm(in_channels)
        # LazyConv2d will infer in_channels on first forward; we only specify out_channels and kernel_size
        self.conv2d = nn.LazyConv2d(out_channels=out_channels, kernel_size=conv_kernel_size, stride=1, padding=0)
        # Simple gating MLP (channel-wise): projects pooled channels to gates
        self.gate_fc = nn.Linear(out_channels, out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (N, C, D, H, W)

        Returns:
            torch.Tensor: Output tensor of shape (N, out_channels, H_out, W_out)
        """
        # Step 1: pad the 3D spatial volume
        x_p = self.pad3d(x)  # shape: (N, C, D2, H2, W2)
        N, C, D2, H2, W2 = x_p.shape

        # Step 2: permute so channels are last for LayerNorm
        x_perm = x_p.permute(0, 2, 3, 4, 1).contiguous()  # (N, D2, H2, W2, C)

        # Step 3: LayerNorm over the channel dimension (last dim)
        x_ln = self.ln(x_perm)  # (N, D2, H2, W2, C)

        # Step 4: permute back and collapse D2 and H2 into a single spatial dimension
        x_back = x_ln.permute(0, 4, 1, 2, 3).contiguous()  # (N, C, D2, H2, W2)
        # Merge depth and height into a single height-like dimension
        x_4d = x_back.view(N, C, D2 * H2, W2)  # (N, C, D2*H2, W2)

        # Step 5: 2D convolution (lazy initializes in_channels) + activation
        conv_out = self.conv2d(x_4d)  # (N, out_channels, H_out, W_out)
        conv_out = self.act(conv_out)

        # Step 6: channel-wise gating computed from global average pooled features
        gap = F.adaptive_avg_pool2d(conv_out, output_size=(1, 1)).view(N, -1)  # (N, out_channels)
        gates = torch.sigmoid(self.gate_fc(gap)).view(N, -1, 1, 1)  # (N, out_channels, 1, 1)

        # Step 7: apply gating (channel-wise modulation)
        out = conv_out * gates  # (N, out_channels, H_out, W_out)
        return out


# Configuration / shapes
batch_size = 8
channels = 16
depth = 4
height = 32
width = 24
out_channels = 32
conv_kernel = (3, 3)
pad_tuple = (1, 1, 1, 1, 1, 1)  # pad W(left,right), H(top,bottom), D(front,back)
pad_value = 0.0

def get_inputs():
    """
    Returns example input tensors for the model.
    """
    x = torch.randn(batch_size, channels, depth, height, width)
    return [x]

def get_init_inputs():
    """
    Returns initialization arguments for constructing Model:
      [in_channels, out_channels, conv_kernel_size, pad_tuple, pad_value]
    """
    return [channels, out_channels, conv_kernel, pad_tuple, pad_value]