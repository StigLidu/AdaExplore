import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# Configuration
batch_size = 16
in_channels = 3
depth = 8
height = 4
width = 16

# Padding specification for ConstantPad3d:
# (pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back)
pad_spec = (1, 1, 0, 0, 1, 1)  # extends width by 2, depth by 2, leaves height unchanged

# LPPool1d parameters
lp_norm = 2
lp_kernel = 2
lp_stride = 2  # we'll use stride equal to kernel to make output length floor(L / 2)

# Adaptive softmax settings
num_classes = 10000
cutoffs = [1000, 5000, 8000]  # must be sorted and < num_classes
hidden_dim = 512  # size of the feature vector fed into AdaptiveLogSoftmaxWithLoss

class Model(nn.Module):
    """
    Complex example module combining 3D constant padding, 1D Lp pooling over a collapsed
    spatial-temporal dimension, a lightweight MLP projection, and an AdaptiveLogSoftmaxWithLoss
    head to produce a loss for classification over a large number of classes.

    Computation steps:
    1. ConstantPad3d on the input tensor (N, C, D, H, W).
    2. Rearrange and collapse to (N, C * H, D * W_padded) to create a per-plane sequence.
    3. Apply LPPool1d to perform a power-average pooling across the sequence dimension.
    4. Apply ReLU, flatten, and project with a Linear layer to a fixed-size hidden vector.
    5. Compute adaptive log-softmax loss against provided targets.
    """
    def __init__(
        self,
        in_channels: int,
        depth: int,
        height: int,
        width: int,
        pad: tuple,
        lp_norm: int,
        lp_kernel: int,
        lp_stride: int,
        hidden_dim: int,
        num_classes: int,
        cutoffs: list,
    ):
        super(Model, self).__init__()

        # Store dims and pad
        self.in_channels = in_channels
        self.depth = depth
        self.height = height
        self.width = width
        self.pad = pad  # (left, right, top, bottom, front, back)

        # Constant padding layer for 5D input (N, C, D, H, W)
        self.const_pad = nn.ConstantPad3d(self.pad, 0.1)

        # LPPool1d applied after collapsing to (N, C * H, L)
        self.lppool = nn.LPPool1d(norm_type=lp_norm, kernel_size=lp_kernel, stride=lp_stride)

        # Activation
        self.act = nn.ReLU()

        # Compute the length L after padding and the pooled length L2.
        pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back = self.pad
        padded_width = self.width + pad_left + pad_right
        padded_depth = self.depth + pad_front + pad_back
        L = padded_depth * padded_width

        # With stride equal to kernel, an approximate L_out is floor(L / kernel)
        L2 = L // lp_stride
        if L2 < 1:
            L2 = 1  # guard against degenerate cases

        # Number of channels after collapsing height into channel dimension
        collapsed_channels = self.in_channels * self.height
        in_features = collapsed_channels * L2

        # Linear projection to hidden_dim which is the input size for adaptive softmax
        self.fc = nn.Linear(in_features, hidden_dim)

        # Adaptive Log Softmax with Loss head
        self.adaptive_log_softmax = nn.AdaptiveLogSoftmaxWithLoss(
            in_features=hidden_dim,
            n_classes=num_classes,
            cutoffs=cutoffs,
            div_value=4.0,
        )

        # Save computed dims for reference/debug
        self._computed_in_features = in_features
        self._pooled_length = L2
        self._collapsed_channels = collapsed_channels

    def forward(self, x: torch.Tensor, target: torch.Tensor):
        """
        Forward computation.

        Args:
            x (torch.Tensor): Input tensor of shape (N, C, D, H, W).
            target (torch.Tensor): Target tensor of shape (N,) with class indices [0, num_classes-1].

        Returns:
            torch.Tensor: Scalar loss tensor computed by AdaptiveLogSoftmaxWithLoss.
        """
        # 1. Pad the 3D spatial-temporal input
        # Input expected shape: (N, C, D, H, W)
        x_padded = self.const_pad(x)

        # 2. Rearrange to collapse height into channel dimension and depth*width into sequence length
        # Current shape: (N, C, D_p, H, W_p)
        N, C, Dp, Hp, Wp = x_padded.shape
        # Permute to (N, C, H, D, W) then collapse to (N, C * H, D * W)
        x_seq = x_padded.permute(0, 1, 3, 2, 4).contiguous()  # (N, C, H, D, W)
        seq_len = Dp * Wp
        x_seq = x_seq.view(N, C * Hp, seq_len)  # (N, C*H, L)

        # 3. Apply LPPool1d over the sequence dimension
        pooled = self.lppool(x_seq)  # (N, C*H, L2)

        # 4. Activation and flatten
        activated = self.act(pooled)
        flat = activated.view(N, -1)  # (N, collapsed_channels * L2)

        # 5. Project to hidden vectors and compute adaptive softmax loss
        hidden = self.fc(flat)  # (N, hidden_dim)

        # AdaptiveLogSoftmaxWithLoss returns a tuple (output, loss) where loss is a 0-dim Tensor
        out = self.adaptive_log_softmax(hidden, target)
        # out[0] is the log-probabilities for the target classes (if requested), out[1] is the loss
        loss = out[1]
        return loss


def get_inputs():
    """
    Creates random example inputs:
    - x: tensor of shape (batch_size, in_channels, depth, height, width)
    - target: integer class indices of shape (batch_size,)
    """
    x = torch.randn(batch_size, in_channels, depth, height, width)
    target = torch.randint(low=0, high=num_classes, size=(batch_size,), dtype=torch.long)
    return [x, target]

def get_init_inputs():
    """
    Returns the initialization parameters required to construct the Model.
    """
    return [
        in_channels,
        depth,
        height,
        width,
        pad_spec,
        lp_norm,
        lp_kernel,
        lp_stride,
        hidden_dim,
        num_classes,
        cutoffs,
    ]