import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Any

# Configuration variables
batch_size = 8
in_channels = 3
height = 32
width = 64

# Padding tuple for CircularPad2d: (pad_left, pad_right, pad_top, pad_bottom)
pad_left = 1
pad_right = 2
pad_top = 1
pad_bottom = 2

# LPPool1d settings
lp_norm = 2.0           # p-norm for LPPool
lp_kernel = 3
lp_stride = 2

# RNN settings
rnn_hidden_size = 256
rnn_num_layers = 2
rnn_nonlinearity = 'tanh'  # 'tanh' or 'relu'

class Model(nn.Module):
    """
    Complex model that:
    - Applies circular 2D padding
    - Rearranges spatial dims to a 1D signal per (channel x height)
    - Pools the 1D signal via LPPool1d
    - Feeds the pooled sequence through a multi-layer RNN
    - Maps the final RNN hidden state back to the original spatial shape via a Linear layer
    """
    def __init__(
        self,
        in_channels: int,
        height: int,
        width: int,
        pad: tuple = (pad_left, pad_right, pad_top, pad_bottom),
        lp_p: float = lp_norm,
        lp_kernel_size: int = lp_kernel,
        lp_stride_size: int = lp_stride,
        rnn_hidden: int = rnn_hidden_size,
        rnn_layers: int = rnn_num_layers,
        rnn_nonlinearity: str = rnn_nonlinearity
    ):
        """
        Initializes the model components.

        Args:
            in_channels (int): Number of input channels.
            height (int): Height of the input feature map.
            width (int): Width of the input feature map.
            pad (tuple): Circular pad amounts (left, right, top, bottom).
            lp_p (float): p value for LPPool1d.
            lp_kernel_size (int): Kernel size for LPPool1d.
            lp_stride_size (int): Stride for LPPool1d.
            rnn_hidden (int): Hidden size for the RNN.
            rnn_layers (int): Number of RNN layers.
            rnn_nonlinearity (str): Nonlinearity for the RNN ('tanh' or 'relu').
        """
        super(Model, self).__init__()

        self.in_channels = in_channels
        self.height = height
        self.width = width
        self.pad = nn.CircularPad2d(pad)

        # Compute padded spatial dimensions (these are fixed given height/width and pad)
        _, _, pad_top, pad_bottom = (0, 0, pad[2], pad[3])
        _, _, _, _ = (pad[0], pad[1], pad[2], pad[3])
        padded_height = height + pad_top + pad_bottom

        # LPPool1d will operate over the width dimension after reshaping
        self.lppool = nn.LPPool1d(norm_type=float(lp_p), kernel_size=lp_kernel_size, stride=lp_stride_size)

        # RNN expects input of size (seq_len, batch, input_size) when batch_first=False.
        # input_size will be in_channels * padded_height because we collapse channels and height into the channel dimension for LPPool1d.
        rnn_input_size = in_channels * padded_height
        self.rnn = nn.RNN(
            input_size=rnn_input_size,
            hidden_size=rnn_hidden,
            num_layers=rnn_layers,
            nonlinearity=rnn_nonlinearity,
            batch_first=False
        )

        # Linear maps final hidden state back to flattened original image shape
        self.fc = nn.Linear(rnn_hidden, in_channels * height * width)

        # Small initializer for stability
        nn.init.kaiming_uniform_(self.fc.weight, a=math.sqrt(5)) if hasattr(nn.init, 'kaiming_uniform_') else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass:
        1. Circular pad the 4D input (N, C, H, W)
        2. Reshape to (N, C * H_padded, W_padded) to prepare for LPPool1d
        3. Apply LPPool1d over the width dimension -> (N, C * H_padded, L_pooled)
        4. Transpose to (L_pooled, N, C * H_padded) and feed to RNN
        5. Use the last hidden state, project via Linear and reshape to (N, C, H, W)
        """
        # 1) Circular padding
        x_padded = self.pad(x)  # (N, C, H_padded, W_padded)

        N, C, H_p, W_p = x_padded.shape

        # 2) Collapse channels and height into a single channel dimension for 1D pooling
        # shape -> (N, C * H_p, W_p)
        x_reshaped = x_padded.view(N, C * H_p, W_p)

        # 3) LPPool1d along last dimension (width)
        x_pooled = self.lppool(x_reshaped)  # (N, C * H_p, L_p)

        # 4) Prepare sequence for RNN: (seq_len, batch, input_size)
        seq_len = x_pooled.shape[2]
        # transpose to (seq_len, N, input_size)
        x_seq = x_pooled.permute(2, 0, 1).contiguous()

        # RNN
        _, h_n = self.rnn(x_seq)  # h_n: (num_layers * num_directions, N, hidden_size)
        # Take the last layer's hidden state
        last_hidden = h_n[-1]  # (N, hidden_size)

        # 5) Map back to image shape
        out_flat = self.fc(last_hidden)  # (N, C * H * W)
        out = out_flat.view(N, self.in_channels, self.height, self.width)

        # Apply a final bounded activation to keep outputs stable
        return torch.tanh(out)


import math

def get_inputs() -> List[torch.Tensor]:
    """
    Returns a list of input tensors for the model's forward method.
    """
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]

def get_init_inputs() -> List[Any]:
    """
    Returns a list of inputs required to initialize the Model constructor.
    The order matches Model.__init__ parameters after self.
    """
    return [in_channels, height, width, (pad_left, pad_right, pad_top, pad_bottom), lp_norm, lp_kernel, lp_stride, rnn_hidden_size, rnn_num_layers, rnn_nonlinearity]