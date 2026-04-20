import torch
import torch.nn as nn

"""
Complex PyTorch kernel module combining ReflectionPad3d, BatchNorm2d, and RNN.

Computation pattern:
1. Pad a 5D video-like tensor (B, C, D, H, W) with ReflectionPad3d.
2. Reshape to apply BatchNorm2d over each frame: (B*D_padded, C, H_padded, W_padded).
3. Spatially pool the normalized frames to produce a sequence per batch for RNN.
4. Process the sequence with an nn.RNN; map the final hidden representation back to channel space.
5. Expand and fuse the RNN-derived channel modulation with the normalized 5D feature map to produce the output volume.
"""

# Configuration / default sizes
BATCH = 8
CHANNELS = 16
DEPTH = 10
HEIGHT = 32
WIDTH = 32
RNN_HIDDEN = 32
RNN_LAYERS = 2
PAD = 1  # padding size applied symmetrically on each spatial dimension


class Model(nn.Module):
    """
    Model that processes a 5D tensor (batch, channels, depth, height, width) with:
      - ReflectionPad3d
      - BatchNorm2d applied per-frame
      - An RNN over the temporal (depth) dimension using per-frame pooled features
      - Projection from RNN hidden state back to channel modulation
      - Fusion of modulation with normalized features to produce the output volume
    """
    def __init__(self, in_channels: int, rnn_hidden: int, rnn_layers: int, pad: int):
        """
        Args:
            in_channels (int): Number of input channels (C).
            rnn_hidden (int): Hidden size of the RNN.
            rnn_layers (int): Number of RNN layers.
            pad (int): Symmetric padding size for ReflectionPad3d on each spatial dim.
        """
        super(Model, self).__init__()
        self.in_channels = in_channels
        self.rnn_hidden = rnn_hidden
        self.rnn_layers = rnn_layers
        self.pad = pad

        # ReflectionPad3d expects (padW_left, padW_right, padH_top, padH_bottom, padD_front, padD_back)
        pad_tuple = (pad, pad, pad, pad, pad, pad)
        self.pad3d = nn.ReflectionPad3d(pad_tuple)

        # BatchNorm2d will be applied to per-frame 4D tensors of shape (N_frames, C, H, W)
        self.bn2d = nn.BatchNorm2d(in_channels, affine=True)

        # RNN: processes sequences of length D_padded where each input feature is 'in_channels'
        # We'll use tanh non-linearity to match common Elman RNN behavior.
        self.rnn = nn.RNN(
            input_size=in_channels,
            hidden_size=rnn_hidden,
            num_layers=rnn_layers,
            nonlinearity='tanh',
            batch_first=False  # we will provide (seq_len, batch, features)
        )

        # Project RNN hidden state back to channel space for per-channel modulation
        self.proj = nn.Linear(rnn_hidden, in_channels, bias=True)

        # Small activation for final fusion
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, D, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (B, C, D_padded, H_padded, W_padded).
        """
        # 1) Reflection pad the input volume
        # Result: (B, C, D_p, H_p, W_p)
        x_padded = self.pad3d(x)

        B, C, Dp, Hp, Wp = x_padded.shape

        # 2) Prepare per-frame 4D tensor for BatchNorm2d:
        #    reshape (B, C, Dp, Hp, Wp) -> (B * Dp, C, Hp, Wp)
        x_frames = x_padded.permute(0, 2, 1, 3, 4).contiguous().view(B * Dp, C, Hp, Wp)

        # 3) Apply BatchNorm2d over frames
        x_frames_norm = self.bn2d(x_frames)  # shape: (B*Dp, C, Hp, Wp)

        # 4) Spatial pooling (global mean over H and W) to create sequence features per frame
        #    -> (B*Dp, C)
        frame_features = x_frames_norm.mean(dim=(2, 3))

        # 5) Reshape to sequence for RNN: (Bp, C) -> (B, Dp, C) then transpose to (Dp, B, C)
        seq = frame_features.view(B, Dp, C).transpose(0, 1)  # (seq_len=Dp, batch=B, input_size=C)

        # 6) RNN over temporal frames
        rnn_out, h_n = self.rnn(seq)  # rnn_out: (Dp, B, rnn_hidden)

        # Use final time-step's output (last frame) as summary per-batch
        rnn_last = rnn_out[-1]  # shape: (B, rnn_hidden)

        # 7) Project back to channel dimension to obtain per-channel modulation vector
        channel_mod = self.proj(rnn_last)  # (B, C)

        # 8) Expand the modulation to full spatial-temporal volume: (B, C, Dp, Hp, Wp)
        modulation = channel_mod.view(B, C, 1, 1, 1).expand(B, C, Dp, Hp, Wp)

        # 9) Restore normalized frames back to 5D (B, C, Dp, Hp, Wp) from x_frames_norm
        x_norm_5d = x_frames_norm.view(B, Dp, C, Hp, Wp).permute(0, 2, 1, 3, 4).contiguous()

        # 10) Fuse modulation with normalized features: gated addition followed by activation
        #     output = ReLU(x_norm * sigmoid(modulation) + x_padded * (1 - sigmoid(modulation)))
        gate = torch.sigmoid(modulation)
        fused = x_norm_5d * gate + x_padded * (1.0 - gate)
        output = self.act(fused)

        return output


def get_inputs():
    """
    Generate a random 5D tensor simulating a small video batch for testing.

    Returns:
        list: [x] where x is a tensor of shape (BATCH, CHANNELS, DEPTH, HEIGHT, WIDTH)
    """
    x = torch.randn(BATCH, CHANNELS, DEPTH, HEIGHT, WIDTH)
    return [x]


def get_init_inputs():
    """
    Return the initialization parameters for the Model: in_channels, rnn_hidden, rnn_layers, pad
    """
    return [CHANNELS, RNN_HIDDEN, RNN_LAYERS, PAD]