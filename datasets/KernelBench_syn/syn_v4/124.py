import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Complex example combining ReflectionPad3d, LayerNorm and ConstantPad1d.
    Workflow:
      - ReflectionPad3d applied to a 5D volumetric tensor (B, C, D, H, W)
      - Flatten spatial dims and apply LayerNorm across (C * D' * H' * W')
      - Project channels -> out_channels for each spatial location (einsum)
      - Global average pool over spatial positions -> (B, outC)
      - ConstantPad1d applied to a 1D sequence per batch -> (B, Lp)
      - Project padded sequence to (B, outC) via matmul with learned projection
      - Elementwise combine pooled visual features with sequence features
      - Produce an outer-product interaction tensor (B, outC, outC) as output
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        pad3d: tuple,
        pad1d: tuple,
        D: int,
        H: int,
        W: int,
        seq_len: int,
    ):
        """
        Args:
            in_channels (int): Number of channels of the volumetric input.
            out_channels (int): Number of output channels after projection.
            pad3d (tuple): 6-tuple for ReflectionPad3d (w_left, w_right, h_left, h_right, d_left, d_right).
            pad1d (tuple): 2-tuple for ConstantPad1d (left, right).
            D, H, W (int): Original spatial dimensions of the volumetric input.
            seq_len (int): Length of the 1D sequence input.
        """
        super(Model, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pad3d = nn.ReflectionPad3d(pad3d)
        self.pad1d = nn.ConstantPad1d(pad1d, 0.0)

        # Compute padded spatial sizes (map pad3d tuple to d/h/w as (w_left, w_right, h_left, h_right, d_left, d_right))
        pad_w_left, pad_w_right, pad_h_left, pad_h_right, pad_d_left, pad_d_right = pad3d
        Dp = D + pad_d_left + pad_d_right
        Hp = H + pad_h_left + pad_h_right
        Wp = W + pad_w_left + pad_w_right
        self.spatial_len = Dp * Hp * Wp

        # LayerNorm will normalize over the flattened (C * spatial_len) dimension per-sample.
        self.ln = nn.LayerNorm(normalized_shape=in_channels * self.spatial_len)

        # Channel projection: projects each channel to out_channels (applied per spatial location)
        # Weight shape (in_channels, out_channels)
        self.ch_proj = nn.Parameter(torch.randn(in_channels, out_channels) * (1.0 / (in_channels ** 0.5)))

        # Sequence projection: after ConstantPad1d, project padded sequence (Lp) -> out_channels
        left_pad, right_pad = pad1d
        self.seq_lp = seq_len + left_pad + right_pad
        # Weight shape (seq_lp, out_channels)
        self.seq_proj = nn.Parameter(torch.randn(self.seq_lp, out_channels) * (1.0 / (self.seq_lp ** 0.5)))

        self.act = nn.ReLU()

    def forward(self, vol: torch.Tensor, seq: torch.Tensor) -> torch.Tensor:
        """
        Args:
            vol: Volumetric tensor of shape (B, C, D, H, W).
            seq: 1D sequence tensor of shape (B, L).

        Returns:
            Tensor of shape (B, out_channels, out_channels) representing pairwise interactions
            of combined volumetric and sequence features.
        """
        B, C, D, H, W = vol.shape

        # 1) 3D reflection padding
        x = self.pad3d(vol)  # (B, C, D', H', W')

        # 2) Flatten spatial dimensions and apply LayerNorm over (C * D' * H' * W')
        x_flat = x.view(B, C * self.spatial_len)  # (B, C * S)
        x_norm = self.ln(x_flat)  # (B, C * S)
        x_resh = x_norm.view(B, C, self.spatial_len)  # (B, C, S)

        # 3) Project channels to out_channels for each spatial location
        #    einsum 'bcs,co->bos' -> (B, outC, S)
        x_proj = torch.einsum('bcs,co->bos', x_resh, self.ch_proj)

        # 4) Global average pooling over spatial positions -> (B, outC)
        pooled = x_proj.mean(dim=2)

        # 5) Pad sequence with ConstantPad1d and project to out_channels -> (B, outC)
        # ConstantPad1d accepts (B, L) shaped tensors
        seq_padded = self.pad1d(seq)  # (B, Lp)
        seq_feat = torch.matmul(seq_padded, self.seq_proj)  # (B, outC)

        # 6) Elementwise combine pooled visual features with sequence features
        combined = pooled * seq_feat  # (B, outC)

        # 7) Produce interaction matrix via outer product -> (B, outC, outC)
        out = torch.matmul(combined.unsqueeze(2), combined.unsqueeze(1))  # (B, outC, outC)

        return self.act(out)


# Module-level configuration (sizes and padding)
BATCH = 4
IN_CHANNELS = 8
OUT_CHANNELS = 12
DEPTH = 6
HEIGHT = 8
WIDTH = 10

# ReflectionPad3d expects a 6-tuple: (w_left, w_right, h_left, h_right, d_left, d_right)
PAD3D = (1, 2, 1, 1, 0, 2)

# ConstantPad1d expects (left, right)
PAD1D = (2, 3)
SEQ_LEN = 17

def get_inputs():
    """
    Returns:
        [vol, seq] where:
          - vol is a random tensor of shape (BATCH, IN_CHANNELS, DEPTH, HEIGHT, WIDTH)
          - seq is a random tensor of shape (BATCH, SEQ_LEN)
    """
    vol = torch.randn(BATCH, IN_CHANNELS, DEPTH, HEIGHT, WIDTH)
    seq = torch.randn(BATCH, SEQ_LEN)
    return [vol, seq]

def get_init_inputs():
    """
    Returns initialization parameters for the Model constructor.
    Order matches Model.__init__ signature:
      in_channels, out_channels, pad3d, pad1d, D, H, W, seq_len
    """
    return [IN_CHANNELS, OUT_CHANNELS, PAD3D, PAD1D, DEPTH, HEIGHT, WIDTH, SEQ_LEN]