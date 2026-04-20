import torch
import torch.nn as nn
import torch.nn.functional as F

# Configuration (module-level)
batch_size = 8
D = 4          # depth
H = 8          # height
W = 8          # width
seq_len = D * H * W  # sequence length must match D*H*W for reshaping
input_dim = 128
lstm_hidden = 64
lstm_layers = 2
bidirectional = True
trans_d_model = 256
nhead = 8
trans_ffn_dim = 512
num_trans_layers = 3
conv_in_channels = 64
conv_out_channels = 32
conv_kernel_size = 3
conv_stride = 2
conv_padding = 1
conv_output_padding = 1

class Model(nn.Module):
    """
    Complex model combining LSTM, stacked TransformerEncoderLayer blocks, and ConvTranspose3d.
    Pipeline:
      - Sequence input (batch, seq_len, input_dim)
      - Bidirectional LSTM -> (batch, seq_len, 2*lstm_hidden)
      - Project LSTM outputs to transformer d_model
      - Pass through a stack of TransformerEncoderLayer modules (with residuals)
      - Project transformer outputs to conv channel dimension and reshape to (batch, C, D, H, W)
      - Apply ConvTranspose3d to upsample spatial dimensions producing final 3D volume output
    """
    def __init__(
        self,
        input_dim: int,
        lstm_hidden: int,
        lstm_layers: int,
        trans_d_model: int,
        nhead: int,
        trans_ffn_dim: int,
        num_trans_layers: int,
        conv_in_channels: int,
        conv_out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        output_padding: int,
        d: int,
        h: int,
        w: int,
        bidirectional: bool = True,
    ):
        super(Model, self).__init__()

        self.input_dim = input_dim
        self.lstm_hidden = lstm_hidden
        self.lstm_layers = lstm_layers
        self.bidirectional = bidirectional
        self.direction_multiplier = 2 if bidirectional else 1
        self.trans_d_model = trans_d_model
        self.D = d
        self.H = h
        self.W = w

        # LSTM to encode the sequence
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )

        # Project LSTM features to transformer d_model
        self.lstm_to_transform = nn.Linear(lstm_hidden * self.direction_multiplier, trans_d_model)

        # A small stack of TransformerEncoderLayer modules
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=trans_d_model, nhead=nhead, dim_feedforward=trans_ffn_dim, batch_first=True)
            for _ in range(num_trans_layers)
        ])

        # LayerNorm used as final normalization before projection to conv
        self.norm = nn.LayerNorm(trans_d_model)

        # Project transformer outputs to conv channels
        self.transform_to_conv = nn.Linear(trans_d_model, conv_in_channels)

        # A ConvTranspose3d to upsample the 3D volume produced from the sequence
        self.deconv3d = nn.ConvTranspose3d(
            in_channels=conv_in_channels,
            out_channels=conv_out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
        )

        # A small conv to refine the upsampled volume
        self.refine_conv = nn.Conv3d(conv_out_channels, conv_out_channels, kernel_size=3, padding=1)

        # Activation
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, input_dim)

        Returns:
            out: (batch, conv_out_channels, D*stride, H*stride, W*stride)
        """
        # 1) Encode with LSTM
        # lstm_out: (batch, seq_len, lstm_hidden * num_directions)
        lstm_out, _ = self.lstm(x)

        # 2) Project LSTM outputs into transformer feature space
        # (batch, seq_len, trans_d_model)
        trans_in = self.lstm_to_transform(lstm_out)

        # 3) Pass through TransformerEncoderLayer stack (batch_first=True)
        # We apply residual connections implicitly as the layers provide them.
        t = trans_in
        for layer in self.transformer_layers:
            t = layer(t)  # (batch, seq_len, trans_d_model)

        # 4) Layer normalization and activation
        t = self.norm(t)
        t = self.act(t)

        # 5) Project transformer features to conv channels
        # (batch, seq_len, conv_in_channels)
        conv_tokens = self.transform_to_conv(t)

        # 6) Reshape tokens into 3D volume (batch, C, D, H, W)
        # Ensure seq_len equals self.D * self.H * self.W
        b, s, c = conv_tokens.shape
        assert s == (self.D * self.H * self.W), (
            f"Sequence length (got {s}) must equal D*H*W ({self.D*self.H*self.W})"
        )
        vol = conv_tokens.permute(0, 2, 1).contiguous()  # (batch, C, seq_len)
        vol = vol.view(b, c, self.D, self.H, self.W)     # (batch, C, D, H, W)

        # 7) Upsample with ConvTranspose3d
        up = self.deconv3d(vol)  # (batch, conv_out_channels, D*stride, H*stride, W*stride)

        # 8) Refinement convolution + activation
        out = self.refine_conv(up)
        out = self.act(out)

        return out

def get_inputs():
    """
    Returns:
        list containing a single input tensor shaped (batch_size, seq_len, input_dim)
    """
    x = torch.randn(batch_size, seq_len, input_dim)
    return [x]

def get_init_inputs():
    """
    Returns:
        Initialization parameters used to construct the Model instance.
        Order matches the Model __init__ signature (excluding 'self').
    """
    return [
        input_dim,
        lstm_hidden,
        lstm_layers,
        trans_d_model,
        nhead,
        trans_ffn_dim,
        num_trans_layers,
        conv_in_channels,
        conv_out_channels,
        conv_kernel_size,
        conv_stride,
        conv_padding,
        conv_output_padding,
        D,
        H,
        W,
        bidirectional,
    ]