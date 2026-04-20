import torch
import torch.nn as nn

"""
Complex composed model combining:
- 1D replication padding over a temporal signal (nn.ReplicationPad1d)
- 2D nearest-neighbor upsampling of an image (nn.UpsamplingNearest2d)
- temporal context aggregation via local averaging (implemented with slicing + mean)
- recurrent processing across time using an RNN cell (nn.RNNCell)
- final linear projection per time-step

The model takes:
- temporal input `temp` of shape (batch, temp_channels, seq_len)
- image input `img` of shape (batch, img_channels, H, W)

It returns:
- output tensor of shape (batch, seq_len, out_dim)
"""

# Configuration variables
batch_size = 8
seq_len = 16
temp_channels = 12
img_channels = 6
H = 32
W = 32
upsample_scale = 2  # scale factor for UpsamplingNearest2d
rnn_hidden = 64
out_dim = 10
context_kernel = 3  # local temporal context window size (must be odd for symmetric padding)


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # Replication pad to provide temporal context at sequence boundaries
        pad_left = context_kernel // 2
        pad_right = context_kernel - 1 - pad_left
        self.pad = nn.ReplicationPad1d((pad_left, pad_right))

        # Upsampling for image modality
        self.upsample = nn.UpsamplingNearest2d(scale_factor=upsample_scale)

        # Spatial pooling to convert upsampled image into a compact per-sample descriptor
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # RNNCell that consumes concatenated (temporal context vector + image descriptor)
        rnn_input_size = temp_channels + img_channels
        self.rnn_cell = nn.RNNCell(input_size=rnn_input_size, hidden_size=rnn_hidden, nonlinearity='tanh')

        # Final projection applied to each time-step's hidden state
        self.out_proj = nn.Linear(rnn_hidden, out_dim)

        # Store kernel size for forward slicing
        self.context_kernel = context_kernel

    def forward(self, temp: torch.Tensor, img: torch.Tensor) -> torch.Tensor:
        """
        Args:
            temp: Temporal input, shape (batch, temp_channels, seq_len)
            img: Image input, shape (batch, img_channels, H, W)

        Returns:
            per-step outputs: Tensor of shape (batch, seq_len, out_dim)
        """
        # Validate shapes (lightweight checks)
        assert temp.dim() == 3, "temp must be (B, C_t, L)"
        assert img.dim() == 4, "img must be (B, C_i, H, W)"
        B, C_t, L = temp.shape
        B2, C_i, H_i, W_i = img.shape
        assert B == B2, "Batch sizes of temp and img must match"
        assert C_t == temp_channels and C_i == img_channels, "Input channels must match config"

        # 1) Pad temporal signal to allow context window at edges
        temp_p = self.pad(temp)  # shape: (B, C_t, L + pad_left + pad_right)

        # 2) Upsample image and pool to a per-sample descriptor
        img_up = self.upsample(img)             # (B, C_i, H*scale, W*scale)
        img_pooled = self.pool(img_up)          # (B, C_i, 1, 1)
        img_feat = img_pooled.view(B, C_i)      # (B, C_i)

        # We'll reuse img_feat for every time-step (same image descriptor broadcast across time)
        # 3) Recurrently process each time-step combining temporal context and image descriptor
        hidden = torch.zeros(B, rnn_hidden, device=temp.device, dtype=temp.dtype)
        outputs = []
        k = self.context_kernel
        for t in range(L):
            # Extract local temporal context window and average across time dimension
            # temp_p shape allows slicing [t : t+k] safely
            context = temp_p[:, :, t:t + k].mean(dim=2)  # (B, C_t)

            # Concatenate context vector with image descriptor -> RNN input
            rnn_in = torch.cat([context, img_feat], dim=1)  # (B, C_t + C_i)

            # Update hidden state with RNNCell
            hidden = self.rnn_cell(rnn_in, hidden)  # (B, rnn_hidden)

            # Project hidden to output dimension for this time-step
            step_out = self.out_proj(hidden)  # (B, out_dim)
            outputs.append(step_out)

        # Stack outputs into shape (seq_len, B, out_dim) -> then transpose to (B, seq_len, out_dim)
        outputs = torch.stack(outputs, dim=0).transpose(0, 1).contiguous()  # (B, L, out_dim)
        return outputs


def get_inputs():
    """
    Returns:
        list: [temp, img] with shapes:
            temp: (batch_size, temp_channels, seq_len)
            img:  (batch_size, img_channels, H, W)
    """
    temp = torch.randn(batch_size, temp_channels, seq_len, dtype=torch.float32)
    img = torch.randn(batch_size, img_channels, H, W, dtype=torch.float32)
    return [temp, img]


def get_init_inputs():
    """
    No special initialization parameters are required for this model.
    """
    return []