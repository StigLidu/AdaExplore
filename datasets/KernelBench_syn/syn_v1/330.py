import torch
import torch.nn as nn

# Configuration / hyper-parameters
BATCH = 8
SEQ_LEN = 16            # T
INPUT_DIM = 64          # D
LSTM_HIDDEN = 32
LSTM_LAYERS = 2
CONV_OUT_CHANNELS = 64
CONV_KERNEL = 4
CONV_STRIDE = 2
CONV_PADDING = 1
# After ConvTranspose1d with the settings above, output length = 2 * SEQ_LEN
# Choose spatial layout that factors this length
SPATIAL_H = 8
SPATIAL_W = 4          # SPATIAL_H * SPATIAL_W must equal Conv output length (2 * SEQ_LEN)
NUM_CLASSES = 10

class Model(nn.Module):
    """
    Complex model combining an LSTM, a ConvTranspose1d upsampler, and a spatial Softmax.
    Pipeline:
      1. Sequence input (B, T, D) -> LSTM (batch_first) -> (B, T, H)
      2. Permute to (B, H, T) and apply ConvTranspose1d to upsample in the temporal axis -> (B, C, L)
      3. Reshape L into spatial dimensions (H_spatial, W_spatial) -> (B, C, H_sp, W_sp)
      4. Apply Softmax2d (softmax over channels at each spatial location)
      5. Use softmax as attention weights to compute a channel-weighted spatial map -> (B, H_sp, W_sp)
      6. Flatten and linearly project to NUM_CLASSES logits
    """
    def __init__(self):
        super(Model, self).__init__()
        # LSTM processes the input sequence
        self.lstm = nn.LSTM(
            input_size=INPUT_DIM,
            hidden_size=LSTM_HIDDEN,
            num_layers=LSTM_LAYERS,
            batch_first=True,
            bidirectional=False
        )
        # ConvTranspose1d upsamples the temporal axis; in_channels must match LSTM hidden size
        self.conv_t = nn.ConvTranspose1d(
            in_channels=LSTM_HIDDEN,
            out_channels=CONV_OUT_CHANNELS,
            kernel_size=CONV_KERNEL,
            stride=CONV_STRIDE,
            padding=CONV_PADDING
        )
        # Softmax2d applies softmax across channel dim for each spatial location
        self.softmax2d = nn.Softmax2d()
        # Final linear projection implemented via parameters (no explicit nn.Linear to show manual projection)
        flat_spatial = SPATIAL_H * SPATIAL_W
        self.class_weight = nn.Parameter(torch.randn(flat_spatial, NUM_CLASSES))
        self.class_bias = nn.Parameter(torch.randn(NUM_CLASSES))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, T, D)

        Returns:
            logits: Tensor of shape (B, NUM_CLASSES)
        """
        # 1) LSTM -> (B, T, H)
        lstm_out, _ = self.lstm(x)  # lstm_out: (B, T, LSTM_HIDDEN)

        # 2) Prepare for ConvTranspose1d: (B, H, T)
        t = lstm_out.permute(0, 2, 1)

        # 3) ConvTranspose1d upsample along temporal axis -> (B, CONV_OUT_CHANNELS, L_out)
        conv_out = self.conv_t(t)

        # Validate that conv_out length matches expected spatial factorization
        B, C, L_out = conv_out.shape
        expected_L = SPATIAL_H * SPATIAL_W
        if L_out != expected_L:
            # If shapes mismatch, adapt by either trimming or padding (deterministic trimming here)
            if L_out > expected_L:
                conv_out = conv_out[:, :, :expected_L]
                L_out = expected_L
            else:
                # pad at the end with zeros
                pad_amount = expected_L - L_out
                pad_tensor = torch.zeros(B, C, pad_amount, device=conv_out.device, dtype=conv_out.dtype)
                conv_out = torch.cat([conv_out, pad_tensor], dim=2)
                L_out = expected_L

        # 4) Reshape to 4D for Softmax2d: (B, C, H_sp, W_sp)
        conv_4d = conv_out.view(B, C, SPATIAL_H, SPATIAL_W)

        # 5) Softmax2d applies softmax across channels for each (h, w)
        attention = self.softmax2d(conv_4d)  # (B, C, H_sp, W_sp)

        # 6) Weighted sum across channels using attention as weights -> (B, H_sp, W_sp)
        weighted_spatial = (attention * conv_4d).sum(dim=1)  # sum over channels

        # 7) Flatten spatial map and project to logits
        flat = weighted_spatial.view(B, -1)  # (B, SPATIAL_H * SPATIAL_W)
        logits = flat.matmul(self.class_weight) + self.class_bias  # (B, NUM_CLASSES)

        return logits

def get_inputs():
    """
    Creates representative inputs for the model:
        - x: random float tensor of shape (BATCH, SEQ_LEN, INPUT_DIM)
    """
    x = torch.randn(BATCH, SEQ_LEN, INPUT_DIM)
    return [x]

def get_init_inputs():
    """
    No special init parameters required externally (model defines its own parameters).
    """
    return []