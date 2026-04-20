import torch
import torch.nn as nn

# Configuration variables
BATCH_SIZE = 8
CHANNELS = 3
HEIGHT = 30
WIDTH = 30

LP_P = 2               # Lp norm for LPPool2d
POOL_KERNEL = 2        # kernel size for pooling
POOL_STRIDE = 2        # stride for pooling
PAD = 1                # zero padding size (ZeroPad2d takes int for symmetric padding)
PATCH_SIZE = 4         # spatial patch size after pooling (patch_h = patch_w = PATCH_SIZE)
HIDDEN_SIZE = 64       # hidden size for LSTMCell

class Model(nn.Module):
    """
    A composite module that:
    - Pads the input with ZeroPad2d
    - Applies LPPool2d to reduce spatial resolution
    - Extracts spatial patches and treats them as a sequence
    - Runs a single-layer LSTMCell recurrently over the patch sequence
    - Projects LSTM hidden states back to patch vectors and reconstructs the pooled feature map
    The output is the pooled-resolution feature map with a residual connection to the pooled input.
    """
    def __init__(
        self,
        channels: int,
        patch_size: int,
        hidden_size: int,
        lp_p: int = LP_P,
        pool_kernel: int = POOL_KERNEL,
        pool_stride: int = POOL_STRIDE,
        pad: int = PAD,
    ):
        """
        Args:
            channels: Number of channels in the input image.
            patch_size: Height and width of patches taken from the pooled feature map.
            hidden_size: Hidden state size for the LSTMCell.
            lp_p: The p value for LPPool2d.
            pool_kernel: Kernel size for LPPool2d.
            pool_stride: Stride for LPPool2d.
            pad: Padding size for ZeroPad2d (symmetric on all sides).
        """
        super(Model, self).__init__()
        self.channels = channels
        self.patch_size = patch_size
        self.hidden_size = hidden_size

        # Layers
        # ZeroPad2d expects a single int for symmetric padding on all sides
        self.pad = nn.ZeroPad2d(pad)
        self.pool = nn.LPPool2d(norm_type=lp_p, kernel_size=pool_kernel, stride=pool_stride)

        # Input size to LSTMCell will be channels * patch_h * patch_w
        input_size = channels * patch_size * patch_size
        self.lstm_cell = nn.LSTMCell(input_size=input_size, hidden_size=hidden_size)
        # Project from hidden state back to patch vector
        self.project = nn.Linear(hidden_size, input_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass:
        1. Apply zero padding
        2. Apply Lp pooling
        3. Extract non-overlapping patches from pooled feature map (patch_size x patch_size)
        4. Run LSTMCell recurrently over the sequence of flattened patches
        5. Project hidden states back to patch vectors and reconstruct pooled map
        6. Add residual connection from pooled input

        Args:
            x: Input tensor of shape (batch, channels, H, W)

        Returns:
            Tensor of shape (batch, channels, H_pooled, W_pooled) where H_pooled and W_pooled
            are the spatial dimensions after pooling.
        """
        B = x.shape[0]

        # 1. Padding
        x_padded = self.pad(x)  # (B, C, H + 2*pad, W + 2*pad)

        # 2. Lp Pooling
        pooled = self.pool(x_padded)  # (B, C, H_pooled, W_pooled)

        # 3. Extract non-overlapping patches using unfold logic via tensor.unfold
        ph = pw = self.patch_size
        # pooled shape: (B, C, Hp, Wp)
        _, C, Hp, Wp = pooled.shape
        # Ensure divisibility; raise helpful error if not divisible
        if (Hp % ph) != 0 or (Wp % pw) != 0:
            raise ValueError(
                f"Pooled spatial dims (Hp={Hp}, Wp={Wp}) must be divisible by patch_size={ph}."
            )

        num_h = Hp // ph
        num_w = Wp // pw
        # Create patches: (B, C, num_h, num_w, ph, pw)
        patches = pooled.unfold(2, ph, ph).unfold(3, pw, pw)
        # Permute to (B, num_h * num_w, C * ph * pw)
        B_, C_, nh, nw, ph_, pw_ = patches.shape
        patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()  # (B, num_h, num_w, C, ph, pw)
        seq_len = nh * nw
        patches = patches.view(B, seq_len, C * ph * pw)  # (B, seq_len, input_size)

        # 4. Run LSTMCell over sequence of patches
        device = x.device
        dtype = x.dtype
        hx = torch.zeros(B, self.hidden_size, device=device, dtype=dtype)
        cx = torch.zeros(B, self.hidden_size, device=device, dtype=dtype)

        hidden_states = []
        # iterate over sequence length (time steps)
        for t in range(seq_len):
            input_t = patches[:, t, :]  # (B, input_size)
            hx, cx = self.lstm_cell(input_t, (hx, cx))  # both (B, hidden_size)
            hidden_states.append(hx.unsqueeze(1))  # keep time dimension

        # Concatenate hidden states: (B, seq_len, hidden_size)
        hidden_seq = torch.cat(hidden_states, dim=1)

        # 5. Project hidden states back to patch vectors
        projected = self.project(hidden_seq)  # (B, seq_len, input_size)

        # Reshape back to spatial layout:
        projected = projected.view(B, num_h, num_w, C, ph, pw)  # (B, num_h, num_w, C, ph, pw)
        # Rearrange to (B, C, num_h*ph, num_w*pw) == (B, C, Hp, Wp)
        reconstructed = projected.permute(0, 3, 1, 4, 2, 5).contiguous().view(B, C, Hp, Wp)

        # 6. Residual addition with pooled features and return
        out = pooled + reconstructed
        return out


def get_inputs():
    """
    Returns:
        List containing a single input tensor of shape (BATCH_SIZE, CHANNELS, HEIGHT, WIDTH)
    """
    x = torch.randn(BATCH_SIZE, CHANNELS, HEIGHT, WIDTH)
    return [x]


def get_init_inputs():
    """
    Returns the list of initialization parameters for constructing the Model:
    [channels, patch_size, hidden_size, lp_p, pool_kernel, pool_stride, pad]
    """
    return [CHANNELS, PATCH_SIZE, HIDDEN_SIZE, LP_P, POOL_KERNEL, POOL_STRIDE, PAD]