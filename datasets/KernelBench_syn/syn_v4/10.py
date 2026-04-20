import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Complex 1D-windowed projection model that demonstrates a mix of padding, window extraction,
    learned linear projection, lazy batch normalization, and nonlinear activation.

    Computation pipeline:
      1. Zero-pad the input along the temporal dimension (ZeroPad1d).
      2. Extract sliding windows using Tensor.unfold to form local context vectors.
      3. Apply a learned linear projection (matrix multiplication) on each window.
      4. Apply LazyBatchNorm1d across the projected feature channels.
      5. Apply Tanh nonlinearity.
      6. Global average pool across windows and apply a final linear projection to produce output.
    """
    def __init__(self):
        super(Model, self).__init__()

        # Layers from the provided list
        self.pad = nn.ZeroPad1d((PADDING_LEFT, PADDING_RIGHT))  # pads the last dimension
        self.bn = nn.LazyBatchNorm1d()  # lazy initialization of feature dimension
        self.tanh = nn.Tanh()

        # Learned projection weights
        in_dim = CHANNELS * KERNEL_SIZE
        proj_out = OUT_FEATURES
        self.W = nn.Parameter(torch.empty(in_dim, proj_out))
        self.V = nn.Parameter(torch.empty(proj_out, FINAL_DIM))

        # Parameter initialization
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.V)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, channels, seq_len)

        Returns:
            Tensor of shape (batch_size, FINAL_DIM)
        """
        # 1) Zero-pad temporal dimension
        x_padded = self.pad(x)  # (B, C, L + left + right)

        # 2) Extract sliding windows -> shape (B, C, num_windows, KERNEL_SIZE)
        windows = x_padded.unfold(dimension=2, size=KERNEL_SIZE, step=STRIDE)

        # 3) Rearrange to (B, num_windows, C * KERNEL_SIZE)
        B, C, num_windows, k = windows.shape
        windows = windows.permute(0, 2, 1, 3).contiguous()  # (B, num_windows, C, K)
        windows_flat = windows.view(B, num_windows, C * k)   # (B, num_windows, in_dim)

        # 4) Linear projection for every window
        # (B, num_windows, in_dim) @ (in_dim, proj_out) -> (B, num_windows, proj_out)
        projected = torch.matmul(windows_flat, self.W)

        # 5) Rearrange to (B, proj_out, num_windows) for BatchNorm1d
        projected = projected.permute(0, 2, 1).contiguous()

        # 6) Batch normalization (lazy - initializes on first call)
        normalized = self.bn(projected)

        # 7) Nonlinearity
        activated = self.tanh(normalized)

        # 8) Global average pooling across windows -> (B, proj_out)
        pooled = activated.mean(dim=2)

        # 9) Final linear projection -> (B, FINAL_DIM)
        out = torch.matmul(pooled, self.V)

        return out

# Module-level configuration variables
BATCH_SIZE = 32
CHANNELS = 16
SEQ_LEN = 128

KERNEL_SIZE = 5
STRIDE = 2
PADDING_LEFT = 2
PADDING_RIGHT = 2

OUT_FEATURES = 64
FINAL_DIM = 10

def get_inputs():
    """
    Returns:
        list containing a single input tensor of shape (BATCH_SIZE, CHANNELS, SEQ_LEN)
    """
    x = torch.randn(BATCH_SIZE, CHANNELS, SEQ_LEN)
    return [x]

def get_init_inputs():
    """
    No special initialization inputs required; model parameters are internally initialized.
    """
    return []