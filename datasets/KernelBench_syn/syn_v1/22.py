import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List

class Model(nn.Module):
    """
    Sequence processing model that combines a lazily-initialized 1D convolution,
    activation, average pooling, and iterative GRUCell processing across the
    time dimension. The model projects the GRU hidden states back to the
    convolutional feature space at each pooled timestep and returns a
    reconstructed sequence.

    Forward pipeline:
      1. LazyConv1d (in_channels inferred on first forward)
      2. ReLU activation
      3. AvgPool1d to reduce temporal resolution
      4. Iterate over pooled timesteps:
         - Update hidden state with GRUCell using feature vector at timestep
         - Project hidden state back to conv feature space with a linear layer
      5. Stack projected outputs to produce (batch, conv_out_channels, pooled_length)
    """
    def __init__(
        self,
        conv_out_channels: int,
        conv_kernel_size: int,
        pool_kernel: int,
        gru_hidden: int,
    ):
        """
        Args:
            conv_out_channels (int): Number of output channels for the Conv1d layer.
            conv_kernel_size (int): Kernel size for Conv1d.
            pool_kernel (int): Kernel size (and stride) for AvgPool1d.
            gru_hidden (int): Hidden size for the GRUCell.
        """
        super(Model, self).__init__()
        # LazyConv1d will infer in_channels from the first input tensor
        self.conv = nn.LazyConv1d(out_channels=conv_out_channels, kernel_size=conv_kernel_size, stride=1)
        self.act = nn.ReLU()
        # Average pooling reduces temporal length; stride equals kernel for non-overlapping pooling
        self.pool = nn.AvgPool1d(kernel_size=pool_kernel, stride=pool_kernel)
        # GRUCell consumes the conv_out_channels per timestep and updates a hidden state of size gru_hidden
        self.gru_cell = nn.GRUCell(input_size=conv_out_channels, hidden_size=gru_hidden)
        # Project GRU hidden state back to conv feature space so we can reconstruct per-timestep features
        self.proj = nn.Linear(gru_hidden, conv_out_channels)

    def forward(self, x: torch.Tensor, h0: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, seq_len).
            h0 (Optional[torch.Tensor]): Optional initial hidden state of shape (batch_size, gru_hidden).
                                         If None, zeros are used.

        Returns:
            torch.Tensor: Reconstructed sequence tensor of shape (batch_size, conv_out_channels, pooled_length).
        """
        # x: (B, C_in, L)
        conv_out = self.conv(x)               # (B, conv_out_channels, L_conv)
        activated = self.act(conv_out)        # (B, conv_out_channels, L_conv)
        pooled = self.pool(activated)         # (B, conv_out_channels, L_pooled)

        B, feat_dim, Lp = pooled.shape

        # Initialize hidden state if not provided
        if h0 is None:
            h = torch.zeros(B, self.gru_cell.hidden_size, dtype=pooled.dtype, device=pooled.device)
        else:
            h = h0
            if h.shape != (B, self.gru_cell.hidden_size):
                raise ValueError(f"h0 must have shape (batch_size, {self.gru_cell.hidden_size}), got {h.shape}")

        outputs: List[torch.Tensor] = []
        # Iterate over temporal dimension; at each step, use the feature vector as input to GRUCell
        # and project the resulting hidden state back to feature space.
        for t in range(Lp):
            # input_t: (B, feat_dim)
            input_t = pooled[:, :, t]
            h = self.gru_cell(input_t, h)            # (B, gru_hidden)
            proj_t = self.proj(h)                    # (B, conv_out_channels)
            outputs.append(proj_t.unsqueeze(2))      # (B, conv_out_channels, 1)

        # Concatenate along the temporal dimension to form (B, conv_out_channels, L_pooled)
        recon = torch.cat(outputs, dim=2)
        return recon

# Configuration / default sizes
batch_size = 8
in_channels = 12   # Will be inferred by LazyConv1d; provided here for input generation
seq_len = 1024
conv_out_channels = 64
conv_kernel = 9
pool_kernel = 4
gru_hidden = 128

def get_inputs():
    """
    Returns a list containing the input sequence tensor.
    Shape: (batch_size, in_channels, seq_len)
    """
    x = torch.randn(batch_size, in_channels, seq_len)
    return [x]

def get_init_inputs():
    """
    Returns a list of initialization parameters to construct the Model.
    Order: conv_out_channels, conv_kernel_size, pool_kernel, gru_hidden
    """
    return [conv_out_channels, conv_kernel, pool_kernel, gru_hidden]