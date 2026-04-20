import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Complex sequence-to-feature extractor that:
      - Processes an input sequence with a GRUCell (step-by-step recursion).
      - Applies circular padding along the temporal dimension.
      - Interprets the padded temporal axis as the depth dimension of a 3D tensor
        and upsamples it with a LazyConvTranspose3d.
      - Projects the upsampled per-timestep features back to the input feature space,
        computes attention weights over the (upsampled) time axis, and returns a
        weighted temporal summary.

    The module demonstrates a non-trivial interplay between recurrent processing,
    circular padding, lazy convolution transpose (3D), and a small attention mechanism.
    """
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        conv_out_channels: int,
        pad: int = 1,
        kernel_depth: int = 3,
        stride_depth: int = 2,
    ):
        """
        Args:
            input_size (int): Dimensionality of input features per time step.
            hidden_size (int): Hidden size of the GRUCell.
            conv_out_channels (int): out_channels for LazyConvTranspose3d.
            pad (int): Circular padding amount on both sides of temporal axis.
            kernel_depth (int): Kernel size along the depth (temporal) axis for ConvTranspose3d.
            stride_depth (int): Stride along the depth axis (controls upsampling factor).
        """
        super(Model, self).__init__()

        # Recurrent element: processes inputs step-by-step.
        self.gru_cell = nn.GRUCell(input_size=input_size, hidden_size=hidden_size)

        # Circular padding along temporal axis; applied to tensors shaped (N, C, L).
        self.circ_pad = nn.CircularPad1d(pad)

        # LazyConvTranspose3d: in_channels will be inferred on first forward.
        # We use kernel/stride only along the depth axis; height and width are 1.
        self.conv_transpose = nn.LazyConvTranspose3d(
            out_channels=conv_out_channels,
            kernel_size=(kernel_depth, 1, 1),
            stride=(stride_depth, 1, 1),
            padding=(kernel_depth // 2, 0, 0),
            output_padding=(stride_depth - 1, 0, 0),
        )

        # Project conv outputs (per-timestep features) back to input feature dimensionality.
        self.post_fc = nn.Linear(conv_out_channels, input_size)

        # Small module to compute per-timestep energy scores for attention (scalar per timestep).
        self.energy = nn.Linear(input_size, 1)

        # Nonlinearity
        self.activation = nn.ReLU()

        # Store config
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.pad = pad
        self.kernel_depth = kernel_depth
        self.stride_depth = stride_depth
        self.conv_out_channels = conv_out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_size)
        Returns:
            torch.Tensor: Weighted summary over upsampled time axis, shape (batch_size, input_size)
        """
        batch_size, seq_len, _ = x.shape

        # 1) Step through time with a GRUCell to build sequence of hidden states.
        h = x.new_zeros(batch_size, self.hidden_size)
        h_seq = []
        for t in range(seq_len):
            h = self.gru_cell(x[:, t, :], h)  # (batch_size, hidden_size)
            h_seq.append(h)
        # Stack to (batch, seq_len, hidden_size)
        h_seq = torch.stack(h_seq, dim=1)

        # 2) Prepare for 1D circular padding: permute to (batch, channels, length)
        h_seq_perm = h_seq.permute(0, 2, 1)  # (batch, hidden_size, seq_len)

        # 3) Circular pad along temporal axis -> (batch, hidden_size, seq_len + 2*pad)
        padded = self.circ_pad(h_seq_perm)

        # 4) Interpret temporal axis as depth for a 3D conv transpose: make (N, C, D, 1, 1)
        conv_input = padded.unsqueeze(-1).unsqueeze(-1)  # (batch, hidden, D, 1, 1)

        # 5) LazyConvTranspose3d upsampling along depth dimension (in_channels inferred)
        conv_out = self.conv_transpose(conv_input)  # (batch, conv_out_channels, D_out, 1, 1)

        # 6) Squeeze spatial singleton dims and permute to (batch, D_out, conv_out_channels)
        conv_out_squeezed = conv_out.squeeze(-1).squeeze(-1).permute(0, 2, 1)

        # 7) Nonlinearity and projection back to input feature dimension: (batch, D_out, input_size)
        proj = self.activation(self.post_fc(conv_out_squeezed))

        # 8) Compute energy scores per timestep and softmax over the temporal axis
        # energy: (batch, D_out, 1) -> squeeze -> (batch, D_out)
        energy = self.energy(proj).squeeze(-1)
        weights = torch.softmax(energy, dim=1)  # (batch, D_out)

        # 9) Weighted temporal sum -> (batch, input_size)
        weights = weights.unsqueeze(-1)  # (batch, D_out, 1)
        weighted_sum = torch.sum(proj * weights, dim=1)

        return weighted_sum


# --- Module-level configuration for tests / instantiation ---
batch_size = 8
seq_len = 16
input_size = 32
hidden_size = 64
conv_out_channels = 32
pad = 2
kernel_depth = 3
stride_depth = 2

def get_inputs():
    """
    Returns a list with a single input tensor suitable for the Model.forward call.
    Shape: (batch_size, seq_len, input_size)
    """
    x = torch.randn(batch_size, seq_len, input_size)
    return [x]

def get_init_inputs():
    """
    Returns initialization parameters for Model:
      [input_size, hidden_size, conv_out_channels, pad, kernel_depth, stride_depth]
    """
    return [input_size, hidden_size, conv_out_channels, pad, kernel_depth, stride_depth]