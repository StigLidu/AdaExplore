import torch
import torch.nn as nn

"""
A more complex sequence-processing module that:
- Pads the temporal dimension with ConstantPad1d
- Builds local context windows with tensor.unfold
- Feeds each context vector through a GRUCell step-by-step
- Applies Tanhshrink nonlinearity to the recurrent hidden states
- Projects hidden states to an output space with a Linear layer

Structure mirrors provided examples:
- Model class inheriting from nn.Module
- __init__ sets up layers and parameters
- forward implements the computation
- get_inputs returns example input tensors
- get_init_inputs returns initialization parameters
"""

# Configuration variables
batch_size = 8
seq_len = 50
input_size = 64
hidden_size = 128
pad_left = 1
pad_right = 1
kernel_size = 3
pad_value = 0.1
output_size = 64  # final projection size


class Model(nn.Module):
    """
    Sequence model that builds local temporal contexts via padding + unfolding,
    runs a GRUCell across time on those contexts, applies Tanhshrink to hidden
    states, and projects to an output dimension per time step.
    """
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        kernel_size: int = 3,
        pad_left: int = 1,
        pad_right: int = 1,
        pad_value: float = 0.0,
        output_size: int = None,
    ):
        """
        Initializes the submodules.

        Args:
            input_size (int): Dimensionality of input features per time step.
            hidden_size (int): Size of GRUCell hidden state.
            kernel_size (int): Temporal context window size (must be >=1).
            pad_left (int): Amount of padding on the left (temporal start).
            pad_right (int): Amount of padding on the right (temporal end).
            pad_value (float): Constant value to use for padding.
            output_size (int or None): If provided, project hidden states to this size.
        """
        super(Model, self).__init__()
        assert kernel_size >= 1, "kernel_size must be >= 1"
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.pad_left = pad_left
        self.pad_right = pad_right
        self.pad_value = pad_value
        self.output_size = output_size if output_size is not None else hidden_size

        # Pads temporal (last) dimension for tensors shaped (batch, channels, seq_len)
        self.pad = nn.ConstantPad1d((self.pad_left, self.pad_right), self.pad_value)

        # GRUCell for step-wise recurrent processing
        self.gru_cell = nn.GRUCell(input_size=self.input_size, hidden_size=self.hidden_size)

        # Nonlinearity applied to the hidden state (elementwise)
        self.tanhshrink = nn.Tanhshrink()

        # Initial projection from an aggregated context to hidden state
        self.init_lin = nn.Linear(self.input_size, self.hidden_size)

        # Final projection from hidden state to desired output size
        self.out_lin = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input of shape (batch, seq_len, input_size)

        Returns:
            torch.Tensor: Output of shape (batch, seq_len, output_size)
        """
        # x: (batch, seq_len, input_size)
        batch, seq_len, input_size = x.shape
        assert input_size == self.input_size, "Input size mismatch"

        # Permute to (batch, channels=input_size, seq_len) to apply ConstantPad1d
        x_perm = x.permute(0, 2, 1)  # (batch, input_size, seq_len)

        # Pad temporal dimension
        x_padded = self.pad(x_perm)  # (batch, input_size, seq_len + pad_left + pad_right)

        # Build sliding windows (contexts) over the temporal dimension
        # Resulting shape: (batch, input_size, seq_len, kernel_size)
        contexts = x_padded.unfold(dimension=2, size=self.kernel_size, step=1)
        # Take the mean across the kernel_size to form a context vector per time step
        # contexts_mean: (batch, input_size, seq_len)
        contexts_mean = contexts.mean(dim=-1)

        # Prepare to iterate over time steps and run GRUCell
        outputs = []
        # Initialize hidden state from a global aggregated context
        # Use mean over time of the contexts_mean as an initial summary
        # init_context: (batch, input_size)
        init_context = contexts_mean.mean(dim=2)
        h = torch.tanh(self.init_lin(init_context))  # (batch, hidden_size)

        # Step through each time step
        for t in range(seq_len):
            # context at time t: (batch, input_size)
            context_t = contexts_mean[:, :, t]

            # GRUCell expects (batch, input_size) and (batch, hidden_size)
            h = self.gru_cell(context_t, h)  # update hidden state

            # Apply Tanhshrink nonlinearity to hidden state elementwise
            h_shrunk = self.tanhshrink(h)

            # Project to output space
            out_t = self.out_lin(h_shrunk)  # (batch, output_size)

            outputs.append(out_t.unsqueeze(1))  # collect along time dim

        # Concatenate along time: (batch, seq_len, output_size)
        outputs = torch.cat(outputs, dim=1)
        return outputs


def get_inputs():
    """
    Returns a list with a single random input tensor:
    - x: shape (batch_size, seq_len, input_size)
    """
    x = torch.randn(batch_size, seq_len, input_size)
    return [x]


def get_init_inputs():
    """
    Returns initialization parameters in the order expected by Model.__init__:
    [input_size, hidden_size, kernel_size, pad_left, pad_right, pad_value, output_size]
    """
    return [input_size, hidden_size, kernel_size, pad_left, pad_right, pad_value, output_size]