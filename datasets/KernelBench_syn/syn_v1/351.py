import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Sequence-to-label model that combines a lazily-initialized 2D convolutional front-end
    with a multi-layer Elman RNN and a small RNNCell-based iterative refinement of the final hidden state.

    Computational pattern:
      1. Apply LazyConv2d independently to each frame in the input sequence (batch*seq as batch).
      2. Flatten spatial features and feed sequence into nn.RNN (multi-layer).
      3. Extract the top-layer final hidden state.
      4. Build a compact context vector by spatially averaging conv features across frames.
      5. Iteratively refine the final RNN hidden state using nn.RNNCell with the context vector.
      6. Map the refined hidden state to class logits with a linear classifier.

    This pattern mixes convolutional spatial processing, sequential modeling, and recurrent refinement.
    """
    def __init__(
        self,
        seq_len: int,
        height: int,
        width: int,
        rnn_hidden_size: int = 256,
        rnn_layers: int = 2,
        out_channels: int = 32,
        refine_steps: int = 3,
        num_classes: int = 10
    ):
        """
        Args:
            seq_len (int): Length of the input sequence (number of frames).
            height (int): Height of each input frame.
            width (int): Width of each input frame.
            rnn_hidden_size (int): Hidden size for the multi-layer RNN and the RNNCell.
            rnn_layers (int): Number of layers for the nn.RNN.
            out_channels (int): Number of output channels from the convolutional front-end.
            refine_steps (int): Number of iterative refinement steps using the RNNCell.
            num_classes (int): Number of output classes for classification logits.
        """
        super(Model, self).__init__()
        self.seq_len = seq_len
        self.height = height
        self.width = width
        self.out_channels = out_channels
        self.rnn_hidden_size = rnn_hidden_size
        self.rnn_layers = rnn_layers
        self.refine_steps = refine_steps
        self.num_classes = num_classes

        # Lazily infer in_channels from the first forward pass.
        # This conv extracts spatial features per frame.
        self.conv = nn.LazyConv2d(out_channels=out_channels, kernel_size=3, padding=1)

        # The RNN consumes flattened spatial features per frame.
        rnn_input_size = out_channels * height * width
        self.rnn = nn.RNN(
            input_size=rnn_input_size,
            hidden_size=rnn_hidden_size,
            num_layers=rnn_layers,
            nonlinearity='tanh',
            batch_first=False,  # we'll feed (seq, batch, features)
        )

        # A small RNNCell that will iteratively refine the final hidden state using a context vector.
        # The context vector is computed from averaged conv features per sequence (size = out_channels).
        self.refine_cell = nn.RNNCell(input_size=out_channels, hidden_size=rnn_hidden_size)

        # Final linear classifier maps refined hidden state to class logits.
        self.classifier = nn.Linear(rnn_hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, seq_len, channels, height, width)

        Returns:
            torch.Tensor: Classification logits of shape (batch_size, num_classes)
        """
        batch_size, seq_len, channels, H, W = x.shape
        assert seq_len == self.seq_len, f"Expected sequence length {self.seq_len}, got {seq_len}"
        assert H == self.height and W == self.width, f"Expected spatial size ({self.height},{self.width}), got ({H},{W})"

        # Merge batch and sequence to process all frames through conv in one go.
        x_flat = x.view(batch_size * seq_len, channels, H, W)  # (batch*seq, C, H, W)
        conv_out = self.conv(x_flat)  # (batch*seq, out_channels, H, W)

        # Restore sequence dimension
        conv_feats = conv_out.view(batch_size, seq_len, self.out_channels, H, W)  # (batch, seq, out_ch, H, W)

        # Prepare RNN inputs: flatten spatial dims and move seq to first dim
        rnn_in = conv_feats.view(batch_size, seq_len, -1).permute(1, 0, 2)  # (seq, batch, out_ch*H*W)

        # Pass through multi-layer RNN
        rnn_out, h_n = self.rnn(rnn_in)  # rnn_out: (seq, batch, hidden), h_n: (num_layers, batch, hidden)

        # Take the final hidden state from the topmost RNN layer
        h_last = h_n[-1]  # (batch, hidden)

        # Build a compact context by spatially averaging conv features and then averaging across time:
        # conv_feats: (batch, seq, out_ch, H, W) -> mean spatially -> (batch, seq, out_ch) -> mean over seq -> (batch, out_ch)
        spatial_avg = conv_feats.mean(dim=[3, 4])  # (batch, seq, out_ch)
        context_vec = spatial_avg.mean(dim=1)      # (batch, out_ch)

        # Iteratively refine the last hidden state using the RNNCell and the fixed context vector
        h_refined = h_last
        for _ in range(self.refine_steps):
            # RNNCell: input (batch, out_ch), hx (batch, hidden) -> (batch, hidden)
            h_refined = self.refine_cell(context_vec, h_refined)

        # Map refined hidden state to class logits
        logits = self.classifier(h_refined)  # (batch, num_classes)
        return logits

# Configuration / default sizes for inputs and initialization
batch_size = 4
seq_len = 12
channels = 3
height = 16
width = 16
rnn_hidden_size = 256
rnn_layers = 2
out_channels = 32
refine_steps = 4
num_classes = 20

def get_inputs():
    """
    Returns a list with a single input tensor of shape (batch_size, seq_len, channels, height, width).
    """
    x = torch.randn(batch_size, seq_len, channels, height, width)
    return [x]

def get_init_inputs():
    """
    Returns the initialization arguments for Model.__init__ in the order:
      [seq_len, height, width, rnn_hidden_size, rnn_layers, out_channels, refine_steps, num_classes]
    """
    return [seq_len, height, width, rnn_hidden_size, rnn_layers, out_channels, refine_steps, num_classes]