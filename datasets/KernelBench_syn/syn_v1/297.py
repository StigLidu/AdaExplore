import torch
import torch.nn as nn
import torch.nn.functional as F

# Configuration
batch_size = 32
in_channels = 3
height = 64
width = 64

conv_out_channels = 64
conv_kernel_size = 3
conv_padding = 1

rnn_hidden_size = 128
rnn_num_layers = 2
rnn_non_linearity = 'relu'  # 'tanh' or 'relu'
rnn_batch_first = True

pool_output_size = 8  # adaptive max pool output length
num_classes = 10

class Model(nn.Module):
    """
    Composite model that:
    1. Applies a 2D convolution to an image-like input.
    2. Converts spatial dimensions to a sequence and processes it with a multi-layer RNN.
    3. Applies adaptive 1D max pooling over the temporal dimension produced by the RNN.
    4. Projects the pooled representation to class scores.

    This pipeline exercises Conv2d, RNN, and AdaptiveMaxPool1d in a single forward pass.
    """
    def __init__(self):
        super(Model, self).__init__()
        # Convolution to extract local spatial features
        self.conv = nn.Conv2d(
            in_channels,
            conv_out_channels,
            kernel_size=conv_kernel_size,
            padding=conv_padding,
            bias=True
        )
        # RNN to process the spatial locations as a sequence
        # The RNN input size must match conv_out_channels
        self.rnn = nn.RNN(
            input_size=conv_out_channels,
            hidden_size=rnn_hidden_size,
            num_layers=rnn_num_layers,
            nonlinearity=rnn_non_linearity,
            batch_first=rnn_batch_first,
            bidirectional=False
        )
        # Adaptive pooling over the sequence length produced by the RNN
        self.pool = nn.AdaptiveMaxPool1d(output_size=pool_output_size)
        # Final projection to classes
        self.fc = nn.Linear(rnn_hidden_size * pool_output_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, in_channels, height, width)

        Returns:
            logits: Tensor of shape (batch_size, num_classes)
        """
        # 1) Convolution + non-linearity -> (B, C_out, H, W)
        conv_out = self.conv(x)
        conv_out = F.relu(conv_out)

        # 2) Collapse spatial dims to a sequence length L = H'*W'
        B, C, H_out, W_out = conv_out.shape
        seq_len = H_out * W_out
        # reshape to (B, C, L)
        seq = conv_out.view(B, C, seq_len)

        # 3) Prepare for RNN: RNN expects (B, L, C) when batch_first=True
        seq = seq.permute(0, 2, 1)  # (B, L, C)

        # 4) RNN processing -> outputs of shape (B, L, hidden_size)
        rnn_out, _hn = self.rnn(seq)

        # 5) Pool across temporal dimension: convert to (B, hidden_size, L) for pooling
        rnn_out_t = rnn_out.permute(0, 2, 1)  # (B, hidden_size, L)
        pooled = self.pool(rnn_out_t)  # (B, hidden_size, pool_output_size)

        # 6) Flatten and project to logits
        flattened = pooled.view(B, -1)  # (B, hidden_size * pool_output_size)
        logits = self.fc(flattened)  # (B, num_classes)
        return logits

def get_inputs():
    """
    Returns a list containing a single input tensor suitable for the model:
    A batch of images with shape (batch_size, in_channels, height, width).
    """
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]

def get_init_inputs():
    """
    No external initialization inputs required for this model; parameters are
    initialized inside the Model constructor. Return an empty list to preserve
    interface compatibility.
    """
    return []