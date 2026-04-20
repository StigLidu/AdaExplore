import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Complex multi-modal module that processes an image-like tensor with InstanceNorm2d,
    a convolution, and LocalResponseNorm, and concurrently processes a sequence with an RNN.
    The spatial features are pooled and fused with the RNN's last time-step output, then
    projected to a final prediction vector.

    Computation steps:
      1. InstanceNorm2d over image input
      2. 2D convolution (preserves spatial dims with padding)
      3. LocalResponseNorm for lateral inhibition-style normalization
      4. ReLU activation + AdaptiveAvgPool2d to produce per-batch feature vector
      5. RNN over sequence input (batch_first=True)
      6. Concatenate image and sequence embeddings
      7. Final linear projection (with optional LogSoftmax for stable outputs)
    """
    def __init__(
        self,
        in_channels: int,
        rnn_input_size: int,
        rnn_hidden_size: int,
        rnn_num_layers: int,
        conv_out_channels: int,
        fc_out_features: int,
    ):
        """
        Initializes the multimodal model.

        Args:
            in_channels: Number of channels in the image input (for InstanceNorm2d & Conv2d).
            rnn_input_size: Feature size per time-step for the RNN input.
            rnn_hidden_size: Hidden size of the RNN.
            rnn_num_layers: Number of stacked RNN layers.
            conv_out_channels: Output channels of the convolution applied to the image.
            fc_out_features: Number of output features from the final linear layer.
        """
        super(Model, self).__init__()
        # Image branch
        self.inst_norm = nn.InstanceNorm2d(num_features=in_channels, affine=True, track_running_stats=False)
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=conv_out_channels, kernel_size=3, padding=1)
        # Local Response Normalization (classic AlexNet-style normalization)
        self.lrn = nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2.0)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))  # produce a (batch, conv_out_channels, 1, 1) tensor

        # Sequence branch
        self.rnn = nn.RNN(
            input_size=rnn_input_size,
            hidden_size=rnn_hidden_size,
            num_layers=rnn_num_layers,
            nonlinearity='tanh',
            batch_first=True,
        )

        # Fusion and projection
        fused_dim = conv_out_channels + rnn_hidden_size
        self.fc = nn.Linear(fused_dim, fc_out_features)
        # Optionally keep softmax/log-probabilities outside; here we return raw logits.

    def forward(self, img: torch.Tensor, seq: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            img: Image-like tensor of shape (batch, in_channels, H, W).
            seq: Sequence tensor of shape (batch, seq_len, rnn_input_size) when batch_first=True.

        Returns:
            Tensor of shape (batch, fc_out_features) with fused predictions.
        """
        # Image branch: InstanceNorm -> Conv2d -> LRN -> ReLU -> Global pool -> flatten
        x = self.inst_norm(img)                              # (B, C, H, W)
        x = self.conv(x)                                     # (B, conv_out, H, W)
        x = self.lrn(x)                                      # (B, conv_out, H, W)
        x = torch.relu(x)                                    # non-linearity
        x = self.pool(x)                                     # (B, conv_out, 1, 1)
        x = x.view(x.size(0), -1)                            # (B, conv_out)

        # Sequence branch: RNN over time -> take last output (last time-step)
        rnn_out, h_n = self.rnn(seq)                         # rnn_out: (B, seq_len, hidden), h_n: (num_layers, B, hidden)
        # Use the last time-step output for the sequence embedding
        seq_feat = rnn_out[:, -1, :]                         # (B, hidden)

        # Fuse features and project
        fused = torch.cat([x, seq_feat], dim=1)              # (B, conv_out + hidden)
        out = self.fc(fused)                                 # (B, fc_out_features)
        return out

# Module-level configuration (example sizes)
batch_size = 8
in_channels = 3
height = 64
width = 96

seq_len = 20
rnn_input_size = 40
rnn_hidden_size = 128
rnn_num_layers = 2

conv_out_channels = 64
fc_out_features = 10

def get_inputs():
    """
    Creates realistic random inputs for the model:
      - img: (batch_size, in_channels, height, width)
      - seq: (batch_size, seq_len, rnn_input_size)
    """
    img = torch.randn(batch_size, in_channels, height, width)
    seq = torch.randn(batch_size, seq_len, rnn_input_size)
    return [img, seq]

def get_init_inputs():
    """
    Returns the initialization parameters used to construct the Model instance in the
    same order as Model.__init__ expects them.
    """
    return [in_channels, rnn_input_size, rnn_hidden_size, rnn_num_layers, conv_out_channels, fc_out_features]