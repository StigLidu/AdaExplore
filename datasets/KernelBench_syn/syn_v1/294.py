import torch
import torch.nn as nn

"""
Complex model combining an RNN encoder, a TransformerDecoder stack, and 3D instance normalization.
Processing pipeline:
 - RNN (encoder) consumes an input sequence.
 - A linear projector maps RNN outputs to the Transformer d_model and acts as memory.
 - A provided target embedding sequence (tgt) is decoded by nn.TransformerDecoder using the RNN memory.
 - The decoder outputs are pooled into a single vector per batch, projected into a 3D spatial tensor,
   normalized with nn.InstanceNorm3d, and finally reduced to class logits.

Structure matches provided examples:
 - Model class inheriting from nn.Module
 - get_inputs() returns runtime inputs
 - get_init_inputs() returns constructor arguments
 - Module-level configuration variables
"""

# Configuration / shapes
batch_size = 8
seq_len = 20
rnn_input_size = 64
rnn_hidden_size = 128
rnn_num_layers = 2
rnn_nonlinearity = "tanh"   # "tanh" or "relu"

d_model = 256                # Transformer embedding size
decoder_layers = 3
nhead = 8
dim_feedforward = 512

tgt_len = 6                  # length of target sequence fed to decoder

# 3D grid dimensions for reconstructing a spatial tensor from decoder outputs
grid_channels = 16
grid_D = 4
grid_H = 8
grid_W = 8

num_classes = 10
inst_norm_eps = 1e-5


class Model(nn.Module):
    """
    Composite model:
      - RNN encoder (nn.RNN)
      - Linear projector to d_model for Transformer memory
      - TransformerDecoder (nn.TransformerDecoder)
      - Aggregate decoder outputs, project to 3D grid, InstanceNorm3d, pool, final classification head
    """
    def __init__(
        self,
        rnn_input_size: int,
        rnn_hidden_size: int,
        rnn_num_layers: int,
        rnn_nonlinearity: str,
        d_model: int,
        decoder_layers: int,
        nhead: int,
        dim_feedforward: int,
        tgt_len: int,
        grid_channels: int,
        grid_D: int,
        grid_H: int,
        grid_W: int,
        num_classes: int,
        inst_norm_eps: float = 1e-5,
    ):
        super(Model, self).__init__()

        # RNN encoder (batch_first=True for convenience)
        self.rnn = nn.RNN(
            input_size=rnn_input_size,
            hidden_size=rnn_hidden_size,
            num_layers=rnn_num_layers,
            nonlinearity=rnn_nonlinearity,
            batch_first=True,
            bidirectional=False,
        )

        # Project RNN output feature dimension to the Transformer d_model
        self.memory_proj = nn.Linear(rnn_hidden_size, d_model)

        # Transformer decoder stack
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            activation="relu",
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=decoder_layers)

        # We expect tgt to already be embeddings of shape (tgt_len, batch, d_model).
        self.tgt_len = tgt_len

        # After decoding, aggregate across target sequence and project into a 3D grid
        self.to_grid = nn.Linear(d_model, grid_channels * grid_D * grid_H * grid_W)

        # Instance normalization over 3D spatial channels
        self.inst_norm3d = nn.InstanceNorm3d(grid_channels, eps=inst_norm_eps, affine=True)

        # Final classifier head: flatten spatial grid and produce class logits
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(grid_channels * grid_D * grid_H * grid_W, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input sequence tensor of shape (batch, seq_len, rnn_input_size).
            tgt: Target embedding tensor for the decoder of shape (tgt_len, batch, d_model).

        Returns:
            logits: Tensor of shape (batch, num_classes).
        """
        # x -> RNN encoder
        # rnn_out: (batch, seq_len, rnn_hidden_size)
        rnn_out, _h_n = self.rnn(x)

        # Project encoder outputs to d_model to serve as memory for the decoder
        # memory expected shape for Transformer: (seq_len, batch, d_model)
        memory = self.memory_proj(rnn_out).transpose(0, 1)

        # Ensure tgt shape is (tgt_len, batch, d_model)
        # Pass through TransformerDecoder
        decoder_output = self.transformer_decoder(tgt, memory)  # (tgt_len, batch, d_model)

        # Aggregate decoder outputs across the target sequence dimension (mean pooling)
        # aggregated: (batch, d_model)
        aggregated = decoder_output.mean(dim=0)

        # Project aggregated vector into a 3D spatial tensor
        grid_flat = self.to_grid(aggregated)  # (batch, grid_channels * D * H * W)
        batch = grid_flat.shape[0]
        grid = grid_flat.view(batch, grid_channels, grid_D, grid_H, grid_W)  # (batch, C, D, H, W)

        # Apply instance normalization (normalizes per-sample per-channel statistics over D*H*W)
        normalized = self.inst_norm3d(grid)

        # Final classification head
        logits = self.classifier(normalized)  # (batch, num_classes)
        return logits


def get_inputs():
    """
    Returns:
      - x: random input sequence (batch, seq_len, rnn_input_size)
      - tgt: random target embeddings (tgt_len, batch, d_model)
    These shapes must match the constructor arguments provided via get_init_inputs().
    """
    x = torch.randn(batch_size, seq_len, rnn_input_size)
    # Transformer decoder expects target shape (tgt_len, batch, d_model)
    tgt = torch.randn(tgt_len, batch_size, d_model)
    return [x, tgt]


def get_init_inputs():
    """
    Return the initialization parameters for constructing Model(...).
    The order matches the __init__ signature used above.
    """
    return [
        rnn_input_size,
        rnn_hidden_size,
        rnn_num_layers,
        rnn_nonlinearity,
        d_model,
        decoder_layers,
        nhead,
        dim_feedforward,
        tgt_len,
        grid_channels,
        grid_D,
        grid_H,
        grid_W,
        num_classes,
        inst_norm_eps,
    ]