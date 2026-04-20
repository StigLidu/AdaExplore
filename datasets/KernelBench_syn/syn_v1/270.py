import torch
import torch.nn as nn

# Configuration
batch_size = 8
seq_len = 10
input_size = 64
hidden_size = 128
channels = 16
height = 32
width = 32
pool_kernel = 4  # must divide height and width evenly

class Model(nn.Module):
    """
    Complex model that processes a temporal sequence with an RNNCell,
    aggregates gated hidden states, projects the aggregation into a 2D
    feature map, and applies spatial average pooling combined with a
    Hardsigmoid gating mechanism to produce the final feature map.

    Forward pipeline:
      - Iterate over time steps with nn.RNNCell to produce hidden states.
      - Compute a per-step gate using nn.Hardsigmoid and accumulate gated hidden.
      - Project the aggregated hidden vector to a (C, H, W) tensor via nn.Linear.
      - Apply nn.AvgPool2d to the projected map.
      - Upsample the pooled map by repeating elements to match original spatial
        resolution and apply an element-wise Hardsigmoid gate.
    """
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        channels: int,
        height: int,
        width: int,
        pool_kernel: int,
    ):
        super(Model, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.channels = channels
        self.height = height
        self.width = width
        self.pool_kernel = pool_kernel

        # Recurrent cell for temporal processing
        self.rnn_cell = nn.RNNCell(input_size, hidden_size, nonlinearity='tanh')

        # Small projection to shape gating signals from hidden states
        self.gate_proj = nn.Linear(hidden_size, hidden_size)

        # Final projection from aggregated hidden to spatial feature map
        self.to_map = nn.Linear(hidden_size, channels * height * width)

        # Spatial pooling and hard-sigmoid nonlinearity
        self.pool = nn.AvgPool2d(kernel_size=pool_kernel, stride=pool_kernel)
        self.hardsigmoid = nn.Hardsigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, seq_len, input_size)

        Returns:
            Tensor of shape (batch, channels, height, width) after temporal
            aggregation, projection, pooling and gating.
        """
        batch, seq_len_in, inp_size = x.shape
        assert inp_size == self.input_size, "Input size mismatch"
        assert seq_len_in > 0, "Sequence length must be positive"
        # Initialize hidden and aggregation tensors on same device/dtype as x
        h = x.new_zeros((batch, self.hidden_size))
        agg = x.new_zeros((batch, self.hidden_size))

        # Temporal processing: RNNCell + gated accumulation
        for t in range(seq_len_in):
            xt = x[:, t, :]  # shape (batch, input_size)
            h = self.rnn_cell(xt, h)  # shape (batch, hidden_size)
            # compute gate from hidden (project -> hardsigmoid)
            gate_logits = self.gate_proj(h)  # (batch, hidden_size)
            gate = self.hardsigmoid(gate_logits)  # elementwise in (0,1)
            # gated accumulation (emphasize timesteps with larger gates)
            agg = agg + h * gate

        # Normalize aggregation by sequence length to keep scale stable
        agg = agg / float(seq_len_in)

        # Project aggregated hidden to spatial feature map
        mapped = self.to_map(agg)  # (batch, channels * H * W)
        mapped = mapped.view(batch, self.channels, self.height, self.width)  # (batch,C,H,W)

        # Spatial avg pooling
        pooled = self.pool(mapped)  # (batch,C,H/pool, W/pool)

        # Upsample pooled map by repeating elements to match original H/W
        # This avoids introducing interpolation ops and keeps shapes integer,
        # requires pool_kernel divides height and width.
        upsampled = pooled.repeat_interleave(self.pool_kernel, dim=2).repeat_interleave(
            self.pool_kernel, dim=3
        )  # (batch,C,H,W)

        # Compute final gate from upsampled pooled features and apply
        spatial_gate = self.hardsigmoid(upsampled)  # (batch,C,H,W)
        gated_map = mapped * spatial_gate  # (batch,C,H,W)

        return gated_map


def get_inputs():
    """
    Returns:
        list: [x] where x is a tensor of shape (batch_size, seq_len, input_size)
    """
    x = torch.randn(batch_size, seq_len, input_size)
    return [x]

def get_init_inputs():
    """
    Returns initialization parameters for the Model constructor in the
    order:
      [input_size, hidden_size, channels, height, width, pool_kernel]
    """
    return [input_size, hidden_size, channels, height, width, pool_kernel]