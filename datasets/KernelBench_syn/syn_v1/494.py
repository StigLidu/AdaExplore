import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Complex 1D feature modulation + channel mixing module.

    Pipeline:
    1. Apply Instance Normalization across (C, L) per instance.
    2. Aggregate temporal context via mean over length -> (B, C).
    3. Pass context through a small MLP (Linear -> ReLU6 -> Linear).
    4. Apply LogSigmoid to MLP output and exponentiate to recover a gating vector in (0,1).
    5. Gate the normalized features (channel-wise modulation).
    6. Apply a learned channel-mixing matrix to the gated features (mix channels at each time step).
    7. Combine mixed result with a learnable per-channel residual scale and final ReLU6 activation.

    This model demonstrates use of nn.InstanceNorm1d, nn.ReLU6, and nn.LogSigmoid,
    and performs non-trivial tensor manipulations (broadcasting, einsum-based channel mixing).
    """
    def __init__(self, channels: int, reduction: int = 4):
        """
        Args:
            channels (int): Number of channels (C) in the input tensor.
            reduction (int): Reduction factor for the hidden dimension of the gating MLP.
        """
        super(Model, self).__init__()
        self.channels = channels
        hidden = max(1, channels // reduction)

        # Normalization across (C, L) for each instance
        self.inst_norm = nn.InstanceNorm1d(num_features=channels, affine=True)

        # Small gating MLP: C -> hidden -> C
        self.fc1 = nn.Linear(channels, hidden, bias=True)
        self.relu6 = nn.ReLU6(inplace=True)
        self.fc2 = nn.Linear(hidden, channels, bias=True)

        # LogSigmoid used to create a numerically stable log-sigmoid output
        self.logsigmoid = nn.LogSigmoid()

        # Channel mixing matrix (learnable)
        self.mix = nn.Parameter(torch.randn(channels, channels) * 0.02)

        # Per-channel residual scaling
        self.channel_scale = nn.Parameter(torch.ones(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, L)

        Returns:
            torch.Tensor: Output tensor of shape (B, C, L)
        """
        # Expecting shape (B, C, L)
        B, C, L = x.shape

        # 1) Instance normalization
        norm = self.inst_norm(x)  # (B, C, L)

        # 2) Temporal global context: average over length
        context = norm.mean(dim=2)  # (B, C)

        # 3) Gating MLP
        s = self.fc1(context)       # (B, hidden)
        s = self.relu6(s)           # (B, hidden)
        s = self.fc2(s)             # (B, C)

        # 4) LogSigmoid -> exponentiate to recover sigmoid(context)
        log_sig = self.logsigmoid(s)    # (B, C) : log(sigmoid(s))
        gate = torch.exp(log_sig)       # (B, C) : sigmoid(s)

        # 5) Gate normalized features (broadcast over length)
        gate_exp = gate.unsqueeze(2)    # (B, C, 1)
        gated = norm * gate_exp         # (B, C, L)

        # 6) Channel mixing via einsum: mix has shape (C, C), norm shape (B, C, L)
        mixed = torch.einsum('ij,bjl->bil', self.mix, gated)  # (B, C, L)

        # 7) Combine with per-channel residual scaling and final activation
        scale = self.channel_scale.view(1, C, 1)  # (1, C, 1) broadcastable
        out = mixed + gated * scale               # (B, C, L)
        out = self.relu6(out)

        return out

# Configuration variables
BATCH_SIZE = 16
CHANNELS = 128
LENGTH = 256
REDUCTION = 4

def get_inputs():
    """
    Returns a list of input tensors to run the model forward with.
    """
    x = torch.randn(BATCH_SIZE, CHANNELS, LENGTH)
    return [x]

def get_init_inputs():
    """
    Returns the arguments required to initialize the Model:
    [channels, reduction]
    """
    return [CHANNELS, REDUCTION]