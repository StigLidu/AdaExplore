import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    A moderately complex module that mixes linear transforms with three different
    non-linearities: Hardshrink, Threshold and Softplus. The design demonstrates
    a gated residual-like pattern plus a small global context projection that
    modulates per-sample features.

    Computation pattern (high level):
      1. Linear projection of input -> hidden
      2. Hardshrink to sparsify small activations
      3. A second linear to produce gates -> sigmoid gating applied elementwise
      4. Threshold to map tiny values to a constant
      5. A global context vector (mean over batch) is projected and passed through Softplus,
         then broadcast and added to hidden features
      6. Final linear projection to outputs plus a skip linear from input to output
    """
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        hardshrink_lambda: float = 0.5,
        threshold: float = 0.0,
        threshold_value: float = 0.0,
        softplus_beta: float = 1.0,
        softplus_threshold: float = 20.0,
    ):
        """
        Initialize the module.

        Args:
            in_features: Dimensionality of the input feature vector.
            hidden_features: Width of the internal hidden representation.
            out_features: Dimensionality of the output feature vector.
            hardshrink_lambda: Lambda parameter for nn.Hardshrink.
            threshold: Threshold parameter for nn.Threshold.
            threshold_value: Value to set below-threshold elements to for nn.Threshold.
            softplus_beta: Beta parameter for nn.Softplus.
            softplus_threshold: Threshold parameter for nn.Softplus.
        """
        super(Model, self).__init__()

        # First projection to hidden space
        self.fc1 = nn.Linear(in_features, hidden_features, bias=True)

        # Sparsifying nonlinearity
        self.hardshrink = nn.Hardshrink(lambd=hardshrink_lambda)

        # A second linear used to produce gates (followed by sigmoid)
        self.gate_fc = nn.Linear(hidden_features, hidden_features, bias=True)

        # Threshold to clamp small activations to a fixed value
        self.threshold = nn.Threshold(threshold, threshold_value)

        # Context projector: compress global (batch) statistics into hidden dim
        self.context_proj = nn.Linear(in_features, hidden_features, bias=True)

        # Softplus to make the projected context smoothly positive and stable
        self.softplus = nn.Softplus(beta=softplus_beta, threshold=softplus_threshold)

        # Output projection and a direct skip path from input -> output
        self.fc_out = nn.Linear(hidden_features, out_features, bias=True)
        self.skip = nn.Linear(in_features, out_features, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Tensor of shape (batch_size, in_features)

        Returns:
            Tensor of shape (batch_size, out_features)
        """
        # 1) Project to hidden
        h = self.fc1(x)  # (B, H)

        # 2) Sparsify small magnitudes
        h = self.hardshrink(h)  # (B, H)

        # 3) Compute gate and apply elementwise
        gate = torch.sigmoid(self.gate_fc(h))  # (B, H)
        h = h * gate  # gated hidden (B, H)

        # 4) Threshold small values to a constant to avoid tiny noisy activations
        h = self.threshold(h)  # (B, H)

        # 5) Compute global context from the input batch, project and softplus it
        #    Broadcast and add to each sample in the batch to provide global information.
        context = torch.mean(x, dim=0, keepdim=True)  # (1, in_features)
        context = self.context_proj(context)  # (1, H)
        context = self.softplus(context)  # (1, H)
        h = h + context  # broadcast-add (B, H)

        # 6) Final projection plus skip connection from input
        out = self.fc_out(h)  # (B, out_features)
        out = out + self.skip(x)  # (B, out_features)

        return out

# Configuration / default sizes for creating inputs
batch_size = 32
in_features = 1024
hidden_features = 2048
out_features = 512

# Nonlinearity hyperparameters
hardshrink_lambda = 0.7
threshold = 0.05
threshold_value = 0.0
softplus_beta = 1.0
softplus_threshold = 20.0

def get_inputs():
    """
    Returns example input tensors for the model.
    """
    x = torch.randn(batch_size, in_features)
    return [x]

def get_init_inputs():
    """
    Returns initialization arguments for Model(...) in the same order as the signature.
    """
    return [
        in_features,
        hidden_features,
        out_features,
        hardshrink_lambda,
        threshold,
        threshold_value,
        softplus_beta,
        softplus_threshold,
    ]