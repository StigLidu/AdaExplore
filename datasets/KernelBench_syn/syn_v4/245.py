import torch
import torch.nn as nn

# Configuration: batch size and layer dimensions
BATCH_SIZE = 12
INPUT_DIM = 2048
HIDDEN_DIM1 = 4096
HIDDEN_DIM2 = 2048
OUTPUT_DIM = 1024

class Model(nn.Module):
    """
    A moderately complex feed-forward module that combines multiple linear projections,
    elementwise nonlinearities (Tanh, SiLU, CELU) and a gating + residual pattern.

    Computation pattern:
      1. Linear projection from input to a large hidden space (W1, b1)
      2. Tanh nonlinearity
      3. Projection to a second hidden space (W2, b2)
      4. Gating of that projection using a SiLU-activated projection of the original input (Wg, bg)
      5. Residual/skip projection from input to the same second hidden space (Wskip, bskip)
      6. Elementwise combination (gated * projected + skip), followed by CELU nonlinearity
      7. Final linear projection to output dimension (Wout, bout)

    This pattern mixes elementwise activations and large matrix multiplications,
    and is functionally distinct from simple matmuls or single activations.
    """
    def __init__(self):
        super(Model, self).__init__()
        # Activation modules
        self.tanh = nn.Tanh()
        self.silu = nn.SiLU()
        self.celu = nn.CELU()

        # Learnable weight matrices and biases (as Parameters)
        # Input -> Hidden1
        self.W1 = nn.Parameter(torch.randn(INPUT_DIM, HIDDEN_DIM1) * (1.0 / (INPUT_DIM ** 0.5)))
        self.b1 = nn.Parameter(torch.zeros(HIDDEN_DIM1))

        # Hidden1 -> Hidden2
        self.W2 = nn.Parameter(torch.randn(HIDDEN_DIM1, HIDDEN_DIM2) * (1.0 / (HIDDEN_DIM1 ** 0.5)))
        self.b2 = nn.Parameter(torch.zeros(HIDDEN_DIM2))

        # Gating projection: Input -> Hidden2 (to create an input-dependent gate)
        self.Wg = nn.Parameter(torch.randn(INPUT_DIM, HIDDEN_DIM2) * (1.0 / (INPUT_DIM ** 0.5)))
        self.bg = nn.Parameter(torch.zeros(HIDDEN_DIM2))

        # Skip/residual projection: Input -> Hidden2
        self.Wskip = nn.Parameter(torch.randn(INPUT_DIM, HIDDEN_DIM2) * (1.0 / (INPUT_DIM ** 0.5)))
        self.bskip = nn.Parameter(torch.zeros(HIDDEN_DIM2))

        # Final output projection: Hidden2 -> Output
        self.Wout = nn.Parameter(torch.randn(HIDDEN_DIM2, OUTPUT_DIM) * (1.0 / (HIDDEN_DIM2 ** 0.5)))
        self.bout = nn.Parameter(torch.zeros(OUTPUT_DIM))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (BATCH, INPUT_DIM)

        Returns:
            Tensor of shape (BATCH, OUTPUT_DIM)
        """
        # Project to hidden1 and apply nonlinearity
        h1 = x @ self.W1 + self.b1  # (BATCH, HIDDEN_DIM1)
        a1 = self.tanh(h1)          # (BATCH, HIDDEN_DIM1)

        # Project to hidden2
        h2 = a1 @ self.W2 + self.b2  # (BATCH, HIDDEN_DIM2)

        # Compute gating from the original input and apply SiLU
        gate_lin = x @ self.Wg + self.bg  # (BATCH, HIDDEN_DIM2)
        gate = self.silu(gate_lin)        # (BATCH, HIDDEN_DIM2)

        # Elementwise gating
        gated = h2 * gate  # (BATCH, HIDDEN_DIM2)

        # Residual/skip connection from input projected into hidden2
        skip = x @ self.Wskip + self.bskip  # (BATCH, HIDDEN_DIM2)

        # Combine, apply CELU, then final projection
        combined = gated + skip              # (BATCH, HIDDEN_DIM2)
        activated = self.celu(combined)      # (BATCH, HIDDEN_DIM2)
        out = activated @ self.Wout + self.bout  # (BATCH, OUTPUT_DIM)

        return out

def get_inputs():
    """
    Returns input tensors for the model.
    """
    x = torch.randn(BATCH_SIZE, INPUT_DIM)
    return [x]

def get_init_inputs():
    """
    Returns initialization parameters if any are required externally.
    For this model, parameters are initialized inside the module, so nothing is required.
    """
    return []