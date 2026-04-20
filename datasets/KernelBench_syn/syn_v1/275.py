import torch
import torch.nn as nn

class Model(nn.Module):
    """
    A moderately complex feed-forward block that demonstrates:
    - a dense projection (fc1)
    - lazy batch normalization (LazyBatchNorm1d) which initializes on first forward
    - a learnable piecewise linear activation (PReLU)
    - a hard thresholding operation (Threshold)
    - a residual connection with a projection when input/output dims differ
    - a final output projection (fc_out)

    The forward pass sequence is:
        x -> fc1(x) -> bn -> prelu -> threshold -> ( + skip ) -> fc_out
    """
    def __init__(self, input_dim: int, hidden_dim: int, threshold_val: float = 0.0):
        """
        Args:
            input_dim (int): Dimensionality of the input features.
            hidden_dim (int): Dimensionality of the hidden layer.
            threshold_val (float): Threshold value for nn.Threshold; values below this
                                   will be replaced with 0.0.
        """
        super(Model, self).__init__()
        # First linear projection
        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=True)
        # Lazy BatchNorm will infer num_features from the first forward call
        self.bn = nn.LazyBatchNorm1d()
        # Per-channel learnable PReLU
        self.prelu = nn.PReLU(num_parameters=hidden_dim)
        # Threshold layer: values <= threshold_val replaced with 0.0
        self.thresh = nn.Threshold(threshold_val, 0.0)
        # If dimensions differ, a projection for the residual connection
        if input_dim != hidden_dim:
            self.skip_proj = nn.Linear(input_dim, hidden_dim, bias=False)
        else:
            self.skip_proj = nn.Identity()
        # Final output projection
        self.fc_out = nn.Linear(hidden_dim, input_dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim)

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, input_dim)
        """
        # Project to hidden dimension
        h = self.fc1(x)                  # Linear projection
        h = self.bn(h)                   # LazyBatchNorm1d (initialized here on first call)
        h = self.prelu(h)                # Learnable activation
        h = self.thresh(h)               # Hard thresholding
        # Residual connection (with projection if needed)
        skip = self.skip_proj(x)
        h = h + skip                     # Elementwise residual add
        out = self.fc_out(h)             # Final projection back to input_dim
        return out

# Configuration (module-level)
batch_size = 32
input_dim = 8192
hidden_dim = 16384
threshold_val = 1e-3

def get_inputs():
    """
    Returns a list containing a single input tensor for the model.

    Shape:
        (batch_size, input_dim)
    """
    x = torch.randn(batch_size, input_dim)
    return [x]

def get_init_inputs():
    """
    Returns the initialization parameters for the Model constructor:
        [input_dim, hidden_dim, threshold_val]
    """
    return [input_dim, hidden_dim, threshold_val]