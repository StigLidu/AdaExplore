import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Complex example model that demonstrates a small two-branch projection network
    combining learnable linear projections with PReLU, LeakyReLU and SELU activations.

    Architecture (forward):
      1. Project input x into two different hidden representations via learned matrices W1 and W2.
      2. Apply PReLU to the first projection (learnable per-hidden-unit parameters).
      3. Apply LeakyReLU to the second projection (configured negative slope).
      4. Concatenate the two activated branches along the feature dimension.
      5. Project the concatenated vector back to input dimensionality with W3.
      6. Apply SELU nonlinearity and add a residual connection to the input (after a matching linear projection).
    """
    def __init__(self, input_dim: int, hidden_dim: int, negative_slope: float = 0.01):
        """
        Initialize the model parameters and activations.

        Args:
            input_dim (int): Dimensionality of the input features.
            hidden_dim (int): Size of each branch's hidden projection.
            negative_slope (float): Negative slope for LeakyReLU activation.
        """
        super(Model, self).__init__()
        # Learnable linear projections implemented as Parameters to show manual matmul usage
        self.W1 = nn.Parameter(torch.randn(input_dim, hidden_dim) * (1.0 / (input_dim ** 0.5)))
        self.b1 = nn.Parameter(torch.zeros(hidden_dim))
        self.W2 = nn.Parameter(torch.randn(input_dim, hidden_dim) * (1.0 / (input_dim ** 0.5)))
        self.b2 = nn.Parameter(torch.zeros(hidden_dim))

        # Projection to restore to input_dim after concatenation of branches
        self.W3 = nn.Parameter(torch.randn(hidden_dim * 2, input_dim) * (1.0 / ((hidden_dim * 2) ** 0.5)))
        self.b3 = nn.Parameter(torch.zeros(input_dim))

        # Small linear projection for the residual path to match dimensions
        self.res_W = nn.Parameter(torch.randn(input_dim, input_dim) * (1.0 / (input_dim ** 0.5)))
        self.res_b = nn.Parameter(torch.zeros(input_dim))

        # Activations
        self.prelu = nn.PReLU(num_parameters=hidden_dim)      # learnable per-hidden-unit slope
        self.leaky_relu = nn.LeakyReLU(negative_slope=negative_slope, inplace=False)
        self.selu = nn.SELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim)

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, input_dim)
        """
        # Branch A: linear projection then PReLU
        # x @ W1 -> (batch, hidden_dim)
        a = torch.matmul(x, self.W1) + self.b1
        a = self.prelu(a)

        # Branch B: linear projection then LeakyReLU
        b = torch.matmul(x, self.W2) + self.b2
        b = self.leaky_relu(b)

        # Concatenate along feature axis -> (batch, hidden_dim * 2)
        combined = torch.cat((a, b), dim=1)

        # Project back to input_dim and apply SELU
        out = torch.matmul(combined, self.W3) + self.b3
        out = self.selu(out)

        # Residual path: simple linear projection of input
        residual = torch.matmul(x, self.res_W) + self.res_b

        # Final output: combine with residual and apply one more SELU for stability
        output = self.selu(out + residual)
        return output

# Module-level configuration variables
batch_size = 8
input_dim = 512
hidden_dim = 256
negative_slope = 0.1

def get_inputs():
    """
    Returns a list containing a single input tensor for the model.
    Shape: (batch_size, input_dim)
    """
    x = torch.randn(batch_size, input_dim)
    return [x]

def get_init_inputs():
    """
    Returns initialization arguments for Model.__init__:
      [input_dim, hidden_dim, negative_slope]
    """
    return [input_dim, hidden_dim, negative_slope]