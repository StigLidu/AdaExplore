import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    A moderately complex CNN-based projection model that demonstrates:
    - Convolutional feature extraction
    - Non-linear Tanhshrink activation
    - Spatial dimensionality reduction with AdaptiveMaxPool2d
    - Gating using LogSigmoid (converted back to sigmoid via exp)
    - Feature projection and final matrix multiplication with an external projection matrix

    The forward pass expects:
        x: input image tensor of shape (batch, in_channels, H, W)
        proj: a projection matrix of shape (proj_dim, out_dim)

    Outputs:
        Tensor of shape (batch, out_dim)
    """
    def __init__(
        self,
        in_channels: int = 3,
        mid_channels: int = 32,
        pool_output: int = 8,
        proj_dim: int = 512,
        out_dim: int = 256,
    ):
        super(Model, self).__init__()
        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.pool_output = pool_output
        self.proj_dim = proj_dim
        self.out_dim = out_dim

        # Convolutional feature extractor (two small conv layers)
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=True)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, bias=True)

        # Non-linearities
        self.tanhshrink = nn.Tanhshrink()         # elementwise Tanhshrink
        self.logsigmoid = nn.LogSigmoid()         # used for computing a gating sigmoid via exp(logsigmoid)

        # Adaptive spatial reduction to a fixed grid (pool_output x pool_output)
        self.adaptive_pool = nn.AdaptiveMaxPool2d((pool_output, pool_output))

        # Fully connected heads: one for gating logits, one for feature projection
        flat_dim = mid_channels * pool_output * pool_output
        self.fc_gate = nn.Linear(flat_dim, proj_dim, bias=True)
        self.fc_feat = nn.Linear(flat_dim, proj_dim, bias=True)

        # Small initialization to keep numerical stability
        nn.init.kaiming_normal_(self.conv1.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv2.weight, nonlinearity='relu')
        nn.init.xavier_uniform_(self.fc_gate.weight)
        nn.init.xavier_uniform_(self.fc_feat.weight)

    def forward(self, x: torch.Tensor, proj: torch.Tensor) -> torch.Tensor:
        """
        Forward computation:
        1. Two conv layers with ReLU then Tanhshrink
        2. Adaptive max-pool to fixed spatial size
        3. Flatten and compute gate logits and features via linear layers
        4. Convert LogSigmoid output back to sigmoid by exponentiation: gate = exp(logsigmoid(gate_logits))
        5. Elementwise gating of features
        6. L2-normalize features and multiply by external projection matrix

        Args:
            x (torch.Tensor): (batch, in_channels, H, W)
            proj (torch.Tensor): (proj_dim, out_dim)

        Returns:
            torch.Tensor: (batch, out_dim)
        """
        # Feature extraction
        x = self.conv1(x)
        x = F.relu(x, inplace=True)
        x = self.conv2(x)
        # Tanhshrink applied elementwise to introduce a different non-linearity
        x = self.tanhshrink(x)

        # Reduce spatial dimensions to a fixed grid (adaptive pooling)
        x = self.adaptive_pool(x)

        # Flatten spatial dimensions
        batch_size = x.shape[0]
        x_flat = x.view(batch_size, -1)  # shape: (batch, mid_channels * pool_output * pool_output)

        # Compute gate logits and features
        gate_logits = self.fc_gate(x_flat)    # (batch, proj_dim)
        feat = self.fc_feat(x_flat)           # (batch, proj_dim)

        # Use LogSigmoid module: logsigmoid(z) = ln(sigmoid(z)), so exp(logsigmoid(z)) == sigmoid(z)
        gate = torch.exp(self.logsigmoid(gate_logits))  # now in (0,1), same shape as feat

        # Gated features
        gated = feat * gate  # elementwise gating

        # Normalize to stabilize the subsequent matmul
        gated = F.normalize(gated, p=2, dim=1)

        # Final projection using external projection matrix
        # proj should be of shape (proj_dim, out_dim)
        out = torch.matmul(gated, proj)  # (batch, out_dim)
        return out

# Configuration / typical sizes
batch_size = 8
in_channels = 3
height = 128
width = 128
mid_channels = 32
pool_output = 8
proj_dim = 512
out_dim = 256

def get_inputs():
    """
    Returns:
        x: random image tensor (batch_size, in_channels, height, width)
        proj: random projection matrix (proj_dim, out_dim)
    """
    x = torch.randn(batch_size, in_channels, height, width)
    proj = torch.randn(proj_dim, out_dim)
    return [x, proj]

def get_init_inputs():
    """
    Returns initialization arguments for the Model constructor in the same order:
    [in_channels, mid_channels, pool_output, proj_dim, out_dim]
    """
    return [in_channels, mid_channels, pool_output, proj_dim, out_dim]