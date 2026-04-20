import torch
import torch.nn as nn
import math

class Model(nn.Module):
    """
    Complex model that combines BatchNorm2d, MaxPool2d and multiple Linear layers
    to perform a per-spatial-position MLP on pooled features followed by a classifier.
    
    Computation performed in forward:
      1. BatchNorm2d -> ReLU
      2. MaxPool2d
      3. For each spatial location, apply an MLP (Linear -> ReLU -> Linear) across channels:
         - treat the input as (B, N_spatial, C) and apply Linear(C -> hidden) then Linear(hidden -> C)
      4. Optionally add a residual connection from the pooled features to the MLP output
      5. Flatten and apply a final classifier Linear to produce output of size out_dim
    """
    def __init__(
        self,
        num_features: int,
        pool_kernel: int,
        pool_stride: int,
        hidden_dim: int,
        out_dim: int,
        input_height: int,
        input_width: int,
        use_residual: bool = True,
        pool_padding: int = 0,
        pool_dilation: int = 1
    ):
        """
        Args:
            num_features: Number of channels in the input (C).
            pool_kernel: Kernel size for MaxPool2d.
            pool_stride: Stride for MaxPool2d.
            hidden_dim: Hidden dimension for the per-location MLP.
            out_dim: Output dimension of the final classifier.
            input_height: Height of the input tensor (H).
            input_width: Width of the input tensor (W).
            use_residual: Whether to add residual connection from pooled features.
            pool_padding: Padding for MaxPool2d (default 0).
            pool_dilation: Dilation for MaxPool2d (default 1).
        """
        super(Model, self).__init__()
        self.num_features = num_features
        self.pool_kernel = pool_kernel
        self.pool_stride = pool_stride
        self.pool_padding = pool_padding
        self.pool_dilation = pool_dilation
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.use_residual = use_residual

        # Normalization and pooling layers
        self.bn = nn.BatchNorm2d(num_features)
        self.pool = nn.MaxPool2d(kernel_size=pool_kernel, stride=pool_stride,
                                 padding=pool_padding, dilation=pool_dilation)

        # Per-spatial-position MLP implemented via Linear layers applied to the channel dimension.
        # These linears operate on shape (..., C) where C = num_features.
        self.fc1 = nn.Linear(num_features, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_features)

        # compute the spatial dimensions after pooling to set up the final classifier
        # Formula for output size of pooling layer:
        # out = floor((in + 2*padding - dilation*(kernel-1) - 1)/stride + 1)
        def pooled_dim(in_size):
            return math.floor((in_size + 2 * pool_padding - pool_dilation * (pool_kernel - 1) - 1) / pool_stride + 1)

        self.pooled_H = pooled_dim(input_height)
        self.pooled_W = pooled_dim(input_width)

        # Final classifier that consumes the flattened pooled + processed features
        classifier_in_features = num_features * self.pooled_H * self.pooled_W
        self.classifier = nn.Linear(classifier_in_features, out_dim)

        # Small initialization for stability
        nn.init.kaiming_uniform_(self.fc1.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.fc2.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.classifier.weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (B, C, H, W)

        Returns:
            Tensor of shape (B, out_dim)
        """
        # Step 1: BatchNorm + activation
        x = self.bn(x)
        x = torch.relu(x)

        # Step 2: Max pooling
        x_pooled = self.pool(x)  # shape (B, C, Hp, Wp)

        B, C, Hp, Wp = x_pooled.shape  # expected C == self.num_features
        # Step 3: Prepare for per-spatial MLP: (B, Hp*Wp, C)
        x_spatial = x_pooled.flatten(2).transpose(1, 2)  # (B, N, C) where N = Hp*Wp

        # Per-spatial-position MLP: Linear(C -> hidden) -> ReLU -> Linear(hidden -> C)
        hidden = self.fc1(x_spatial)  # (B, N, hidden_dim)
        hidden = torch.relu(hidden)
        processed = self.fc2(hidden)  # (B, N, C)

        # Reshape back to (B, C, Hp, Wp)
        processed = processed.transpose(1, 2).view(B, C, Hp, Wp)

        # Step 4: Optional residual connection
        if self.use_residual:
            out = x_pooled + processed
        else:
            out = processed

        # Step 5: Flatten spatial dims and classify
        out_flat = out.view(B, -1)  # (B, C*Hp*Wp)
        logits = self.classifier(out_flat)  # (B, out_dim)

        return logits

# Module-level configuration variables
batch_size = 8
channels = 64
height = 32
width = 32
pool_kernel = 3
pool_stride = 2
hidden_dim = 128
out_dim = 10
use_residual = True
pool_padding = 0
pool_dilation = 1

def get_inputs():
    """
    Returns a list containing a single input tensor with shape (batch_size, channels, height, width).
    """
    x = torch.randn(batch_size, channels, height, width)
    return [x]

def get_init_inputs():
    """
    Returns the list of initialization parameters that should be passed to Model(...) to
    create an instance matching the configuration above.
    """
    return [channels, pool_kernel, pool_stride, hidden_dim, out_dim, height, width, use_residual, pool_padding, pool_dilation]