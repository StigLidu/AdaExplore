import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Complex model that composes Hardsigmoid gating, Hardshrink sparsification,
    and FractionalMaxPool2d pooling to produce a compact channel-wise summary.

    Computation steps:
    1. Apply Hardsigmoid to input to produce a gating mask in [0, 1].
    2. Gate the input by element-wise multiplication (input * gate).
    3. Apply Hardshrink to sparsify small values.
    4. Take absolute values to focus on magnitude and feed to FractionalMaxPool2d.
    5. Reduce spatial dimensions by fractional max-pooling.
    6. Produce channel-wise aggregated features by computing spatial max and mean,
       and combine them into a normalized channel descriptor.
    """
    def __init__(self, kernel_size, output_ratio=None, return_indices: bool = False, shrink_lambda: float = 0.5):
        """
        Initializes the composed layers.

        Args:
            kernel_size (int or tuple): Kernel size passed to FractionalMaxPool2d.
            output_ratio (tuple, optional): Output ratio for FractionalMaxPool2d.
            return_indices (bool): Whether FractionalMaxPool2d returns indices.
            shrink_lambda (float): Lambda parameter for Hardshrink (threshold).
        """
        super(Model, self).__init__()
        # Non-linear gating that squeezes inputs to [0,1]
        self.hardsigmoid = nn.Hardsigmoid()
        # Sparse activation that zeros small entries (using lambd parameter name)
        self.hardshrink = nn.Hardshrink(lambd=shrink_lambda)
        # Fractional max pooling to reduce spatial dimensions in a flexible way
        # Accepts kernel_size and optional output_ratio; return_indices can be useful for unpooling
        self.fracpool = nn.FractionalMaxPool2d(kernel_size, output_ratio=output_ratio, return_indices=return_indices)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass combining gating, shrinkage, fractional pooling, and channel aggregation.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Tensor of shape (batch_size, channels, 1) containing
                          a normalized channel-wise descriptor.
        """
        # 1) Gating: compute gate in [0,1] and multiply with input (elementwise)
        gate = self.hardsigmoid(x)
        gated = x * gate

        # 2) Sparsify small activations with Hardshrink
        shrunk = self.hardshrink(gated)

        # 3) Focus on magnitude for pooling by taking absolute values
        mag = torch.abs(shrunk)

        # 4) Fractional max pooling reduces spatial resolution
        pooled = self.fracpool(mag)
        # FractionalMaxPool2d may return a tuple (output, indices) if return_indices=True
        if isinstance(pooled, tuple):
            pooled = pooled[0]

        # 5) Channel-wise spatial aggregation:
        #    - compute spatial max and mean for each channel
        B, C, H, W = pooled.size()
        flattened = pooled.view(B, C, -1)  # shape (B, C, H*W)
        spatial_max = flattened.amax(dim=2)   # (B, C)
        spatial_mean = flattened.mean(dim=2)  # (B, C)

        # 6) Combine into a normalized descriptor: max / (1 + mean)
        descriptor = spatial_max / (1.0 + spatial_mean)

        # 7) Return with an extra spatial dim for consistency (B, C, 1)
        return descriptor.unsqueeze(-1)


# Configuration variables
batch_size = 8
channels = 16
height = 32
width = 48

# Parameters for FractionalMaxPool2d and Hardshrink
kernel_size = (2, 3)               # spatial kernel for fractional pooling
output_ratio = (0.6, 0.5)          # target ratio (output / input) for H and W respectively
return_indices = False
shrink_lambda = 0.4

def get_inputs():
    """
    Returns example input tensors for the model.
    """
    x = torch.randn(batch_size, channels, height, width)
    return [x]

def get_init_inputs():
    """
    Returns the initialization parameters for the Model constructor, in order.
    """
    return [kernel_size, output_ratio, return_indices, shrink_lambda]