import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Complex model that combines Circular padding, a lazily-initialized ConvTranspose2d
    (for learned upsampling), and a 1D average pooling applied over the flattened spatial
    dimension to create a texture-aware gating mechanism.

    The forward pass:
      1. Apply circular padding to preserve edge continuity.
      2. Apply a LazyConvTranspose2d to upsample and mix channels.
      3. Apply ReLU non-linearity.
      4. Flatten spatial dims into a sequence and apply AvgPool1d (temporal smoothing).
      5. Reshape pooled sequence back to spatial shape and form a gating map.
      6. Combine the ConvTranspose output with the gating map (elementwise multiply),
         add a small learned residual scaling, and reduce channels to a single map
         via channel mean.

    This architecture is functionally distinct from the examples and uses:
      - nn.CircularPad2d
      - nn.LazyConvTranspose2d
      - nn.AvgPool1d
    """
    def __init__(self,
                 out_channels: int,
                 kernel_size: int = 4,
                 stride: int = 2,
                 padding: int = 1,
                 avgpool_kernel: int = 3):
        """
        Initializes the model components.

        Args:
            out_channels (int): Number of output channels produced by ConvTranspose2d.
            kernel_size (int): Kernel size for the ConvTranspose2d.
            stride (int): Stride (upsampling factor) for the ConvTranspose2d.
            padding (int): Padding for the ConvTranspose2d.
            avgpool_kernel (int): Kernel size for the AvgPool1d smoothing.
        """
        super(Model, self).__init__()
        # Circular padding to preserve wrap-around continuity on spatial borders
        self.pad = nn.CircularPad2d(1)

        # Lazy transposed convolution: in_channels will be inferred on first call
        self.deconv = nn.LazyConvTranspose2d(out_channels=out_channels,
                                             kernel_size=kernel_size,
                                             stride=stride,
                                             padding=padding)

        # 1D average pooling applied to flattened spatial dimension (temporal smoothing)
        # use padding to keep length unchanged so we can reshape back
        self.avgpool = nn.AvgPool1d(kernel_size=avgpool_kernel, stride=1, padding=avgpool_kernel // 2)

        # Small learnable scaling applied to the residual branch
        self.rescale = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward computation.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C_in, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (B, 1, H_up, W_up) where H_up and W_up
                          depend on the ConvTranspose2d stride/kernel.
        """
        # 1) Circular pad to keep edge continuity
        x_padded = self.pad(x)

        # 2) Learned upsampling / channel mixing
        z = self.deconv(x_padded)  # (B, C_out, H_up, W_up)

        # 3) Non-linearity
        z = torch.relu(z)

        B, C, H_up, W_up = z.shape

        # 4) Flatten spatial dims into a sequence and apply 1D average pooling
        seq = z.view(B, C, -1)  # (B, C, L) where L = H_up * W_up
        seq_pooled = self.avgpool(seq)  # (B, C, L), smoothed along the spatial sequence

        # 5) Reshape pooled sequence back to spatial grid to create a gating map
        gated = seq_pooled.view(B, C, H_up, W_up)

        # 6) Combine: elementwise gating, add scaled residual, then reduce channels
        combined = z * torch.sigmoid(gated) + self.rescale * z

        # Reduce channels by mean to produce a single-channel map (B, 1, H_up, W_up)
        out = combined.mean(dim=1, keepdim=True)

        return out

# Module-level configuration variables
batch_size = 8
in_channels = 3
out_channels = 16
height = 32
width = 32
kernel_size = 4
stride = 2
padding = 1
avgpool_kernel = 3

def get_inputs():
    """
    Returns a list containing a single input tensor consistent with the module-level
    configuration variables.
    """
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]

def get_init_inputs():
    """
    Returns initialization parameters for the Model constructor in the same order.
    """
    return [out_channels, kernel_size, stride, padding, avgpool_kernel]