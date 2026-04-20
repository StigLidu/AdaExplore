import torch
import torch.nn as nn

class Model(nn.Module):
    """
    A composite model that demonstrates interaction between 3D convolutions (temporal/volumetric data)
    and 2D image features. The model:
      1. Applies a lazy 3D convolution to a (B, C_in, D, H, W) volume.
      2. Collapses the depth dimension with an average to produce a 2D feature map.
      3. Applies 2D average pooling to reduce spatial resolution.
      4. Upsamples the pooled features back with bilinear interpolation.
      5. Computes an attention map from a separate 2D image and gates the upsampled features.
      6. Adds a residual connection from the collapsed 2D features and performs L2 normalization across channels.
    """
    def __init__(self,
                 conv_out_channels: int = 32,
                 conv_kernel: int = 3,
                 pool_kernel: int = 2,
                 upsample_scale: int = 2,
                 eps: float = 1e-6):
        super(Model, self).__init__()
        # LazyConv3d will infer in_channels from the input at first forward pass.
        self.conv3d = nn.LazyConv3d(out_channels=conv_out_channels,
                                    kernel_size=conv_kernel,
                                    padding=conv_kernel // 2)  # preserve spatial dims
        self.pool2d = nn.AvgPool2d(kernel_size=pool_kernel, stride=pool_kernel)
        # Bilinear upsampling for 2D feature maps
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=upsample_scale)
        self.eps = eps

    def forward(self, volume: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            volume: A 5D tensor of shape (B, C_in, D, H, W)
            image: A 4D tensor of shape (B, C_img, H, W)

        Returns:
            A 4D tensor of shape (B, C_out, H, W) which is the normalized fused feature map.
        """
        # 1) 3D convolution (lazy in_channels will be set here)
        feat3d = self.conv3d(volume)                     # (B, C_out, D, H, W)

        # 2) Collapse depth to create a 2D feature map by averaging over D
        feat2d = feat3d.mean(dim=2)                      # (B, C_out, H, W)

        # 3) Spatial average pooling to reduce resolution
        pooled = self.pool2d(feat2d)                     # (B, C_out, H/2, W/2)

        # 4) Upsample back to original spatial resolution
        upsampled = self.upsample(pooled)                # (B, C_out, H, W)

        # 5) Compute a simple attention map from the 2D image and gate features
        #    Use channel-average of the image to get a single-channel attention map
        att = torch.sigmoid(image.mean(dim=1, keepdim=True))  # (B, 1, H, W)
        gated = upsampled * att                           # broadcast over channels (B, C_out, H, W)

        # 6) Residual connection: add the original collapsed 2D features
        fused = gated + feat2d                            # (B, C_out, H, W)

        # 7) L2-normalize across the channel dimension for numerical stability
        norm = torch.norm(fused, p=2, dim=1, keepdim=True).clamp_min(self.eps)
        out = fused / norm                                # (B, C_out, H, W)

        return out

# Configuration variables for input shapes
BATCH_SIZE = 4
IN_CHANNELS_3D = 3   # will be inferred by LazyConv3d, but useful for get_inputs
DEPTH = 8
HEIGHT = 64
WIDTH = 64
IMAGE_CHANNELS = 3

def get_inputs():
    """
    Generates a random volumetric input and a corresponding 2D image input.

    Returns:
        [volume, image] where:
          - volume has shape (BATCH_SIZE, IN_CHANNELS_3D, DEPTH, HEIGHT, WIDTH)
          - image has shape (BATCH_SIZE, IMAGE_CHANNELS, HEIGHT, WIDTH)
    """
    volume = torch.randn(BATCH_SIZE, IN_CHANNELS_3D, DEPTH, HEIGHT, WIDTH)
    image = torch.randn(BATCH_SIZE, IMAGE_CHANNELS, HEIGHT, WIDTH)
    return [volume, image]

def get_init_inputs():
    """
    No special initialization parameters are required; LazyConv3d will infer in_channels lazily.
    """
    return []