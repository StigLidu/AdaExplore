import torch
import torch.nn as nn

# Configuration
batch_size = 8

# 2D image tensor dimensions
img_c = 16
img_h = 64
img_w = 64

# 3D volume tensor dimensions (D, H, W)
vol_c = 12
vol_d = 8
vol_h = 16
vol_w = 16

# Feature projection dimensions
proj_dim = 128
out_dim = 64


class Model(nn.Module):
    """
    A composite model that fuses 2D image features and 3D volumetric features.
    - Image branch: Conv2d -> BatchNorm2d -> ReLU -> FeatureAlphaDropout -> AdaptiveAvgPool2d -> Linear
    - Volume branch: LazyInstanceNorm3d -> ReLU -> global mean pooling -> Linear
    - Fusion: elementwise gating (multiply), residual add, projection and final classification head

    The model demonstrates usage of nn.BatchNorm2d, nn.FeatureAlphaDropout, and nn.LazyInstanceNorm3d.
    """
    def __init__(self):
        super(Model, self).__init__()

        # Image branch
        self.conv1 = nn.Conv2d(in_channels=img_c, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.dropout_feat = nn.FeatureAlphaDropout(p=0.1)
        self.pool2d = nn.AdaptiveAvgPool2d((4, 4))
        self.fc_img = nn.Linear(32 * 4 * 4, proj_dim)

        # Volume branch: lazy instance norm to demonstrate lazy initialization behavior
        # The module will infer num_features on the first forward pass.
        self.inst_norm3d = nn.LazyInstanceNorm3d()
        self.fc_vol = nn.Linear(vol_c, proj_dim)

        # Fusion and classification head
        self.fusion_proj = nn.Linear(proj_dim, proj_dim)
        self.final_fc = nn.Linear(proj_dim, out_dim)

    def forward(self, image: torch.Tensor, volume: torch.Tensor) -> torch.Tensor:
        """
        Forward pass combining image and volume modalities.

        Args:
            image: Tensor of shape (N, img_c, img_h, img_w)
            volume: Tensor of shape (N, vol_c, vol_d, vol_h, vol_w)

        Returns:
            Tensor of shape (N, out_dim) - fused representation after projection.
        """
        # Image branch
        x = self.conv1(image)              # (N, 32, H, W)
        x = self.bn1(x)                    # BatchNorm2d over channels
        x = self.relu(x)
        x = self.dropout_feat(x)           # FeatureAlphaDropout masks whole channels
        x = self.pool2d(x)                 # (N, 32, 4, 4)
        x = x.view(x.size(0), -1)          # Flatten -> (N, 32*4*4)
        img_proj = self.fc_img(x)          # (N, proj_dim)

        # Volume branch
        # LazyInstanceNorm3d will initialize num_features based on volume.shape[1] when called first time
        v = self.inst_norm3d(volume)       # (N, vol_c, D, H, W)
        v = self.relu(v)
        # Global average pooling across spatial dims (D, H, W) -> (N, vol_c)
        v = v.mean(dim=(2, 3, 4))
        vol_proj = self.fc_vol(v)          # (N, proj_dim)

        # Fusion: gate image features by volumetric features and add residual
        gated = img_proj * torch.sigmoid(vol_proj)   # elementwise gating
        fused = gated + img_proj                      # residual connection

        # Projection and final output
        fused = self.relu(self.fusion_proj(fused))
        out = self.final_fc(fused)                    # (N, out_dim)
        return out


def get_inputs():
    """
    Generate synthetic inputs for the model:
    - image: (batch_size, img_c, img_h, img_w)
    - volume: (batch_size, vol_c, vol_d, vol_h, vol_w)
    """
    image = torch.randn(batch_size, img_c, img_h, img_w)
    volume = torch.randn(batch_size, vol_c, vol_d, vol_h, vol_w)
    return [image, volume]


def get_init_inputs():
    """
    No special initialization inputs required for this module.
    """
    return []