import torch
import torch.nn as nn
from typing import Tuple

class Model(nn.Module):
    """
    Complex fusion model that:
    - Pads a 4D image tensor by temporarily introducing a depth dimension and using ZeroPad3d.
    - Merges the padded depth into the channel dimension.
    - Applies a ConvTranspose2d (deconvolution) to upsample the feature map.
    - Globally pools the upsampled map to produce an image feature vector.
    - Fuses the image feature vector with an auxiliary vector via nn.Bilinear.
    - Produces a gating vector from the bilinear fusion that modulates the upsampled feature map.
    - Returns a compact pooled representation of the gated feature map.
    """
    def __init__(
        self,
        in_channels: int,
        conv_out_channels: int,
        vec_dim: int,
        bilinear_out: int,
        pad: Tuple[int, int, int, int, int, int],
        kernel_size: int = 4,
        stride: int = 2,
        padding: int = 1
    ):
        """
        Initializes modules and computes derived channel sizes.

        Args:
            in_channels (int): Number of input image channels.
            conv_out_channels (int): Number of output channels for the ConvTranspose2d.
            vec_dim (int): Dimensionality of the auxiliary vector to fuse.
            bilinear_out (int): Output features from the Bilinear fusion.
            pad (tuple): 6-element padding for ZeroPad3d (left,right,top,bottom,front,back).
            kernel_size (int): Kernel size for ConvTranspose2d.
            stride (int): Stride for ConvTranspose2d.
            padding (int): Padding for ConvTranspose2d.
        """
        super(Model, self).__init__()
        assert len(pad) == 6, "pad must be a 6-tuple for ZeroPad3d"
        self.in_channels = in_channels
        self.conv_out_channels = conv_out_channels
        self.vec_dim = vec_dim
        self.bilinear_out = bilinear_out
        self.pad_tuple = pad

        # After adding an explicit depth dim of 1, ZeroPad3d will add front+back
        self.depth_after = 1 + pad[4] + pad[5]
        conv_in_channels = in_channels * self.depth_after

        # Modules
        self.pad3d = nn.ZeroPad3d(self.pad_tuple)  # expects (L,R,T,B,F,B)
        self.deconv = nn.ConvTranspose2d(
            in_channels=conv_in_channels,
            out_channels=conv_out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )
        # Fusion between pooled image features and auxiliary vector
        self.bilinear = nn.Bilinear(conv_out_channels, vec_dim, bilinear_out)
        # Project fusion to a gating vector matching conv_out_channels
        self.fc_gate = nn.Linear(bilinear_out, conv_out_channels)
        # Small projection for final compact representation (optional)
        self.fc_final = nn.Linear(conv_out_channels, conv_out_channels)

    def forward(self, img: torch.Tensor, aux: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            img (torch.Tensor): Input image tensor of shape (N, C, H, W).
            aux (torch.Tensor): Auxiliary vector tensor of shape (N, vec_dim).

        Returns:
            torch.Tensor: Compact pooled representation of the gated upsampled features,
                          shape (N, conv_out_channels).
        """
        # 1) Introduce a singleton depth dimension to allow 3D padding
        #    img: (N, C, H, W) -> (N, C, 1, H, W)
        x = img.unsqueeze(2)

        # 2) ZeroPad3d expands (D,H,W) according to pad_tuple
        #    after padding: (N, C, D', H', W')
        x = self.pad3d(x)

        # 3) Merge depth into channel dimension: (N, C*D', H', W')
        x = x.flatten(1, 2)

        # 4) Upsample / transform with ConvTranspose2d and activation
        x = self.deconv(x)
        x = torch.relu(x)

        # 5) Global average pooling over spatial dims to obtain image feature vector
        img_feat = x.mean(dim=(2, 3))  # shape (N, conv_out_channels)

        # 6) Bilinear fusion between image features and auxiliary vector
        fused = self.bilinear(img_feat, aux)  # shape (N, bilinear_out)

        # 7) Produce gating vector and apply to upsampled feature map
        gate = torch.sigmoid(self.fc_gate(fused))  # (N, conv_out_channels)
        gate = gate.unsqueeze(-1).unsqueeze(-1)  # (N, conv_out_channels, 1, 1)
        gated_map = x * gate  # broadcast multiply (N, conv_out_channels, H2, W2)

        # 8) Final compact representation: global pool + small projection + residual
        pooled = gated_map.mean(dim=(2, 3))  # (N, conv_out_channels)
        out = torch.relu(self.fc_final(pooled) + pooled)  # residual projection

        return out


# Configuration / default sizes
batch_size = 8
in_channels = 3
conv_out_channels = 32
H = 16
W = 16
vec_dim = 64
bilinear_out = 128
# ZeroPad3d expects (left, right, top, bottom, front, back)
pad = (0, 0, 1, 1, 1, 1)
kernel_size = 4
stride = 2
padding = 1

def get_inputs():
    """
    Returns example inputs:
    - img: batch of images with shape (batch_size, in_channels, H, W)
    - aux: auxiliary vector with shape (batch_size, vec_dim)
    """
    img = torch.randn(batch_size, in_channels, H, W)
    aux = torch.randn(batch_size, vec_dim)
    return [img, aux]

def get_init_inputs():
    """
    Returns initialization parameters for the Model constructor in the expected order.
    """
    return [in_channels, conv_out_channels, vec_dim, bilinear_out, pad, kernel_size, stride, padding]