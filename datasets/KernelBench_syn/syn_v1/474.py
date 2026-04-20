import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Complex example combining 2D and 3D processing paths with ReflectionPad2d,
ReplicationPad3d and Hardswish activation. The model processes an image-like
2D tensor and a volume-like 3D tensor in parallel, projects both to a shared
embedding space, fuses them elementwise, and produces a compact output vector.
"""

# Configuration (module-level)
BATCH = 8
C2D = 3         # channels for 2D input (e.g., RGB image)
H2D = 64
W2D = 64

C3D = 2         # channels for 3D input (e.g., small volumetric sensor)
D3D = 8
H3D = 16
W3D = 16

CONV2D_OUT = 16
CONV3D_OUT = 16
EMBED_DIM = 32
OUTPUT_DIM = 10

class Model(nn.Module):
    """
    Model with two parallel branches:
    - 2D branch: ReflectionPad2d -> Conv2d -> Hardswish -> spatial mean
    - 3D branch: ReplicationPad3d -> Conv3d -> Hardswish -> spatiotemporal mean

    The pooled outputs are each projected to an embedding (Linear), fused
    element-wise, passed through another Hardswish and a final Linear layer
    to produce OUTPUT_DIM outputs per batch element.
    """
    def __init__(self):
        super(Model, self).__init__()
        # 2D branch components
        self.pad2d = nn.ReflectionPad2d(3)  # reflect padding of 3 pixels on each side
        self.conv2d = nn.Conv2d(in_channels=C2D, out_channels=CONV2D_OUT,
                                kernel_size=5, stride=1, padding=0, bias=True)
        
        # 3D branch components
        # ReplicationPad3d expects (padL, padR, padT, padB, padF, padB) style tuple
        self.pad3d = nn.ReplicationPad3d((1, 1, 1, 1, 1, 1))
        self.conv3d = nn.Conv3d(in_channels=C3D, out_channels=CONV3D_OUT,
                                kernel_size=3, stride=1, padding=0, bias=True)
        
        # Shared projection heads and fusion
        self.proj2d = nn.Linear(CONV2D_OUT, EMBED_DIM, bias=True)
        self.proj3d = nn.Linear(CONV3D_OUT, EMBED_DIM, bias=True)
        self.act = nn.Hardswish()
        self.fusion_fc = nn.Linear(EMBED_DIM, EMBED_DIM // 2, bias=True)
        self.final_fc = nn.Linear(EMBED_DIM // 2, OUTPUT_DIM, bias=True)

    def forward(self, x2d: torch.Tensor, v3d: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x2d (torch.Tensor): 4D tensor (BATCH, C2D, H2D, W2D)
            v3d (torch.Tensor): 5D tensor (BATCH, C3D, D3D, H3D, W3D)

        Returns:
            torch.Tensor: Output tensor of shape (BATCH, OUTPUT_DIM)
        """
        # 2D branch: pad -> conv -> activation -> spatial global mean
        x = self.pad2d(x2d)                       # (B, C2D, H2D+6, W2D+6)
        x = self.conv2d(x)                        # (B, CONV2D_OUT, H2D+2, W2D+2) depending on kernel
        x = self.act(x)                           # element-wise nonlinearity
        x = x.mean(dim=(2, 3))                    # global spatial pooling -> (B, CONV2D_OUT)
        x = self.proj2d(x)                        # -> (B, EMBED_DIM)
        x = self.act(x)

        # 3D branch: pad -> conv3d -> activation -> spatiotemporal global mean
        v = self.pad3d(v3d)                       # (B, C3D, D3D+2, H3D+2, W3D+2)
        v = self.conv3d(v)                        # (B, CONV3D_OUT, D', H', W')
        v = self.act(v)
        v = v.mean(dim=(2, 3, 4))                 # global pooling over D,H,W -> (B, CONV3D_OUT)
        v = self.proj3d(v)                        # -> (B, EMBED_DIM)
        v = self.act(v)

        # Fuse the two embeddings element-wise (Hadamard), then further process
        fused = x * v                             # (B, EMBED_DIM)
        fused = self.act(self.fusion_fc(fused))   # -> (B, EMBED_DIM//2)
        out = self.final_fc(fused)                # -> (B, OUTPUT_DIM)
        return out

def get_inputs():
    """
    Generates example input tensors for testing the module.

    Returns:
        list: [x2d, v3d] where:
            x2d is shape (BATCH, C2D, H2D, W2D)
            v3d is shape (BATCH, C3D, D3D, H3D, W3D)
    """
    x2d = torch.randn(BATCH, C2D, H2D, W2D)
    v3d = torch.randn(BATCH, C3D, D3D, H3D, W3D)
    return [x2d, v3d]

def get_init_inputs():
    """
    No special initialization parameters required for this module beyond defaults.

    Returns:
        list: empty list
    """
    return []