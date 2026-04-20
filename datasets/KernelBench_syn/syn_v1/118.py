import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Complex model combining 3D instance normalization, depth-to-plane reduction,
    reflection padding, channel-wise FeatureAlphaDropout and a final projection.

    Computation steps:
        1. Apply LazyInstanceNorm3d to the input volume X (N, C, D, H, W).
           This will lazily initialize num_features the first time it's called.
        2. Reduce the normalized volume along the depth dimension (D) by mean to get a 4D tensor (N, C, H, W).
        3. Apply ReflectionPad2d to pad spatial dimensions.
        4. Spatially global-average the padded tensor to produce per-channel descriptors (N, C).
        5. Apply FeatureAlphaDropout across channels to randomly mask channels.
        6. Project the channel descriptors through a provided weight matrix W (C, M) to get (N, M).
    """
    def __init__(self, dropout_p: float = 0.1, pad: int = 1):
        super(Model, self).__init__()
        # LazyInstanceNorm3d will infer num_features from the input on first forward pass
        self.inst_norm3d = nn.LazyInstanceNorm3d()
        # Reflection padding for 4D tensors (N, C, H, W)
        self.pad2d = nn.ReflectionPad2d(pad)
        # FeatureAlphaDropout operates on channels; set dropout probability
        self.feat_dropout = nn.FeatureAlphaDropout(p=dropout_p)

    def forward(self, X: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            X (torch.Tensor): 5D input tensor with shape (N, C, D, H, W).
            W (torch.Tensor): 2D projection matrix with shape (C, M).

        Returns:
            torch.Tensor: Output tensor with shape (N, M).
        """
        # 1) Normalize across channel/instance for the 3D volume
        Xn = self.inst_norm3d(X)  # shape: (N, C, D, H, W)

        # 2) Collapse the depth dimension into a 4D spatial feature map by averaging over D
        Xm = Xn.mean(dim=2)  # shape: (N, C, H, W)

        # 3) Reflection-pad the spatial dimensions to increase receptive context
        Xp = self.pad2d(Xm)  # shape: (N, C, H+2*pad, W+2*pad)

        # 4) Global spatial pooling (mean over H and W) to get per-channel descriptors
        # Use mean over the last two dimensions
        Xdesc = Xp.mean(dim=(-2, -1))  # shape: (N, C)

        # 5) Apply channel-wise FeatureAlphaDropout
        Xd = self.feat_dropout(Xdesc)  # shape: (N, C)

        # 6) Final linear projection using provided weight matrix W (C, M)
        # Ensure W has shape (C, M); perform batch matmul
        out = torch.matmul(Xd, W)  # shape: (N, M)

        return out

# Configuration / tensor sizes
N = 4      # batch size
C = 16     # number of channels/features
D = 6      # depth slices of the 3D volume
H = 32     # height
W_sp = 32  # width
M = 128    # projection output dimension
PAD = 1    # padding for ReflectionPad2d
DROPOUT_P = 0.15  # dropout probability for FeatureAlphaDropout

def get_inputs():
    """
    Returns a list with:
      - X: a random 5D tensor (N, C, D, H, W)
      - W: a random projection matrix (C, M)
    """
    X = torch.randn(N, C, D, H, W_sp)
    W = torch.randn(C, M)
    return [X, W]

def get_init_inputs():
    """
    No special initialization inputs required for lazy modules here.
    """
    return []