import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Complex example combining ZeroPad2d, SyncBatchNorm, and multiple Linear layers.
    The model accepts an image-like tensor X (B, C, H, W) and an auxiliary vector V (B, E).
    Computation steps:
      1. Zero-pad the spatial dimensions of X.
      2. Apply synchronized batch normalization across channels.
      3. Global average pool spatial dimensions to obtain per-channel descriptors.
      4. Project channel descriptors and auxiliary vector into a shared hidden space.
      5. Fuse the two projections with elementwise and additive interactions, then concatenate.
      6. Final linear projection produces the output vector.
    """
    def __init__(self,
                 in_channels: int,
                 aux_dim: int,
                 hidden_dim: int,
                 out_dim: int,
                 pad: tuple = (1, 1, 1, 1)):
        """
        Args:
            in_channels: Number of input channels C for X.
            aux_dim: Dimensionality of auxiliary input V.
            hidden_dim: Hidden dimensionality used for intermediate projections.
            out_dim: Final output dimensionality.
            pad: ZeroPad2d padding as (left, right, top, bottom).
        """
        super(Model, self).__init__()
        # Padding layer for spatial expansion
        self.pad = nn.ZeroPad2d(pad)

        # Synchronized BatchNorm across channels (suitable for multi-process; works in single-process too)
        self.bn = nn.SyncBatchNorm(num_features=in_channels)

        # Project global-pooled channel descriptors into hidden space
        self.lin_channel = nn.Linear(in_channels, hidden_dim, bias=True)

        # Project auxiliary vector into hidden space
        self.lin_aux = nn.Linear(aux_dim, hidden_dim, bias=True)

        # Combine (elementwise interaction and residual) then map to output
        # combined dim = hidden_dim * 2 (we'll concat two hidden vectors)
        self.lin_out = nn.Linear(hidden_dim * 2, out_dim, bias=True)

        # Small gating projection to modulate the final output (adds another non-linear interaction)
        self.lin_gate = nn.Linear(hidden_dim, out_dim, bias=True)

        # Store config
        self.in_channels = in_channels
        self.aux_dim = aux_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

    def forward(self, X: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            X: Image-like tensor of shape (B, C, H, W).
            V: Auxiliary tensor of shape (B, E).

        Returns:
            Tensor of shape (B, out_dim).
        """
        # 1) Zero-padding spatial dimensions
        Xp = self.pad(X)  # -> (B, C, H+pad_h, W+pad_w)

        # 2) Synchronized batch normalization across channels
        Xn = self.bn(Xp)  # -> (B, C, H', W')

        # 3) Global average pooling over spatial dims to obtain (B, C)
        desc = Xn.mean(dim=(2, 3))  # -> (B, C)

        # 4) Project both descriptors and auxiliary vector into shared hidden space
        ch_proj = F.relu(self.lin_channel(desc))  # -> (B, hidden_dim)
        aux_proj = F.relu(self.lin_aux(V))        # -> (B, hidden_dim)

        # 5) Create two interaction patterns:
        #    a) elementwise multiplication (gating-like)
        #    b) additive residual
        interacted_mul = ch_proj * aux_proj       # -> (B, hidden_dim)
        interacted_add = ch_proj + aux_proj       # -> (B, hidden_dim)

        # Concatenate the two interaction patterns
        combined = torch.cat([interacted_mul, interacted_add], dim=1)  # -> (B, hidden_dim*2)

        # 6) Final projection
        out_main = self.lin_out(combined)  # -> (B, out_dim)

        # 7) Additional gating: produce a gate from ch_proj to modulate out_main
        gate = torch.sigmoid(self.lin_gate(ch_proj))  # -> (B, out_dim)

        # 8) Apply gating to produce final output
        out = out_main * gate  # -> (B, out_dim)

        return out

# Configuration / dimensions
BATCH = 8
C = 64        # input channels
H = 32
W = 32
AUX_DIM = 16  # dimensionality of auxiliary vector V
HIDDEN = 128
OUT_DIM = 10
PADDING = (2, 2, 1, 1)  # (left, right, top, bottom)

def get_inputs():
    """
    Generate a batch of image-like tensors and auxiliary vectors.

    Returns:
        list: [X, V] where
              X: Tensor of shape (BATCH, C, H, W)
              V: Tensor of shape (BATCH, AUX_DIM)
    """
    X = torch.randn(BATCH, C, H, W)
    V = torch.randn(BATCH, AUX_DIM)
    return [X, V]

def get_init_inputs():
    """
    Return the initialization parameters used to construct the Model instance.

    Returns:
        list: [in_channels, aux_dim, hidden_dim, out_dim, pad]
    """
    return [C, AUX_DIM, HIDDEN, OUT_DIM, PADDING]