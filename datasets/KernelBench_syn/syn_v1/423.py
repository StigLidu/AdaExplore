import torch
import torch.nn as nn

# Configuration / shapes
B = 8          # batch size
C1 = 16        # channels for 3D input A
D = 4          # depth for 3D input A
H = 32         # height for inputs
W = 32         # width for inputs
C2 = 24        # channels for 2D input B

ADAPT_H = 7    # adaptive pooling output height
ADAPT_W = 7    # adaptive pooling output width

PROJ_OUT = 64      # output channels for linear projection
FINAL_OUT = 128    # final output feature size after final matmul

class Model(nn.Module):
    """
    Complex module combining 3D max-pooling, spatial-channel reshaping,
    adaptive average pooling, spatial softmax, and cross-tensor attention-like
    contraction followed by a learned projection and final matmul.

    Forward computation (high-level):
        1. MaxPool3d on A to reduce spatial resolution.
        2. Collapse the depth dimension into channels -> treat depth as extra channels.
        3. AdaptiveAvgPool2d to fixed spatial resolution (ADAPT_H x ADAPT_W).
        4. Apply Softmax2d across channels at each spatial location.
        5. AdaptiveAvgPool2d on B to same spatial resolution.
        6. Compute cross-correlation via einsum over spatial dims:
               T[b, c_collapsed, k] = sum_{p,q} S[b,c_collapsed,p,q] * B_pooled[b,k,p,q]
        7. Apply a learnable linear projection over the k dimension to produce features.
        8. Average over collapsed channels and final matmul with external matrix M.

    Inputs:
        A: torch.Tensor of shape (B, C1, D, H, W)
        B: torch.Tensor of shape (B, C2, H, W)
        M: torch.Tensor of shape (PROJ_OUT, FINAL_OUT)  # external projection matrix

    Output:
        torch.Tensor of shape (B, FINAL_OUT)
    """
    def __init__(self):
        super(Model, self).__init__()
        # 3D max pooling reduces depth, height, width by factor of 2
        self.pool3d = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        # Adaptive avg pool to a fixed 2D spatial size for both streams
        self.adapt2d_A = nn.AdaptiveAvgPool2d((ADAPT_H, ADAPT_W))
        self.adapt2d_B = nn.AdaptiveAvgPool2d((ADAPT_H, ADAPT_W))
        # Softmax over channels for each spatial location
        self.softmax2d = nn.Softmax2d()
        # Linear projection applied to the C2 dimension after contraction
        self.proj = nn.Linear(C2, PROJ_OUT, bias=True)
        # Small non-linearity for stability
        self.relu = nn.ReLU()

    def forward(self, A: torch.Tensor, B_tensor: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
        """
        Forward pass as described in the class docstring.

        Args:
            A: (B, C1, D, H, W)
            B_tensor: (B, C2, H, W)
            M: (PROJ_OUT, FINAL_OUT)

        Returns:
            out: (B, FINAL_OUT)
        """
        # 1) Reduce spatial resolution (depth, height, width)
        A_pooled = self.pool3d(A)  # (B, C1, D//2, H//2, W//2)

        # 2) Collapse depth into channel dimension: (B, C1 * D2, H2, W2)
        B_dim = A_pooled.size()
        # B_dim: (B, C1, D2, H2, W2)
        _, c1, d2, h2, w2 = B_dim
        A_collapsed = A_pooled.view(A_pooled.size(0), c1 * d2, h2, w2)

        # 3) Adaptive average pool to fixed spatial dims
        A_spatial = self.adapt2d_A(A_collapsed)  # (B, Cx, ADAPT_H, ADAPT_W)
        B_spatial = self.adapt2d_B(B_tensor)     # (B, C2, ADAPT_H, ADAPT_W)

        # 4) Softmax over channels for A_spatial at each (H,W)
        S = self.softmax2d(A_spatial)  # (B, Cx, ADAPT_H, ADAPT_W)

        # 5) Cross-tensor spatial contraction: sum over spatial dims p,q
        # Result T[b, cx, k] = sum_{p,q} S[b, cx, p, q] * B_spatial[b, k, p, q]
        T = torch.einsum("bcpq,bkpq->bck", S, B_spatial)  # (B, Cx, C2)

        # 6) Project the C2 axis to PROJ_OUT using a linear layer applied to last dim
        # Linear expects (..., C2) so this works directly
        projected = self.proj(T)  # (B, Cx, PROJ_OUT)
        projected = self.relu(projected)

        # 7) Aggregate over the collapsed channels (mean) -> (B, PROJ_OUT)
        aggregated = projected.mean(dim=1)

        # 8) Final matmul with external matrix M to produce FINAL_OUT features
        out = aggregated.matmul(M)  # (B, FINAL_OUT)

        return out

def get_inputs():
    # Create input tensors with the configured shapes
    A = torch.randn(B, C1, D, H, W)
    B_tensor = torch.randn(B, C2, H, W)
    # External matrix M to map PROJ_OUT -> FINAL_OUT
    M = torch.randn(PROJ_OUT, FINAL_OUT)
    return [A, B_tensor, M]

def get_init_inputs():
    # No special initialization inputs needed for this module; parameters are internal
    return []