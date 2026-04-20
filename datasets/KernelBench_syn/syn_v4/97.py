import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Complex 3D feature aggregation module:
    - Projects per-voxel channel features to a hidden embedding (nn.Linear applied to channels)
    - Applies ReLU6 non-linearity
    - Rearranges to (N, hidden, D, H, W) and applies 3D average pooling
    - Computes a spatial summary (mean) of pooled features
    - Produces gating logits from the summary (nn.Linear) and converts them to sigmoid via LogSigmoid->exp
    - Combines the gated coefficients with an external projection matrix B to produce final outputs
    """
    def __init__(self, in_channels: int, hidden_features: int, out_features: int, pool_kernel: int = 2):
        super(Model, self).__init__()
        # Project per-voxel channel vector to a hidden embedding
        self.proj = nn.Linear(in_channels, hidden_features, bias=True)
        # Small MLP head to produce gating logits from pooled spatial summary
        self.gate = nn.Linear(hidden_features, out_features, bias=True)
        # Non-linearities and pooling
        self.relu6 = nn.ReLU6()
        self.logsig = nn.LogSigmoid()
        self.pool = nn.AvgPool3d(kernel_size=pool_kernel, stride=pool_kernel, padding=0)
        # Small learned scalar bias to allow shifting final outputs
        self.out_bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input volumetric tensor of shape (N, C, D, H, W)
            B: External projection matrix of shape (hidden_features, out_features)

        Returns:
            Tensor of shape (N, out_features)
        """
        N, C, D, H, W = x.shape

        # 1) Move channels to last dim and flatten spatial dims: (N, S, C) where S = D*H*W
        S = D * H * W
        x_flat = x.permute(0, 2, 3, 4, 1).contiguous().view(N, S, C)

        # 2) Channel-wise linear projection -> non-linearity: (N, S, hidden)
        hidden_tokens = self.proj(x_flat)          # (N, S, hidden)
        hidden_tokens = self.relu6(hidden_tokens)  # elementwise ReLU6

        # 3) Reshape back to volumetric form for pooling: (N, hidden, D, H, W)
        hidden_vol = hidden_tokens.view(N, D, H, W, -1).permute(0, 4, 1, 2, 3).contiguous()

        # 4) 3D average pooling reduces spatial resolution
        pooled = self.pool(hidden_vol)  # (N, hidden, D', H', W')

        # 5) Spatial summary: mean across spatial dims -> (N, hidden)
        pooled_flat = pooled.view(N, pooled.shape[1], -1)
        spatial_summary = pooled_flat.mean(dim=2)  # global summary per-sample

        # 6) Produce gating logits and convert to sigmoid via LogSigmoid->exp for numerical stability
        gate_logits = self.gate(spatial_summary)       # (N, out_features)
        gating = torch.exp(self.logsig(gate_logits))   # (N, out_features), values in (0,1)

        # 7) External projection: multiply summary by provided matrix B (hidden -> out_features)
        # Ensure B has compatible shape
        # spatial_summary: (N, hidden) ; B: (hidden, out_features) -> result: (N, out_features)
        global_projection = spatial_summary @ B

        # 8) Elementwise gating, add learned bias, and final non-linearity
        out = gating * global_projection + self.out_bias
        out = self.relu6(out)

        return out

# Configuration / default sizes for test inputs
BATCH = 4
IN_CHANNELS = 16
DEPTH = 8
HEIGHT = 16
WIDTH = 16
HIDDEN = 32
OUT_FEATURES = 64
POOL_KERNEL = 2

def get_inputs():
    """
    Returns the actual input tensors used in forward:
    - x: volumetric input (N, C, D, H, W)
    - B: external projection matrix (hidden_features, out_features)
    """
    x = torch.randn(BATCH, IN_CHANNELS, DEPTH, HEIGHT, WIDTH)
    B = torch.randn(HIDDEN, OUT_FEATURES)
    return [x, B]

def get_init_inputs():
    """
    Returns the parameters to initialize the Model instance:
    [in_channels, hidden_features, out_features, pool_kernel]
    """
    return [IN_CHANNELS, HIDDEN, OUT_FEATURES, POOL_KERNEL]