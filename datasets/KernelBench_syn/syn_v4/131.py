import torch
import torch.nn as nn

# Configuration / module-level constants
BATCH = 8
CHANNELS = 64
DEPTH = 16
HEIGHT = 32
WIDTH = 32

POOL_OUT = (4, 4, 4)    # Adaptive pooling output spatial dims
PROJ_DIM = 512          # Dimension after first projection
CTX_DIM = 128           # Context vector dimension
FINAL_DIM = 1024        # Final output dimension after fusion

class Model(nn.Module):
    """
    Complex model that demonstrates a multi-branch 3D feature extractor with:
      - AdaptiveMaxPool3d to reduce spatial resolution
      - Linear projection of pooled features
      - A small channel-attention branch derived from projections (using LogSigmoid)
      - Fusion of a context vector with the projected features
      - Final elementwise modulation and LogSigmoid activation
    
    Inputs:
      - vol: a 5D tensor of shape (BATCH, CHANNELS, DEPTH, HEIGHT, WIDTH)
      - ctx: a 2D context tensor of shape (BATCH, CTX_DIM)
    
    Output:
      - Tensor of shape (BATCH, FINAL_DIM) after feature fusion and LogSigmoid activation.
    """
    def __init__(self):
        super(Model, self).__init__()
        # Spatial reducer
        self.pool = nn.AdaptiveMaxPool3d(output_size=POOL_OUT)
        # Project flattened pooled features to a compact latent
        pooled_size = CHANNELS * POOL_OUT[0] * POOL_OUT[1] * POOL_OUT[2]
        self.proj = nn.Linear(pooled_size, PROJ_DIM, bias=True)
        # Small attention branch: map projected features -> per-channel logits
        self.attn = nn.Linear(PROJ_DIM, CHANNELS, bias=True)
        # Map context + projected features to a final representation
        self.fusion = nn.Linear(PROJ_DIM + CTX_DIM, FINAL_DIM, bias=True)
        # Map channel summary into same FINAL_DIM for elementwise modulation
        self.summary_lin = nn.Linear(CHANNELS, FINAL_DIM, bias=True)
        # Use LogSigmoid as the final nonlinearity (and intermediate channel gating)
        self.logsigmoid = nn.LogSigmoid()
        # A simple nonlinearity between projections
        self.relu = nn.ReLU(inplace=True)

    def forward(self, vol: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        """
        Forward pass combining spatial pooling, linear projections, attention gating,
        context fusion, and final LogSigmoid activation.

        Steps:
          1. Adaptive max pool the 3D volume -> (BATCH, CHANNELS, 4,4,4)
          2. Flatten spatial dims and project -> (BATCH, PROJ_DIM)
          3. Produce channel attention logits from projection -> (BATCH, CHANNELS)
             and convert to gating weights using LogSigmoid + exp (i.e., sigmoid)
          4. Apply channel gates to the pooled tensor, then global-sum spatial dims
             to produce a channel summary vector -> (BATCH, CHANNELS)
          5. Map channel summary to FINAL_DIM
          6. Concatenate projection with context and fuse -> (BATCH, FINAL_DIM)
          7. Modulate fused output by summary mapping (elementwise multiply)
          8. Apply final LogSigmoid and return
        """
        B = vol.shape[0]

        # 1) Spatial reduction
        pooled = self.pool(vol)  # shape: (B, C, pD, pH, pW)

        # 2) Flatten spatial dims and project
        flattened = pooled.view(B, -1)  # shape: (B, CHANNELS * pD * pH * pW)
        proj_feat = self.proj(flattened)  # shape: (B, PROJ_DIM)
        proj_feat = self.relu(proj_feat)

        # 3) Channel attention logits -> use LogSigmoid -> exp to get sigmoid
        attn_logits = self.attn(proj_feat)  # shape: (B, CHANNELS)
        # LogSigmoid gives log(sigmoid(x)); exp -> sigmoid(x)
        channel_gates = torch.exp(self.logsigmoid(attn_logits)).view(B, CHANNELS, 1, 1, 1)

        # 4) Apply per-channel gates to pooled features and summarize spatially
        gated_pooled = pooled * channel_gates  # shape: (B, C, pD, pH, pW)
        # Global sum over spatial dims -> channel summary
        channel_summary = gated_pooled.sum(dim=(2, 3, 4))  # shape: (B, CHANNELS)

        # 5) Map channel summary into FINAL_DIM
        summary_proj = self.summary_lin(channel_summary)  # shape: (B, FINAL_DIM)
        summary_proj = self.relu(summary_proj)

        # 6) Concatenate projection with context and fuse
        fused_in = torch.cat([proj_feat, ctx], dim=1)  # shape: (B, PROJ_DIM + CTX_DIM)
        fused = self.fusion(fused_in)  # shape: (B, FINAL_DIM)
        fused = self.relu(fused)

        # 7) Modulate fused representation by channel-derived summary (elementwise)
        modulated = fused * summary_proj  # shape: (B, FINAL_DIM)

        # 8) Final activation with LogSigmoid
        out = self.logsigmoid(modulated)  # shape: (B, FINAL_DIM)
        return out

# Input configuration for testing / benchmarking
B = BATCH
C = CHANNELS
D = DEPTH
H = HEIGHT
W = WIDTH
CTX = CTX_DIM

def get_inputs():
    """
    Returns a list of input tensors for the Model.forward:
      - vol: random 5D tensor (BATCH, CHANNELS, DEPTH, HEIGHT, WIDTH)
      - ctx: random context tensor (BATCH, CTX_DIM)
    """
    vol = torch.randn(B, C, D, H, W)
    ctx = torch.randn(B, CTX)
    return [vol, ctx]

def get_init_inputs():
    """
    No special initialization inputs required beyond default layer params.
    """
    return []