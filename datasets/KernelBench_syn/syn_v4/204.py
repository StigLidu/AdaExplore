import torch
import torch.nn as nn

# Configuration variables
BATCH = 4        # Batch size
CHANNELS = 16    # Number of channels
DEPTH = 6        # Depth dimension (e.g., time or slices)
HEIGHT = 64      # Height of spatial map
WIDTH = 48       # Width of spatial map

# Padding for 2D replication padding: (left, right, top, bottom)
PAD2D = (1, 1, 2, 2)

# 3D pooling configuration: (depth, height, width)
POOL_KERNEL = (1, 3, 3)
POOL_STRIDE = (1, 2, 2)

# Dropout probability for Dropout1d
DROPOUT_P = 0.25

# Output channel projection dimension for internal linear mixing
PROJ_DIM = CHANNELS  # we keep same channel dim here, could differ


class Model(nn.Module):
    """
    Complex multi-step module that demonstrates interplay between 2D padding,
    3D pooling, channel-wise dropout, and a learned per-channel projection.

    Input:
        x: Tensor of shape (BATCH, CHANNELS, DEPTH, HEIGHT, WIDTH)

    Computation overview:
        1. For each depth slice, apply replication padding in HxW via ReplicationPad2d.
           (We merge batch and depth into a single 4D batch dim to reuse ReplicationPad2d.)
        2. Restore to 5D and apply MaxPool3d across (D,H,W).
        3. Collapse spatial HxW into a sequence length L and apply Dropout1d to zero
           out entire channels for each (batch*depth) sample.
        4. Perform a learned channel projection (nn.Linear) applied to each spatial location.
        5. Aggregate across spatial positions (mean) to produce a per-(batch,depth,channel)
           descriptor, then permute to (batch, channel, depth) and apply a depth-wise softmax
           to emphasize depth-located features.
    """
    def __init__(self):
        super(Model, self).__init__()
        # Pad 2D boundaries by replication for each (batch, depth)-slice
        self.pad2d = nn.ReplicationPad2d(PAD2D)
        # 3D max pooling to reduce spatial dims (and optionally depth)
        self.pool3d = nn.MaxPool3d(kernel_size=POOL_KERNEL, stride=POOL_STRIDE)
        # Dropout1d to randomly zero entire channels per sample across the sequence length
        self.drop1d = nn.Dropout1d(p=DROPOUT_P)
        # Linear projection that mixes channels (applied per spatial location)
        # We'll apply this to tensors shaped (N, L, C) treating last dim as features
        self.channel_proj = nn.Linear(CHANNELS, PROJ_DIM, bias=True)

        # A small normalization layer to stabilize the aggregated descriptors
        self.norm = nn.LayerNorm(PROJ_DIM)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor with shape (B, C, D, H, W)

        Returns:
            torch.Tensor: Output tensor with shape (B, C, Dp) where Dp is pooled depth.
                          Values are normalized and softmaxed across the depth dim.
        """
        B, C, D, H, W = x.shape  # expected to match config, but keep generality

        # 1) Merge batch and depth to use ReplicationPad2d (expects 4D)
        #    permute to (B, D, C, H, W) -> reshape to (B*D, C, H, W)
        x_4d = x.permute(0, 2, 1, 3, 4).reshape(B * D, C, H, W)

        # Apply 2D replication padding to each (batch*depth) slice
        x_padded = self.pad2d(x_4d)  # shape: (B*D, C, H_p, W_p)

        # Restore to 5D: (B, D, C, H_p, W_p) -> permute to (B, C, D, H_p, W_p) for pool3d
        H_p = x_padded.shape[2]
        W_p = x_padded.shape[3]
        x_5d = x_padded.reshape(B, D, C, H_p, W_p).permute(0, 2, 1, 3, 4)  # (B, C, D, H_p, W_p)

        # 2) Apply 3D max pooling
        pooled = self.pool3d(x_5d)  # shape: (B, C, Dp, Hp, Wp)
        Bp, Cp, Dp, Hp, Wp = pooled.shape

        # 3) Collapse spatial HxW into sequence length L for Dropout1d
        # Reshape to (B, C, Dp, L) then merge (B*Dp) into batch for Dropout1d: (B*Dp, C, L)
        L = Hp * Wp
        pooled_seq = pooled.reshape(Bp, Cp, Dp, L).permute(0, 2, 1, 3).reshape(Bp * Dp, Cp, L)

        # Apply Dropout1d: zeros out entire channels for each (B*Dp) sample
        dropped = self.drop1d(pooled_seq)  # (B*Dp, C, L)

        # 4) Prepare for channel projection: move sequence length to middle for nn.Linear
        # shape -> (B*Dp, L, C)
        dropped_t = dropped.permute(0, 2, 1)

        # Apply learned channel projection per spatial position
        projected = self.channel_proj(dropped_t)  # (B*Dp, L, PROJ_DIM)

        # 5) Aggregate across spatial positions (mean pooling) -> (B*Dp, PROJ_DIM)
        agg = projected.mean(dim=1)

        # Normalize aggregated descriptors
        agg_norm = self.norm(agg)  # (B*Dp, PROJ_DIM)

        # Reshape back to (B, Dp, C) (we keep PROJ_DIM == C for compatibility)
        agg_bd = agg_norm.reshape(Bp, Dp, PROJ_DIM)

        # Permute to (B, C, Dp)
        out = agg_bd.permute(0, 2, 1)

        # Apply softmax across depth to highlight depth-localized features per channel
        out = torch.softmax(out, dim=-1)

        return out


def get_inputs():
    """
    Generates a batch of 5D tensors to exercise the module.

    Returns:
        list: A single-element list containing input tensor x of shape (BATCH, CHANNELS, DEPTH, HEIGHT, WIDTH)
    """
    x = torch.randn(BATCH, CHANNELS, DEPTH, HEIGHT, WIDTH)
    return [x]


def get_init_inputs():
    """
    Returns any initialization parameters required for the module.
    For this model there are no extra inputs required to initialize beyond the class construction.

    Returns:
        list: Empty list.
    """
    return []