import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Model(nn.Module):
    """
    Complex vision-processing module that combines PixelUnshuffle, GroupNorm,
    Dropout1d and learned projections to produce a compact feature vector per image.

    Pipeline:
    1. PixelUnshuffle to reduce spatial resolution while increasing channels.
    2. GroupNorm over the increased channels.
    3. ReLU activation.
    4. Spatial flattening and channel-wise Dropout1d.
    5. Global spatial pooling (mean) to get per-channel descriptors.
    6. Two learned linear projections:
       - A main projection from pooled channels to out_features.
       - A gating projection that produces per-channel gates applied back to spatial maps
         (demonstrates mixing of spatial and channel operations).
    7. The final output concatenates the main projection with a secondary summary (max-pooled
       gated-response) to form the returned feature vector.
    """
    def __init__(self,
                 in_channels: int,
                 downscale_factor: int,
                 num_groups: int,
                 dropout_p: float,
                 out_features: int):
        super(Model, self).__init__()
        self.in_channels = in_channels
        self.downscale = downscale_factor
        # After PixelUnshuffle: channels' = in_channels * downscale_factor^2
        self.post_channels = in_channels * (downscale_factor ** 2)

        # Pixel unshuffle layer (reduces H,W by factor downscale_factor)
        self.pixel_unshuffle = nn.PixelUnshuffle(downscale_factor)

        # GroupNorm over the expanded channels (num_groups must divide post_channels)
        # If requested num_groups doesn't divide, we clamp it to a divisor.
        if self.post_channels % num_groups != 0:
            # find largest divisor <= num_groups (but at least 1)
            divisor = 1
            for g in range(min(num_groups, self.post_channels), 0, -1):
                if self.post_channels % g == 0:
                    divisor = g
                    break
            self.num_groups = divisor
        else:
            self.num_groups = num_groups
        self.gn = nn.GroupNorm(self.num_groups, self.post_channels)

        # Channel-wise dropout across the spatial length (expects input shaped (N, C, L))
        self.dropout = nn.Dropout1d(p=dropout_p)

        # Learned projection parameters (using Parameters rather than nn.Linear to show explicit init)
        self.out_features = out_features
        # Main projection: post_channels -> out_features
        self.proj_weight = nn.Parameter(torch.empty(self.post_channels, out_features))
        self.proj_bias = nn.Parameter(torch.empty(out_features))

        # Gating projection: post_channels -> post_channels (produces a gate per channel)
        self.gate_weight = nn.Parameter(torch.empty(self.post_channels, self.post_channels))
        self.gate_bias = nn.Parameter(torch.empty(self.post_channels))

        # Initialize weights with a stable scheme
        nn.init.kaiming_uniform_(self.proj_weight, a=math.sqrt(5))
        fan_in = self.proj_weight.size(0)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.proj_bias, -bound, bound)

        nn.init.kaiming_uniform_(self.gate_weight, a=math.sqrt(5))
        fan_in_g = self.gate_weight.size(0)
        bound_g = 1 / math.sqrt(fan_in_g)
        nn.init.uniform_(self.gate_bias, -bound_g, bound_g)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input image tensor of shape (N, in_channels, H, W).
                              H and W must be divisible by downscale_factor.

        Returns:
            torch.Tensor: Feature tensor of shape (N, out_features + post_channels),
                          concatenation of main projection and gated max summary.
        """
        # Step 1: PixelUnshuffle -> (N, post_channels, H', W')
        y = self.pixel_unshuffle(x)

        # Step 2: GroupNorm across channels
        y = self.gn(y)

        # Step 3: Non-linearity
        y = F.relu(y)

        N, C, Hs, Ws = y.shape  # C == self.post_channels
        L = Hs * Ws

        # Step 4: Flatten spatial dims so Dropout1d can zero entire channels consistently
        y_flat = y.view(N, C, L)  # shape (N, C, L)
        y_flat = self.dropout(y_flat)

        # Step 5: Global pooling (mean) to get per-channel descriptors
        pooled = y_flat.mean(dim=2)  # shape (N, C)

        # Step 6a: Main learned projection -> (N, out_features)
        main_out = pooled @ self.proj_weight + self.proj_bias  # linear projection

        # Step 6b: Gating: produce per-channel gates from pooled descriptors
        gates = torch.sigmoid(pooled @ self.gate_weight + self.gate_bias)  # (N, C)

        # Apply gates back to the spatial maps (broadcast over spatial length)
        gated_spatial = y_flat * gates.unsqueeze(2)  # (N, C, L)

        # Summary of gated response: max over spatial positions per channel, then mean across channels -> (N,)
        # We'll also keep the per-channel max summary to concatenate
        per_channel_max = gated_spatial.max(dim=2).values  # (N, C)

        # Secondary summary: mean of per-channel max (gives single scalar per example)
        secondary_scalar = per_channel_max.mean(dim=1, keepdim=True)  # (N, 1)

        # Final output: concatenate main_out with per-channel max summary compressed to a vector
        # To keep size manageable, reduce per_channel_max via a simple linear combine into out_features sized vector
        # We reuse proj_weight's transpose (detached) to map C -> out_features in a parameter-free way
        # (This is just a deterministic mixing to produce a balanced output size.)
        mix = per_channel_max @ (self.proj_weight / (torch.norm(self.proj_weight, dim=0, keepdim=True) + 1e-6))  # (N, out_features)

        final = torch.cat([main_out, mix], dim=1)  # shape (N, out_features * 2)

        return final

# Configuration variables
batch_size = 8
in_channels = 3
height = 64
width = 64
downscale_factor = 2
num_groups = 3
dropout_p = 0.25
out_features = 128

def get_inputs():
    """
    Returns:
        list: single input tensor shaped (batch_size, in_channels, height, width)
    """
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]

def get_init_inputs():
    """
    Returns:
        list: initialization arguments for Model: [in_channels, downscale_factor, num_groups, dropout_p, out_features]
    """
    return [in_channels, downscale_factor, num_groups, dropout_p, out_features]