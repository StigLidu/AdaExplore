import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Complex feature-modulation block that:
    - Applies synchronized batch normalization to spatial inputs
    - Aggregates spatial information via global average pooling
    - Fuses a conditional/style vector with pooled features
    - Produces per-channel scale and bias via a small MLP
    - Applies feature-wise modulation, a SiLU non-linearity, and FeatureAlphaDropout

    Inputs:
        x: Tensor of shape (B, C, H, W)
        style: Tensor of shape (B, style_dim)

    Output:
        Tensor of shape (B, C, H, W)
    """
    def __init__(self, num_features: int, style_dim: int, dropout_p: float = 0.1):
        super(Model, self).__init__()
        self.num_features = num_features
        self.style_dim = style_dim

        # Normalize features across batch & spatial dims (SyncBatchNorm follows BatchNorm semantics)
        self.syncbn = nn.SyncBatchNorm(num_features)

        # Small MLP to produce per-channel scale (gamma) and bias (beta) from pooled features + style
        # Hidden layer keeps capacity modest
        hidden = max(num_features // 2, 16)
        self.fc = nn.Sequential(
            nn.Linear(num_features + style_dim, hidden),
            nn.SiLU(),  # internal non-linearity
            nn.Linear(hidden, 2 * num_features)  # outputs gamma and beta concatenated
        )

        # SiLU as explicit activation after modulation
        self.silu = nn.SiLU()

        # FeatureAlphaDropout will randomly drop entire channels, preserving self-normalizing properties
        self.dropout = nn.FeatureAlphaDropout(p=dropout_p)

    def forward(self, x: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        """
        Forward pass implementing conditional feature modulation.

        Steps:
            1. Synchronized batch normalization on input features.
            2. Global average pooling to obtain a (B, C) descriptor.
            3. Concatenate pooled descriptor with external style vector.
            4. Predict per-channel gamma and beta via small MLP.
            5. Modulate normalized features: y = x * (1 + gamma) + beta
            6. Apply SiLU non-linearity and FeatureAlphaDropout.

        Args:
            x (torch.Tensor): Input tensor (B, C, H, W).
            style (torch.Tensor): Conditioning tensor (B, style_dim).

        Returns:
            torch.Tensor: Output tensor (B, C, H, W).
        """
        # 1. Normalize
        x_norm = self.syncbn(x)

        # 2. Global average pooling over spatial dims -> (B, C)
        pooled = x_norm.mean(dim=[2, 3])

        # 3. Concatenate pooled features with style vector -> (B, C + style_dim)
        fused = torch.cat([pooled, style], dim=1)

        # 4. Predict gamma and beta -> (B, 2*C) then split
        params = self.fc(fused)
        gamma, beta = params.chunk(2, dim=1)  # each is (B, C)

        # Reshape to apply per-channel modulation
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)    # (B, C, 1, 1)

        # 5. Feature-wise modulation
        out = x_norm * (1.0 + gamma) + beta

        # 6. Non-linearity + dropout
        out = self.silu(out)
        out = self.dropout(out)

        return out

# Configuration variables
BATCH = 16
CHANNELS = 64
HEIGHT = 128
WIDTH = 128
STYLE_DIM = 32
DROPOUT_P = 0.08

def get_inputs():
    """
    Returns a list containing:
      - x: Tensor of shape (BATCH, CHANNELS, HEIGHT, WIDTH)
      - style: Tensor of shape (BATCH, STYLE_DIM)
    """
    x = torch.randn(BATCH, CHANNELS, HEIGHT, WIDTH)
    style = torch.randn(BATCH, STYLE_DIM)
    return [x, style]

def get_init_inputs():
    """
    Returns initialization arguments for Model:
      - num_features (int)
      - style_dim (int)
      - dropout_p (float)
    """
    return [CHANNELS, STYLE_DIM, DROPOUT_P]