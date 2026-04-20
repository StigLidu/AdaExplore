import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Complex module that fuses 3D volumetric context with a token sequence.
    Pipeline:
      - Apply Dropout3d to volumetric input.
      - Spatially average the volume to produce a channel-level context vector.
      - Project the channel context into the token embedding space.
      - Prepend the projected context as a "global token" to the input token sequence.
      - Use LPPool1d to downsample the token sequence (including the global token).
      - Process the downsampled sequence with a TransformerEncoderLayer (batch_first=True).
      - Extract the transformed global token, combine it elementwise with another projection of the volume,
        and run a final linear layer to produce the output embedding.
    """
    def __init__(self,
                 in_channels: int,
                 embed_dim: int,
                 out_dim: int,
                 transformer_nhead: int = 8,
                 transformer_ff: int = None,
                 dropout3d_p: float = 0.25,
                 lppool_norm: int = 2,
                 lppool_kernel: int = 3,
                 lppool_stride: int = 2):
        super(Model, self).__init__()

        if transformer_ff is None:
            transformer_ff = embed_dim * 4

        # Drop entire channels in the 3D volume
        self.dropout3d = nn.Dropout3d(p=dropout3d_p)

        # Project averaged volume channels into token embedding space
        self.proj_from_vol = nn.Linear(in_channels, embed_dim)
        # Another projection of volume for elementwise combination later
        self.proj_vol_fuse = nn.Linear(in_channels, embed_dim)

        # 1D Lp pooling over the token (sequence) dimension. Expects (N, C, L)
        self.lp_pool = nn.LPPool1d(norm_type=lppool_norm, kernel_size=lppool_kernel, stride=lppool_stride)

        # Transformer encoder layer operating on (batch, seq_len, embed_dim)
        self.transformer = nn.TransformerEncoderLayer(d_model=embed_dim,
                                                      nhead=transformer_nhead,
                                                      dim_feedforward=transformer_ff,
                                                      dropout=0.1,
                                                      activation="relu",
                                                      batch_first=True)

        # Final projection to desired output dimension
        self.final_fc = nn.Linear(embed_dim, out_dim)

    def forward(self, vol: torch.Tensor, seq: torch.Tensor) -> torch.Tensor:
        """
        Args:
            vol: volumetric input tensor of shape (batch, channels, D, H, W)
            seq: token sequence tensor of shape (batch, seq_len, embed_dim)

        Returns:
            Tensor of shape (batch, out_dim)
        """
        # 1) Stochastically drop channels in the 3D volume
        vol_dropped = self.dropout3d(vol)

        # 2) Global average pool the spatial dims of the volume -> (batch, channels)
        # Using adaptive_avg_pool3d to always reduce to (1,1,1)
        batch = vol_dropped.size(0)
        channels = vol_dropped.size(1)
        vol_pooled = F.adaptive_avg_pool3d(vol_dropped, output_size=(1, 1, 1)).view(batch, channels)

        # 3) Project the pooled channel vector into the token embedding space -> (batch, embed_dim)
        vol_emb = self.proj_from_vol(vol_pooled)

        # 4) Prepend the volumetric global token to the sequence: new_seq_len = seq_len + 1
        seq_with_vol = torch.cat([vol_emb.unsqueeze(1), seq], dim=1)

        # 5) LPPool1d expects (N, C, L) so permute: (batch, embed_dim, L)
        seq_perm = seq_with_vol.permute(0, 2, 1)

        # 6) Downsample the token sequence using Lp pooling -> (batch, embed_dim, L_down)
        seq_pooled = self.lp_pool(seq_perm)

        # 7) Restore (batch, L_down, embed_dim) for transformer
        seq_down = seq_pooled.permute(0, 2, 1)

        # 8) Transformer encoder processes the reduced sequence
        transformed = self.transformer(seq_down)

        # 9) Extract the transformed global token (first token)
        global_token = transformed[:, 0, :]  # (batch, embed_dim)

        # 10) Fuse with another projection of the volume via elementwise multiplication
        vol_fuse = self.proj_vol_fuse(vol_pooled)  # (batch, embed_dim)
        fused = global_token * vol_fuse

        # 11) Final projection
        out = self.final_fc(fused)  # (batch, out_dim)
        return out

# Configuration / hyperparameters
batch_size = 8
channels = 16
D = 4
H = 8
W = 8

seq_len = 128
embed_dim = 64
out_dim = 32

def get_inputs():
    # Volumetric input: (batch, channels, D, H, W)
    vol = torch.randn(batch_size, channels, D, H, W)
    # Token sequence: (batch, seq_len, embed_dim)
    seq = torch.randn(batch_size, seq_len, embed_dim)
    return [vol, seq]

def get_init_inputs():
    # Return initialization inputs (if any). We provide the constructor args for Model for reproducibility.
    return [channels, embed_dim, out_dim]