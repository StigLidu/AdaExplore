import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Complex model combining 3D batch normalization, 2D constant padding, and a TransformerDecoder.
    The model ingests a 5D volumetric tensor (N, C, D, H, W) and a target sequence (T, N, E).
    Processing steps:
      - Apply BatchNorm3d over the volumetric input.
      - Compute two different pooled summaries:
          1) Global mean over (D, H, W).
          2) Mean from a middle depth slice after 2D constant padding.
      - Project both summaries into the transformer's embedding space (d_model) and stack them to form the 'memory'
        sequence consumed by the TransformerDecoder.
      - Run the TransformerDecoder with the provided target sequence and the constructed memory.
      - Return the decoder outputs aggregated over the target time dimension.
    """
    def __init__(self, num_features: int, d_model: int, nhead: int, num_layers: int, pad_value: float):
        """
        Args:
            num_features: Number of channels/features in the volumetric input (C).
            d_model: Transformer embedding dimension (must be divisible by nhead).
            nhead: Number of attention heads in the TransformerDecoderLayer.
            num_layers: Number of layers in the TransformerDecoder stack.
            pad_value: Constant value for ConstantPad2d padding.
        """
        super(Model, self).__init__()
        # Normalize across the channel dimension for 5D inputs: (N, C, D, H, W)
        self.bn3d = nn.BatchNorm3d(num_features)

        # 2D constant padding for a middle depth slice (left, right, top, bottom)
        # Choose a moderate padding amount that could change spatial statistics meaningfully
        self.pad2d = nn.ConstantPad2d((1, 1, 2, 2), pad_value)

        # Linear projection from channel space (num_features) to transformer embedding (d_model)
        self.proj = nn.Linear(num_features, d_model)

        # Transformer decoder stack: build from TransformerDecoderLayer
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Optional small feed-forward after decoding to mix outputs (keeps output shape consistent)
        self.post_ff = nn.Linear(d_model, d_model)

    def forward(self, vol: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            vol: Volumetric input tensor with shape (N, C, D, H, W).
            tgt: Target sequence for the TransformerDecoder with shape (T, N, d_model).

        Returns:
            Tensor of shape (N, d_model) representing the aggregated decoder output per batch element.
        """
        # Validate shapes minimally to provide clearer error messages
        if vol.dim() != 5:
            raise ValueError(f"vol must be a 5D tensor (N, C, D, H, W), got shape {tuple(vol.shape)}")
        if tgt.dim() != 3:
            raise ValueError(f"tgt must be a 3D tensor (T, N, d_model), got shape {tuple(tgt.shape)}")

        N, C, D, H, W = vol.shape

        # 1) Batch normalization across the volumetric input
        x = self.bn3d(vol)  # shape (N, C, D, H, W)

        # 2) Global mean pooling over depth and spatial dims -> (N, C)
        mean_global = x.mean(dim=(2, 3, 4))  # (N, C)

        # 3) Extract the central depth slice and apply 2D constant padding then pool -> (N, C)
        mid_idx = D // 2
        slice_mid = x[:, :, mid_idx, :, :]  # (N, C, H, W)
        padded = self.pad2d(slice_mid)      # (N, C, H + pad_top+pad_bottom, W + pad_left+pad_right)
        mean_slice = padded.mean(dim=(2, 3))  # (N, C)

        # 4) Project both summaries into d_model space -> (N, d_model)
        mem0 = self.proj(mean_global)  # (N, d_model)
        mem1 = self.proj(mean_slice)   # (N, d_model)

        # 5) Stack projected summaries to create memory sequence for TransformerDecoder
        #    memory shape -> (S, N, d_model) where S=2 (two summary tokens)
        memory = torch.stack([mem0, mem1], dim=0)  # (2, N, d_model)

        # 6) Run the TransformerDecoder: tgt is expected shape (T, N, d_model)
        decoded = self.decoder(tgt, memory)  # (T, N, d_model)

        # 7) Aggregate decoder outputs across the temporal dimension (mean over T), apply final feed-forward
        out = decoded.mean(dim=0)            # (N, d_model)
        out = self.post_ff(out)              # (N, d_model)

        return out


# Configuration / default sizes
BATCH = 8
C = 16            # num_features for BatchNorm3d
D = 8
H = 32
W = 32

D_MODEL = 64      # must be divisible by NHEAD
NHEAD = 8
NUM_LAYERS = 2
TGT_LEN = 10
PAD_VALUE = 0.1

def get_inputs():
    """
    Create example input tensors:
      - vol: volumetric input (N, C, D, H, W)
      - tgt: transformer target sequence (T, N, d_model)
    """
    vol = torch.randn(BATCH, C, D, H, W)
    # Transformer expects (T, N, E)
    tgt = torch.randn(TGT_LEN, BATCH, D_MODEL)
    return [vol, tgt]

def get_init_inputs():
    """
    Initialization parameters for Model constructor in the order:
      num_features, d_model, nhead, num_layers, pad_value
    """
    return [C, D_MODEL, NHEAD, NUM_LAYERS, PAD_VALUE]