import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Complex model combining circular padding, 2D convolutional projection,
    a TransformerDecoder stack, learned positional embeddings generated from
    2D coordinates, and a final reconstruction with Hardshrink non-linearity.

    The forward pass:
      - Circularly pad the input image.
      - Project to an embedding space via Conv2d.
      - Flatten spatial dims to a sequence and add learned 2D positional encodings.
      - Decode a learned target sequence against this memory using nn.TransformerDecoder.
      - Reshape the decoded sequence back to spatial feature maps.
      - Apply Hardshrink activation and a final Conv2d to reconstruct image channels.
      - Crop center region so output matches original spatial dimensions.
    """
    def __init__(
        self,
        in_channels: int = 3,
        embed_dim: int = 128,
        nhead: int = 8,
        num_layers: int = 3,
        tgt_len: int = 64,
        pad_size: int = 1,
        shrink_lambda: float = 0.5,
    ):
        super(Model, self).__init__()
        assert embed_dim % nhead == 0, "embed_dim must be divisible by nhead"

        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.nhead = nhead
        self.num_layers = num_layers
        self.tgt_len = tgt_len
        self.pad_size = pad_size
        self.shrink_lambda = shrink_lambda

        # Circular padding applied to the spatial input
        self.pad = nn.CircularPad2d(self.pad_size)

        # Project image to embedding space (no internal padding; padding handled outside)
        # Kernel size 3 to mix local spatial context
        self.conv_proj = nn.Conv2d(in_channels, embed_dim, kernel_size=3, padding=0)

        # Small MLP to create 2D positional encodings from normalized coordinates
        self.pos_mlp = nn.Linear(2, embed_dim)

        # Learnable target tokens used as "queries" for the decoder
        self.tgt_tokens = nn.Parameter(torch.randn(self.tgt_len, embed_dim))

        # Transformer decoder stack
        decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=nhead, dim_feedforward=embed_dim * 4, activation='relu')
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Hardshrink non-linearity
        self.hardshrink = nn.Hardshrink(lambd=self.shrink_lambda)

        # Reconstruction conv: bring embeddings back to image channels
        # Use padding=1 so spatial dims can be preserved or easily cropped to original size
        self.conv_recon = nn.Conv2d(embed_dim, in_channels, kernel_size=3, padding=1)

        # Small normalization for stability
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: input image tensor of shape (B, C, H, W)
        Returns:
            recon: reconstructed tensor of shape (B, C, H, W)
        """
        B, C, H, W = x.shape

        # 1) Circular padding
        x_pad = self.pad(x)  # (B, C, H + 2*pad, W + 2*pad)

        # 2) Project to embedding space via conv
        feat = self.conv_proj(x_pad)  # (B, E, Hf, Wf)
        B, E, Hf, Wf = feat.shape

        # 3) Flatten spatial dims into sequence for transformer memory: (S, B, E)
        memory = feat.flatten(2).permute(2, 0, 1)  # (S=Hf*Wf, B, E)

        # 4) Create learned 2D positional encodings for each spatial location
        # Generate normalized coordinate grid in range [-1, 1]
        ys = torch.linspace(-1.0, 1.0, steps=Hf, device=x.device, dtype=x.dtype)
        xs = torch.linspace(-1.0, 1.0, steps=Wf, device=x.device, dtype=x.dtype)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')  # (Hf, Wf)
        coords = torch.stack([grid_x, grid_y], dim=-1).reshape(-1, 2)  # (S, 2)
        # Map coords to positional embeddings and add to memory
        pos_emb = self.pos_mlp(coords)  # (S, E)
        pos_emb = pos_emb.unsqueeze(1)  # (S, 1, E)
        memory = memory + pos_emb  # broadcast to (S, B, E)

        # 5) Prepare target sequence (tgt_len, B, E) from learned tokens and add a small positional bias
        tgt = self.tgt_tokens.unsqueeze(1).expand(-1, B, -1).contiguous()  # (tgt_len, B, E)
        # Optionally add a tiny learned temporal bias for the target
        # (reuse pos_mlp by giving 1D index to get some variety)
        idx = torch.linspace(-1.0, 1.0, steps=self.tgt_len, device=x.device, dtype=x.dtype).unsqueeze(1)  # (tgt_len, 1)
        tgt_pos = self.pos_mlp(torch.cat([idx, -idx], dim=1))  # (tgt_len, E)
        tgt = tgt + tgt_pos.unsqueeze(1)

        # 6) Run TransformerDecoder: outputs (tgt_len, B, E)
        # No masks provided, full-attention decoder over memory
        decoded = self.decoder(tgt=tgt, memory=memory)  # (tgt_len, B, E)

        # 7) Aggregate decoded sequence back to spatial feature map
        # If tgt_len equals spatial sequence length, we could reshape directly.
        # Instead, we tile/reshape the decoded output to match spatial dims:
        # Compute an intermediate tensor by averaging decoded tokens into patches
        # For simplicity, project decoded tokens to spatial grid by repeating/clipping
        # Create an index mapping from spatial positions to target tokens
        S = Hf * Wf
        # Map each spatial index to a target token index (simple modulo tiling)
        token_indices = (torch.arange(S, device=x.device) % self.tgt_len).long()  # (S,)
        dec_per_pos = decoded[token_indices]  # (S, B, E)
        feat_dec = dec_per_pos.permute(1, 2, 0).reshape(B, E, Hf, Wf)  # (B, E, Hf, Wf)

        # 8) Optional normalization
        # Apply LayerNorm across embedding dimension for each spatial position
        feat_dec = self.norm(feat_dec.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)  # (B, E, Hf, Wf)

        # 9) Non-linearity: Hardshrink applied element-wise to reduce small activations
        feat_shrunk = self.hardshrink(feat_dec)

        # 10) Reconstruct to image channels
        recon = self.conv_recon(feat_shrunk)  # (B, C, Hf, Wf)

        # 11) Crop center region to match original H, W
        # Compute crop offsets
        crop_h = recon.shape[2] - H
        crop_w = recon.shape[3] - W
        if crop_h < 0 or crop_w < 0:
            # if reconstruction is smaller (unexpected), fallback to center pad
            recon = F.pad(recon, [0, max(0, -crop_w), 0, max(0, -crop_h)], mode='constant', value=0)
            crop_h = recon.shape[2] - H
            crop_w = recon.shape[3] - W

        start_h = crop_h // 2
        start_w = crop_w // 2
        recon_cropped = recon[:, :, start_h:start_h + H, start_w:start_w + W]

        return recon_cropped


# Configuration variables at module level
batch_size = 4
in_channels = 3
height = 32
width = 32

def get_inputs():
    """
    Returns:
        A list with a single input tensor representing a batch of images:
        shape (batch_size, in_channels, height, width)
    """
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]

def get_init_inputs():
    """
    Returns arguments to initialize the Model:
      - in_channels, embed_dim, nhead, num_layers, tgt_len, pad_size, shrink_lambda
    """
    return [in_channels, 128, 8, 3, 64, 1, 0.5]