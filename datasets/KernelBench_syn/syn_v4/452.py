import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Complex reconstruction model that:
    - Extracts non-overlapping image patches using nn.Unfold
    - Projects patches into an embedding space
    - Uses a stack of TransformerDecoder layers (nn.TransformerDecoder)
      to process patch embeddings conditioned on an external memory sequence
    - Reconstructs image patches and folds them back to the image space

    The model demonstrates combining convolution-like patch extraction with
    transformer decoder layers to perform conditional patch-wise processing.
    """
    def __init__(
        self,
        in_channels: int,
        height: int,
        width: int,
        patch_size: int,
        embed_dim: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        memory_len: int,
    ):
        """
        Args:
            in_channels (int): Number of image channels (e.g., 3 for RGB).
            height (int): Image height (must be divisible by patch_size).
            width (int): Image width (must be divisible by patch_size).
            patch_size (int): Patch size (patches are patch_size x patch_size).
            embed_dim (int): Dimension of transformer embeddings.
            nhead (int): Number of attention heads in the transformer.
            num_layers (int): Number of TransformerDecoderLayer layers.
            dim_feedforward (int): Feedforward hidden dimension inside the transformer layers.
            memory_len (int): Length of the memory (sequence length) provided to the decoder.
        """
        super(Model, self).__init__()

        assert height % patch_size == 0 and width % patch_size == 0, "Height and width must be divisible by patch_size"

        self.in_channels = in_channels
        self.height = height
        self.width = width
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.memory_len = memory_len

        # Number of patches per image
        self.num_patches_h = height // patch_size
        self.num_patches_w = width // patch_size
        self.num_patches = self.num_patches_h * self.num_patches_w

        # Unfold to extract non-overlapping patches
        self.unfold = nn.Unfold(kernel_size=patch_size, stride=patch_size)

        # Patch flat dimension (C * patch_size * patch_size)
        self.patch_dim = in_channels * patch_size * patch_size

        # Linear projections to/from embedding space
        self.patch_proj = nn.Linear(self.patch_dim, embed_dim, bias=True)
        self.patch_reconstruct = nn.Linear(embed_dim, self.patch_dim, bias=True)

        # Positional embeddings for patch sequence and memory sequence
        self.patch_pos = nn.Parameter(torch.randn(self.num_patches, embed_dim))
        self.mem_pos = nn.Parameter(torch.randn(memory_len, embed_dim))

        # Transformer decoder (stack of decoder layers)
        decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=nhead, dim_feedforward=dim_feedforward)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Final normalization on decoder outputs
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input image tensor of shape (B, C, H, W)
            memory (torch.Tensor): Memory sequence of shape (S, B, embed_dim), where S == memory_len

        Returns:
            torch.Tensor: Reconstructed image of shape (B, C, H, W)
        """
        B, C, H, W = x.shape
        assert C == self.in_channels and H == self.height and W == self.width, "Input shape mismatch"
        S, B_mem, E = memory.shape
        assert B_mem == B and S == self.memory_len and E == self.embed_dim, "Memory shape mismatch"

        # 1) Extract patches: -> (B, patch_dim, L)
        patches_flat = self.unfold(x)  # shape: (B, patch_dim, L)
        L = patches_flat.shape[-1]  # should equal self.num_patches

        # 2) Convert to (B, L, patch_dim) and project to embedding space -> (B, L, E)
        patches = patches_flat.transpose(1, 2)  # (B, L, patch_dim)
        patches_emb = self.patch_proj(patches)  # (B, L, E)

        # 3) Add positional embeddings (broadcast over batch)
        patches_emb = patches_emb + self.patch_pos.unsqueeze(0)  # (B, L, E)

        # 4) Prepare for transformer (seq_len, batch, embed_dim)
        tgt = patches_emb.transpose(0, 1)  # (L, B, E)
        mem = memory + self.mem_pos.unsqueeze(1)  # (S, B, E)

        # 5) Transformer decoder: process target conditioned on memory -> (L, B, E)
        decoded = self.decoder(tgt=tgt, memory=mem)  # (L, B, E)

        # 6) Normalize and add residual connection to preserve information
        decoded = self.norm(decoded + tgt)  # (L, B, E)

        # 7) Project back to patch space: (B, L, patch_dim)
        decoded_batched = decoded.transpose(0, 1)  # (B, L, E)
        reconstructed_patches = self.patch_reconstruct(decoded_batched)  # (B, L, patch_dim)

        # 8) Fold back to image: need shape (B, patch_dim, L)
        reconstructed_patches = reconstructed_patches.transpose(1, 2)  # (B, patch_dim, L)
        output = F.fold(
            reconstructed_patches,
            output_size=(self.height, self.width),
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )  # (B, C, H, W)

        return output

# Configuration variables
batch_size = 8
in_channels = 3
height = 64
width = 64
patch_size = 8
embed_dim = 256
nhead = 8
num_layers = 4
dim_feedforward = 512
memory_len = 16  # length of the conditioning memory sequence

def get_inputs():
    """
    Returns a list containing:
    - x: random image batch tensor of shape (B, C, H, W)
    - memory: random memory tensor of shape (S, B, embed_dim)
    """
    x = torch.randn(batch_size, in_channels, height, width)
    memory = torch.randn(memory_len, batch_size, embed_dim)
    return [x, memory]

def get_init_inputs():
    """
    Returns the initialization inputs for Model.__init__ in the exact order:
    (in_channels, height, width, patch_size, embed_dim, nhead, num_layers, dim_feedforward, memory_len)
    """
    return [in_channels, height, width, patch_size, embed_dim, nhead, num_layers, dim_feedforward, memory_len]