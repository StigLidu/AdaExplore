import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Complex model that:
    - Extracts 2D patches via Unfold
    - Projects patches into embeddings
    - Uses a TransformerDecoder with a learned/global memory token
    - Projects back to patches and reconstructs the image via Fold
    - Applies a 3D BatchNorm (lazy) on a synthetic depth dimension
    """
    def __init__(
        self,
        in_channels: int,
        height: int,
        width: int,
        kernel_size: int,
        stride: int,
        padding: int,
        embed_dim: int,
        nhead: int,
        num_decoder_layers: int,
    ):
        """
        Initializes the composite module.

        Args:
            in_channels (int): Number of input channels.
            height (int): Spatial height of the input images.
            width (int): Spatial width of the input images.
            kernel_size (int): Patch kernel size used for Unfold/Fold.
            stride (int): Stride for the patch extraction.
            padding (int): Padding applied before extracting patches.
            embed_dim (int): Embedding dimension used by TransformerDecoder.
            nhead (int): Number of attention heads for the decoder layers.
            num_decoder_layers (int): Number of decoder layers in TransformerDecoder.
        """
        super(Model, self).__init__()
        self.in_channels = in_channels
        self.height = height
        self.width = width
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.embed_dim = embed_dim

        # Extract patches from the image: shape -> (N, C*ks*ks, L)
        self.unfold = nn.Unfold(kernel_size=kernel_size, padding=padding, stride=stride)

        # Reconstruct the image from patches: need to know output_size
        self.fold = nn.Fold(output_size=(height, width), kernel_size=kernel_size, padding=0, stride=stride)

        # Dimension of each raw patch vector
        self.patch_dim = in_channels * kernel_size * kernel_size

        # Linear projections: patch -> embed, embed -> patch
        self.input_proj = nn.Linear(self.patch_dim, embed_dim)
        self.output_proj = nn.Linear(embed_dim, self.patch_dim)

        # Transformer decoder: a stack of decoder layers
        decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=nhead, dim_feedforward=embed_dim * 4, activation='gelu')
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        # A small learned projection to create a memory token per batch (S=1)
        self.memory_proj = nn.Linear(self.patch_dim, embed_dim)

        # LazyBatchNorm3d will initialize based on the incoming channel count at first forward pass
        self.bn3d = nn.LazyBatchNorm3d()

        # A lightweight residual projection if needed (keeps channel dims stable)
        # This will be identity-like (1x1 conv) to mix channels after fold
        self.res_proj = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass combining unfold -> transformer decoder -> fold -> 3D batchnorm.

        Args:
            x (torch.Tensor): Input tensor of shape (N, C, H, W)

        Returns:
            torch.Tensor: Output tensor of shape (N, C, 1, H, W) after LazyBatchNorm3d
        """
        # 1) Extract patches: (N, C*ks*ks, L)
        patches = self.unfold(x)  # (N, patch_dim, L)

        # 2) Prepare sequence for Transformer: (L, N, patch_dim)
        patches_seq = patches.permute(2, 0, 1)  # (L, N, patch_dim)

        # 3) Project patches into embedding space: (L, N, embed_dim)
        embedded = self.input_proj(patches_seq)  # (L, N, embed_dim)

        # 4) Create a memory token from global pooled patch features: (1, N, embed_dim)
        #    Use mean pooling over patches per batch as a simple "encoder" summary.
        global_feat = patches_seq.mean(dim=0)  # (N, patch_dim)
        memory = self.memory_proj(global_feat).unsqueeze(0)  # (1, N, embed_dim)

        # 5) Run through TransformerDecoder: target=embedded (L,N,E), memory=(1,N,E)
        decoded = self.transformer_decoder(tgt=embedded, memory=memory)  # (L, N, embed_dim)

        # 6) Project back to patch vectors: (L, N, patch_dim)
        out_patches_seq = self.output_proj(decoded)  # (L, N, patch_dim)

        # 7) Rearrange to (N, patch_dim, L) for folding
        out_patches = out_patches_seq.permute(1, 2, 0)  # (N, patch_dim, L)

        # 8) Fold back into (N, C, H, W)
        reconstructed = self.fold(out_patches)  # (N, C, H, W)

        # 9) Residual mixing and non-linearity
        mixed = self.res_proj(reconstructed) + x  # (N, C, H, W)
        activated = F.gelu(mixed)

        # 10) Add a synthetic depth dimension to apply 3D BatchNorm
        activated_5d = activated.unsqueeze(2)  # (N, C, 1, H, W)

        # 11) Apply LazyBatchNorm3d (will be initialized on first forward)
        normalized = self.bn3d(activated_5d)  # (N, C, 1, H, W)

        return normalized


# Configuration variables (module level)
batch_size = 8
in_channels = 3
height = 64
width = 64
kernel_size = 8
stride = 8
padding = 0

embed_dim = 128
nhead = 8
num_decoder_layers = 2

def get_inputs():
    """
    Returns a list of input tensors to the model.
    Single input: a batch of images (N, C, H, W)
    """
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]

def get_init_inputs():
    """
    Returns the initialization parameters for Model.__init__ in order.
    """
    return [in_channels, height, width, kernel_size, stride, padding, embed_dim, nhead, num_decoder_layers]