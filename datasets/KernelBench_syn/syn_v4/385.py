import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Complex model that:
    - Uses an EmbeddingBag to pool variable-length token bags into per-batch embeddings.
    - Applies Hardshrink nonlinearity to sparsify those embeddings.
    - Projects the sparse embeddings into a small spatial feature map via a Linear layer and reshaping.
    - Upsamples the small feature map with UpsamplingNearest2d to match a provided spatial input tensor.
    - Fuses the upsampled embedding map with the spatial input and refines via a convolution.
    """
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        proj_channels: int,
        spatial_channels: int,
        small_h: int,
        small_w: int,
        out_h: int,
        out_w: int,
        shrink_lambda: float = 0.5
    ):
        """
        Initializes layers and parameters.

        Args:
            num_embeddings (int): Size of embedding dictionary.
            embedding_dim (int): Dimensionality of embeddings produced by EmbeddingBag.
            proj_channels (int): Number of channels produced from the embedding projection (before upsampling).
            spatial_channels (int): Number of channels in the spatial input to be fused.
            small_h (int), small_w (int): Spatial size for the projected embedding map.
            out_h (int), out_w (int): Desired output spatial size (must be integer multiples of small_h/small_w).
            shrink_lambda (float): Lambda parameter for Hardshrink.
        """
        super(Model, self).__init__()
        self.embedding_dim = embedding_dim
        self.proj_channels = proj_channels
        self.small_h = small_h
        self.small_w = small_w
        self.out_h = out_h
        self.out_w = out_w

        # EmbeddingBag pools variable-length bags into fixed-size vectors (mode='mean')
        self.emb_bag = nn.EmbeddingBag(num_embeddings=num_embeddings, embedding_dim=embedding_dim, mode='mean', include_last_offset=False)

        # Hardshrink to sparsify embeddings
        self.hardshrink = nn.Hardshrink(lambd=shrink_lambda)

        # Linear projection from embedding vector -> flattened small spatial map
        self.fc = nn.Linear(embedding_dim, proj_channels * small_h * small_w)

        # Nearest neighbor upsampling to match spatial input resolution
        scale_h = out_h // small_h
        scale_w = out_w // small_w
        if (out_h % small_h) != 0 or (out_w % small_w) != 0:
            raise ValueError("out_h/out_w must be integer multiples of small_h/small_w")
        self.upsample = nn.UpsamplingNearest2d(scale_factor=(scale_h, scale_w))

        # Fusion convolution: combine upsampled projected embeddings with the spatial input
        fused_channels = proj_channels + spatial_channels
        self.fuse_conv = nn.Conv2d(in_channels=fused_channels, out_channels=spatial_channels, kernel_size=3, padding=1, bias=True)

        # A small refinement convolution for additional mixing
        self.refine_conv = nn.Conv2d(in_channels=spatial_channels, out_channels=spatial_channels, kernel_size=1, bias=True)

    def forward(self, indices: torch.Tensor, offsets: torch.Tensor, spatial_input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            indices (torch.Tensor): 1D tensor of concatenated token indices for all bags (dtype=torch.long).
            offsets (torch.Tensor): 1D tensor of starting offsets for each bag (dtype=torch.long), length == batch_size.
            spatial_input (torch.Tensor): Spatial tensor of shape (batch_size, spatial_channels, out_h, out_w).

        Returns:
            torch.Tensor: Refined fused tensor of shape (batch_size, spatial_channels, out_h, out_w).
        """
        # 1) Pool variable-length token bags into per-batch embeddings
        bag_emb = self.emb_bag(indices, offsets)            # (batch_size, embedding_dim)

        # 2) Sparsify embeddings
        bag_emb = self.hardshrink(bag_emb)                  # (batch_size, embedding_dim)

        # 3) Project sparse embeddings into a small spatial map
        proj = self.fc(bag_emb)                             # (batch_size, proj_channels * small_h * small_w)
        batch_size = proj.shape[0]
        proj_map = proj.view(batch_size, self.proj_channels, self.small_h, self.small_w)  # (batch, proj_ch, small_h, small_w)

        # 4) Upsample projected map to match spatial input resolution
        up_proj = self.upsample(proj_map)                   # (batch, proj_ch, out_h, out_w)

        # 5) Fuse with provided spatial input (concatenate along channels)
        fused = torch.cat([up_proj, spatial_input], dim=1)  # (batch, proj_ch + spatial_channels, out_h, out_w)

        # 6) Refine fused features with convolution and activation
        out = self.fuse_conv(fused)                         # (batch, spatial_channels, out_h, out_w)
        out = F.relu(out, inplace=True)
        out = self.refine_conv(out)                         # (batch, spatial_channels, out_h, out_w)

        return out


# Configuration variables
BATCH_SIZE = 8
NUM_EMBEDDINGS = 10000
EMBEDDING_DIM = 128
PROJ_CHANNELS = 32
SPATIAL_CHANNELS = 16
SMALL_H = 4
SMALL_W = 4
OUT_H = 16
OUT_W = 16
MAX_BAG_LEN = 20  # maximum tokens per bag for random input generation

def get_inputs():
    """
    Generates:
    - indices (1D LongTensor): concatenated token indices for all bags
    - offsets (1D LongTensor): start offsets for each bag (length == BATCH_SIZE)
    - spatial_input (FloatTensor): spatial feature map of shape (BATCH_SIZE, SPATIAL_CHANNELS, OUT_H, OUT_W)

    Returns:
        list: [indices, offsets, spatial_input]
    """
    # Random variable lengths for each bag between 1 and MAX_BAG_LEN
    lengths = torch.randint(1, MAX_BAG_LEN + 1, (BATCH_SIZE,), dtype=torch.long)
    offsets = torch.empty(BATCH_SIZE, dtype=torch.long)
    offsets[0] = 0
    offsets[1:] = torch.cumsum(lengths, dim=0)[:-1]
    total_indices = int(lengths.sum().item())

    # Random token indices in range [0, NUM_EMBEDDINGS)
    indices = torch.randint(low=0, high=NUM_EMBEDDINGS, size=(total_indices,), dtype=torch.long)

    # Spatial input to fuse with (batch, channels, H, W)
    spatial_input = torch.randn(BATCH_SIZE, SPATIAL_CHANNELS, OUT_H, OUT_W)

    return [indices, offsets, spatial_input]

def get_init_inputs():
    """
    Returns initialization arguments for Model in order:
    [num_embeddings, embedding_dim, proj_channels, spatial_channels, small_h, small_w, out_h, out_w]

    Returns:
        list: initialization parameters
    """
    return [NUM_EMBEDDINGS, EMBEDDING_DIM, PROJ_CHANNELS, SPATIAL_CHANNELS, SMALL_H, SMALL_W, OUT_H, OUT_W]