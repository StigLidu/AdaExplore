import torch
import torch.nn as nn
from typing import List, Any

class Model(nn.Module):
    """
    Complex model that combines a lazy ConvTranspose3d upsampling of a volumetric input,
    a FractionalMaxPool2d on an image input, and a TransformerDecoder that attends
    from the pooled image tokens (as target) to the upsampled volumetric features (as memory).

    The computation pattern:
      1. Upsample a 5D volumetric tensor with LazyConvTranspose3d (lazy in_channels).
      2. Flatten spatial dims of the upsampled volume and project to a transformer memory (seq_len_mem, batch, embed_dim).
      3. Fractionally max-pool a 2D image, flatten spatial dims and project to transformer target embeddings (seq_len_tgt, batch, embed_dim).
      4. Run a TransformerDecoder with the target and memory.
      5. Pool decoded target outputs across sequence and apply a final projection.

    Inputs:
      - vol: Tensor of shape (batch, in_channels, D, H, W)
      - img: Tensor of shape (batch, img_channels, H_img, W_img)

    Returns:
      - Tensor of shape (batch, embed_dim)
    """
    def __init__(
        self,
        deconv_out_channels: int,
        deconv_kernel_size: int,
        deconv_stride: int,
        deconv_padding: int,
        frac_output_ratio_h: float,
        frac_output_ratio_w: float,
        embed_dim: int,
        nhead: int,
        num_decoder_layers: int
    ):
        super(Model, self).__init__()

        # Lazy ConvTranspose3d: in_channels is inferred at first forward call.
        # This upsamples the volumetric input to produce rich spatial memory features.
        self.deconv = nn.LazyConvTranspose3d(
            out_channels=deconv_out_channels,
            kernel_size=deconv_kernel_size,
            stride=deconv_stride,
            padding=deconv_padding
        )

        # FractionalMaxPool2d: we use a fixed small kernel but allow the output ratio
        # to control the pooled spatial size relative to the input image.
        self.frac_pool = nn.FractionalMaxPool2d(
            kernel_size=(2, 2),
            output_ratio=(frac_output_ratio_h, frac_output_ratio_w)
        )

        # Linear projections to map channel dims into the transformer embedding space.
        # memory_proj maps volumetric channels -> embed_dim
        self.memory_proj = nn.Linear(deconv_out_channels, embed_dim)
        # tgt_proj maps image channels -> embed_dim
        # Image channels are known at runtime from the input; we assume typical channels (e.g., 3).
        # To keep __init__ consistent we set tgt_proj to expect 3 channels by design of get_inputs.
        self.tgt_proj = nn.Linear(3, embed_dim)

        # Transformer decoder stack: decoder layers followed by the decoder module.
        decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=nhead)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        # Final projection after decoding and pooling across target sequence.
        self.final_proj = nn.Linear(embed_dim, embed_dim)

        # Small normalization and dropout for stability (optional, but common pattern)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, vol: torch.Tensor, img: torch.Tensor) -> torch.Tensor:
        """
        Forward pass combining volumetric upsampling, image pooling, and transformer decoding.

        Args:
            vol (torch.Tensor): Volumetric input of shape (batch, in_channels, D, H, W).
                                in_channels is inferred by LazyConvTranspose3d on first call.
            img (torch.Tensor): Image input of shape (batch, img_channels, H_img, W_img).

        Returns:
            torch.Tensor: Output tensor of shape (batch, embed_dim).
        """
        # 1) Upsample volumetric input with lazy ConvTranspose3d
        # Output shape: (batch, deconv_out_channels, D2, H2, W2)
        vol_up = self.deconv(vol)

        # 2) Prepare transformer memory from volumetric features
        # Flatten spatial dims and permute to (seq_len_mem, batch, channels)
        batch = vol_up.shape[0]
        vol_flat = vol_up.flatten(2).permute(2, 0, 1)  # (N_mem, batch, channels)
        # Project channel dimension into embed_dim -> (N_mem, batch, embed_dim)
        memory = self.memory_proj(vol_flat)

        # 3) Fractional max pool on image input
        # FractionalMaxPool2d returns (output, indices) tuple; take the output tensor
        pooled_out = self.frac_pool(img)
        if isinstance(pooled_out, tuple) or isinstance(pooled_out, list):
            pooled = pooled_out[0]
        else:
            pooled = pooled_out  # defensive

        # pooled shape: (batch, img_channels, Hp, Wp)
        # Flatten pooled spatial dims and permute to (seq_len_tgt, batch, img_channels)
        tgt_flat = pooled.flatten(2).permute(2, 0, 1)  # (N_tgt, batch, img_channels)
        # Project image channels into embedding dimension
        tgt = self.tgt_proj(tgt_flat)  # (N_tgt, batch, embed_dim)

        # 4) Transformer decoder: decode target sequence using volumetric memory
        decoded = self.decoder(tgt=tgt, memory=memory)  # (N_tgt, batch, embed_dim)

        # 5) Aggregate decoded outputs (mean pooling across target sequence),
        # apply normalization, dropout and final projection.
        decoded_mean = decoded.mean(dim=0)  # (batch, embed_dim)
        decoded_norm = self.norm(decoded_mean)
        decoded_drop = self.dropout(decoded_norm)
        out = self.final_proj(decoded_drop)  # (batch, embed_dim)

        return out


# Configuration variables at module level
batch_size = 4

# Volumetric input configuration (in_channels will be inferred)
vol_in_channels = 3
vol_depth = 8
vol_height = 8
vol_width = 8

# ConvTranspose3d parameters (passed to Model.__init__)
deconv_out_channels = 16
deconv_kernel_size = 3
deconv_stride = 2
deconv_padding = 1

# Image input configuration
img_channels = 3
img_height = 64
img_width = 64

# Fractional pooling output ratio (fraction of input dims)
frac_output_ratio_h = 0.5
frac_output_ratio_w = 0.5

# Transformer configuration
embed_dim = 64
nhead = 8
num_decoder_layers = 2

def get_inputs() -> List[torch.Tensor]:
    """
    Returns a list of input tensors for the Model's forward method:
      - vol: (batch_size, vol_in_channels, vol_depth, vol_height, vol_width)
      - img: (batch_size, img_channels, img_height, img_width)
    """
    vol = torch.randn(batch_size, vol_in_channels, vol_depth, vol_height, vol_width)
    img = torch.randn(batch_size, img_channels, img_height, img_width)
    return [vol, img]

def get_init_inputs() -> List[Any]:
    """
    Returns the list of initialization arguments for constructing the Model instance.
    The ordering matches Model.__init__ signature.
    """
    return [
        deconv_out_channels,
        deconv_kernel_size,
        deconv_stride,
        deconv_padding,
        frac_output_ratio_h,
        frac_output_ratio_w,
        embed_dim,
        nhead,
        num_decoder_layers
    ]