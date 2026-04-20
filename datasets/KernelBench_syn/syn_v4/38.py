import torch
import torch.nn as nn
import torch.nn.functional as F

# Configuration variables
batch_size = 8
num_embeddings = 10000
embedding_dim = 64
bag_size = 20  # number of indices per bag for EmbeddingBag (fixed-size bags here)

conv_in_channels = 8
conv_out_channels = 16
base_width = 32            # width (sequence length) for the 1D conv inputs
conv_kernel_size = 5
conv_stride = 2
conv_padding = 1
conv_output_padding = 1

pad_w = 2  # circular padding applied to the width dimension (left and right)

final_out_dim = 32  # final projection dimension returned by the model


class Model(nn.Module):
    """
    Complex model that:
    - Aggregates token embeddings per example using nn.EmbeddingBag (mode='mean')
    - Projects embedding bags into a 2D feature map, reshapes into (B, C, W)
    - Applies circular 3D padding (with singleton D and H dims) via nn.CircularPad3d
    - Runs a nn.ConvTranspose1d on the padded sequence to expand temporal dimension
    - Pools the conv output, concatenates with the embedding aggregate, and projects to output

    This combines sparse-style operations (EmbeddingBag), padding (CircularPad3d), and
    transposed convolution (ConvTranspose1d) into a single forward pass.
    """
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        conv_in_channels: int,
        conv_out_channels: int,
        base_width: int,
        conv_kernel_size: int,
        conv_stride: int,
        conv_padding: int,
        conv_output_padding: int,
        pad_w: int,
        final_out_dim: int,
    ):
        super(Model, self).__init__()

        # EmbeddingBag aggregates variable-length bags efficiently (we use fixed-size bags here)
        self.emb_bag = nn.EmbeddingBag(num_embeddings, embedding_dim, mode='mean', sparse=False)

        # Project embedding bag to a feature map that will be reshaped into (batch, conv_in_channels, base_width)
        self.proj_to_map = nn.Linear(embedding_dim, conv_in_channels * base_width)

        # Circular padding in 3D; we will use singleton D and H dimensions and only pad W (last dim)
        # Padding format (pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back)
        self.circular_pad3d = nn.CircularPad3d((pad_w, pad_w, 0, 0, 0, 0))

        # ConvTranspose1d to expand/transform the temporal dimension
        self.deconv1d = nn.ConvTranspose1d(
            conv_in_channels,
            conv_out_channels,
            kernel_size=conv_kernel_size,
            stride=conv_stride,
            padding=conv_padding,
            output_padding=conv_output_padding
        )

        # A small normalization and final projection layer
        self.norm = nn.LayerNorm(embedding_dim + conv_out_channels)
        self.final_proj = nn.Linear(embedding_dim + conv_out_channels, final_out_dim)

        # Activation
        self.act = nn.GELU()

        # Save some sizes for forward
        self.base_width = base_width
        self.conv_in_channels = conv_in_channels

    def forward(self, indices: torch.Tensor, offsets: torch.Tensor, conv_input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass that combines embedding aggregation and transposed convolution features.

        Args:
            indices (torch.Tensor): 1D LongTensor containing concatenated indices for all bags.
            offsets (torch.Tensor): 1D LongTensor with starting offsets for each bag (length = batch_size).
            conv_input (torch.Tensor): FloatTensor of shape (batch_size, conv_in_channels, base_width)
                                       representing an auxiliary sequence input per batch.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, final_out_dim)
        """
        # 1) Aggregate embeddings per bag -> (batch_size, embedding_dim)
        emb_agg = self.emb_bag(indices, offsets)  # mean over each bag

        # 2) Project aggregated embeddings into a feature map and reshape -> (B, C_in, W)
        proj = self.proj_to_map(emb_agg)  # (B, C_in * base_width)
        feat_map = proj.view(-1, self.conv_in_channels, self.base_width)  # (B, C_in, W)

        # 3) Prepare conv_input: accept (B, C_in, W) and add singleton D,H dims to make 5D for CircularPad3d
        #    then apply circular padding only along the width dimension, and squeeze back to 3D.
        conv_input_5d = conv_input.unsqueeze(2).unsqueeze(3)  # (B, C_in, 1, 1, W)
        padded = self.circular_pad3d(conv_input_5d)           # Circular pad across W
        padded_3d = padded.squeeze(3).squeeze(2)             # (B, C_in, W_padded)

        # 4) Combine the proj-derived feature map and the padded conv input by element-wise addition.
        #    We need to align widths: feat_map has width base_width, padded_3d has base_width + 2*pad_w.
        #    We'll center-crop the padded tensor to base_width for alignment (robust and deterministic).
        padded_width = padded_3d.size(2)
        start = (padded_width - self.base_width) // 2
        if start < 0:
            # If padded width is smaller (shouldn't happen), pad to the right
            padded_3d = F.pad(padded_3d, (0, self.base_width - padded_width))
            start = 0
        cropped = padded_3d[:, :, start:start + self.base_width]  # (B, C_in, base_width)

        combined_map = feat_map + cropped  # (B, C_in, base_width)
        combined_map = self.act(combined_map)

        # 5) Run transposed convolution to transform/expand temporal dimension
        conv_out = self.deconv1d(combined_map)  # (B, conv_out_channels, W_out)

        # 6) Global pooling over width to get a fixed-size vector per batch
        pooled = conv_out.mean(dim=2)  # (B, conv_out_channels)

        # 7) Concatenate pooled conv features with embedding aggregate and project
        concat = torch.cat([emb_agg, pooled], dim=1)  # (B, embedding_dim + conv_out_channels)
        normalized = self.norm(concat)
        activated = self.act(normalized)
        out = self.final_proj(activated)  # (B, final_out_dim)

        return out


def get_inputs():
    """
    Returns a list of input tensors:
    - indices: 1D LongTensor with concatenated indices for EmbeddingBag (length batch_size * bag_size)
    - offsets: 1D LongTensor of length batch_size with starting offsets for each bag
    - conv_input: FloatTensor of shape (batch_size, conv_in_channels, base_width)
    """
    # Create indices and offsets for EmbeddingBag: fixed-size bags for simplicity
    total_indices = batch_size * bag_size
    indices = torch.randint(low=0, high=num_embeddings, size=(total_indices,), dtype=torch.long)
    offsets = torch.arange(0, total_indices, bag_size, dtype=torch.long)  # length = batch_size

    # Auxiliary conv input
    conv_input = torch.randn(batch_size, conv_in_channels, base_width)

    return [indices, offsets, conv_input]


def get_init_inputs():
    """
    Returns initialization parameters used to construct the Model instance.
    """
    return [
        num_embeddings,
        embedding_dim,
        conv_in_channels,
        conv_out_channels,
        base_width,
        conv_kernel_size,
        conv_stride,
        conv_padding,
        conv_output_padding,
        pad_w,
        final_out_dim,
    ]