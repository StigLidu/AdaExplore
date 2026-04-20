import math
import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Complex model that combines 1D Lp pooling, Conv1d projections, a TransformerDecoderLayer,
    and 2D MaxPooling. The flow is:
      1) LPPool1d over the input sequence to reduce temporal resolution.
      2) Two 1x1 Conv1d projections to produce 'tgt' and 'memory' embeddings for the transformer decoder.
      3) TransformerDecoderLayer to allow cross-attention between pooled target sequence and full-resolution memory.
      4) Reshape decoder output into a 2D spatial layout and apply MaxPool2d.
      5) Flatten to produce a final vector per batch element.
    """
    def __init__(
        self,
        in_channels: int,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        lp_norm: float,
        lp_kernel: int,
        maxpool_kernel: int = 2,
        dropout: float = 0.1,
    ):
        """
        Initializes the composite model.

        Args:
            in_channels (int): Number of input channels (C) of the sequence tensor.
            d_model (int): Embedding dimension for transformer (must be divisible by nhead).
            nhead (int): Number of attention heads.
            dim_feedforward (int): Feedforward dimension inside the transformer layer.
            lp_norm (float): The 'p' norm used by LPPool1d.
            lp_kernel (int): Kernel size (and stride) for LPPool1d to reduce sequence length.
            maxpool_kernel (int): Kernel size for MaxPool2d after reshaping to 2D.
            dropout (float): Dropout probability used inside the TransformerDecoderLayer.
        """
        super(Model, self).__init__()

        # 1D Lp pooling to downsample the temporal dimension
        self.lp_pool = nn.LPPool1d(norm_type=lp_norm, kernel_size=lp_kernel, stride=lp_kernel)

        # Project input channels into d_model for 'tgt' (from pooled sequence)
        # and for 'memory' (from full-resolution sequence).
        # Using 1x1 Conv1d (kernel_size=1) for channel projection while preserving lengths.
        self.tgt_proj = nn.Conv1d(in_channels=in_channels, out_channels=d_model, kernel_size=1)
        self.mem_proj = nn.Conv1d(in_channels=in_channels, out_channels=d_model, kernel_size=1)

        # Transformer decoder layer with cross-attention (self-attn, multihead-attn, FFN)
        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='relu',
            batch_first=False  # expects (T, N, E)
        )

        # 2D max pooling applied after reshaping transformer output into (N, C, H, W)
        self.maxpool2d = nn.MaxPool2d(kernel_size=maxpool_kernel)

        # small safeguard linear to produce a stable final vector size (optional)
        # We'll leave final flattening to forward; this layer is not strictly necessary.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input of shape (batch_size, in_channels, seq_len).

        Returns:
            torch.Tensor: Flattened output tensor of shape (batch_size, D) after pooling.
        """
        # x: (N, C, L)
        N, C, L = x.shape

        # 1) Downsample temporal resolution using LPPool1d
        # pooled: (N, C, Lp) where Lp = floor((L - kernel)/stride + 1) usually L/stride for exact divisibility.
        pooled = self.lp_pool(x)

        # 2) Project pooled sequence to d_model to form decoder 'tgt'
        # tgt_proj output: (N, d_model, Lp)
        tgt_proj = self.tgt_proj(pooled)

        # 3) Project full-resolution sequence to d_model to form 'memory' for cross-attention
        # mem_proj output: (N, d_model, L)
        mem_proj = self.mem_proj(x)

        # 4) TransformerDecoderLayer expects shapes (T, N, E) for both tgt and memory.
        # Permute accordingly.
        # tgt: (Lp, N, d_model)
        tgt = tgt_proj.permute(2, 0, 1)
        # memory: (L, N, d_model)
        memory = mem_proj.permute(2, 0, 1)

        # 5) Apply TransformerDecoderLayer (self-attn on tgt then cross-attn with memory + FFN)
        decoded = self.decoder_layer(tgt=tgt, memory=memory)  # (Lp, N, d_model)

        # 6) Permute back to (N, d_model, Lp) for 2D reshaping
        decoded = decoded.permute(1, 2, 0)  # (N, d_model, Lp)
        _, E, Lp = decoded.shape

        # Ensure we can reshape Lp to a near-square 2D grid. We expect Lp to be a perfect square
        # given the module-level configuration; but compute dims robustly:
        H = int(math.isqrt(Lp))
        if H * H != Lp:
            # If not a perfect square, choose H as floor(sqrt(Lp)) and W accordingly
            H = int(math.floor(math.sqrt(Lp)))
            W = Lp // H
            # If still not exact, pad the last dimension (simple zero-padding) to reach H*W
            if H * W < Lp:
                pad_len = (H * (W + 1)) - Lp
                # pad on the right in the sequence dimension
                decoded = torch.nn.functional.pad(decoded, (0, pad_len))
                Lp = decoded.shape[2]
                H = int(math.isqrt(Lp))
                W = Lp // H
        else:
            W = H

        # Now reshape into (N, E, H, W)
        decoded_2d = decoded.view(N, E, H, W)

        # 7) Apply 2D MaxPool
        pooled_2d = self.maxpool2d(decoded_2d)  # (N, E, H', W')

        # 8) Flatten spatial dimensions to produce final vector per sample
        out = pooled_2d.flatten(start_dim=1)  # (N, E * H' * W')

        return out


# Module-level configuration variables
batch_size = 16
in_channels = 32
seq_len = 256  # choose 256 so that after lp_kernel=4 we get 64 (perfect square 8x8)
d_model = 64   # must be divisible by nhead
nhead = 8
dim_feedforward = 256
lp_norm_power = 2.0
lp_kernel_size = 4  # stride will be same as kernel_size to reduce length by factor 4 -> 256->64
maxpool_kernel = 2

def get_inputs():
    """
    Generates a sample input tensor for the Model.

    Returns:
        List[torch.Tensor]: [x] where x has shape (batch_size, in_channels, seq_len)
    """
    x = torch.randn(batch_size, in_channels, seq_len)
    return [x]

def get_init_inputs():
    """
    Returns the initialization parameters required to construct the Model.
    Order matches Model.__init__ signature.

    Returns:
        List: [in_channels, d_model, nhead, dim_feedforward, lp_norm, lp_kernel, maxpool_kernel]
    """
    return [in_channels, d_model, nhead, dim_feedforward, lp_norm_power, lp_kernel_size, maxpool_kernel]