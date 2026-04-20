import torch
import torch.nn as nn

# Module-level configuration
BATCH_SIZE = 8
TGT_SEQ_LEN = 64
MEM_SEQ_LEN = 32
D_MODEL = 64
NHEAD = 8
NUM_LAYERS = 2
OUT_CHANNELS = 16
DEPTH = 8
HEIGHT = 8
WIDTH = 4
POOL_KERNEL = (2, 2, 2)


class Model(nn.Module):
    """
    A composite model that:
    - Runs a TransformerDecoder over a target sequence with encoder 'memory'.
    - Pools the decoder outputs over the time dimension (mean).
    - Projects the pooled embedding into a volumetric tensor.
    - Applies a Tanh non-linearity and a 3D MaxPool to produce a downsampled volume.
    """

    def __init__(
        self,
        d_model: int = D_MODEL,
        nhead: int = NHEAD,
        num_layers: int = NUM_LAYERS,
        out_channels: int = OUT_CHANNELS,
        depth: int = DEPTH,
        height: int = HEIGHT,
        width: int = WIDTH,
        pool_kernel=(2, 2, 2),
    ):
        """
        Initializes TransformerDecoder, a linear projector to volumetric tensor,
        a Tanh activation and a MaxPool3d layer.

        Args:
            d_model (int): Embedding dimension for transformer.
            nhead (int): Number of attention heads.
            num_layers (int): Number of decoder layers.
            out_channels (int): Number of channels for the volumetric output.
            depth (int): Depth dimension of the volumetric output.
            height (int): Height dimension of the volumetric output.
            width (int): Width dimension of the volumetric output.
            pool_kernel (tuple): Kernel size for MaxPool3d.
        """
        super(Model, self).__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.out_channels = out_channels
        self.depth = depth
        self.height = height
        self.width = width

        # Transformer decoder stack
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Project pooled embedding into a volumetric tensor
        self.volume_size = out_channels * depth * height * width
        self.projector = nn.Linear(d_model, self.volume_size)

        # Non-linear activation and 3D pooling
        self.tanh = nn.Tanh()
        self.maxpool3d = nn.MaxPool3d(kernel_size=pool_kernel, stride=pool_kernel)

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tgt (torch.Tensor): Target sequence embeddings of shape (T_tgt, B, d_model).
            memory (torch.Tensor): Memory/encoder output of shape (T_mem, B, d_model).

        Returns:
            torch.Tensor: Downsampled volumetric tensor after activation and pooling.
                          Shape will be (B, out_channels, D', H', W') where
                          D' = depth / pool_kernel[0], etc.
        """
        # 1) Run the transformer decoder: output shape (T_tgt, B, d_model)
        decoder_out = self.decoder(tgt, memory)

        # 2) Pool across the time dimension (mean over sequence length): (B, d_model)
        pooled = decoder_out.mean(dim=0)

        # 3) Project into a volumetric flattened vector: (B, out_channels * depth * height * width)
        projected = self.projector(pooled)

        # 4) Reshape to volumetric form: (B, out_channels, depth, height, width)
        volume = projected.view(
            projected.size(0),
            self.out_channels,
            self.depth,
            self.height,
            self.width,
        )

        # 5) Apply non-linearity
        activated = self.tanh(volume)

        # 6) Apply 3D max pooling to downsample spatial dimensions
        pooled3d = self.maxpool3d(activated)

        return pooled3d


def get_inputs():
    """
    Returns a list of input tensors for the model's forward method:
    [tgt, memory]
    - tgt shape: (TGT_SEQ_LEN, BATCH_SIZE, D_MODEL)
    - memory shape: (MEM_SEQ_LEN, BATCH_SIZE, D_MODEL)
    """
    tgt = torch.randn(TGT_SEQ_LEN, BATCH_SIZE, D_MODEL)
    memory = torch.randn(MEM_SEQ_LEN, BATCH_SIZE, D_MODEL)
    return [tgt, memory]


def get_init_inputs():
    """
    Returns initialization parameters for Model:
    [d_model, nhead, num_layers, out_channels, depth, height, width]
    """
    return [D_MODEL, NHEAD, NUM_LAYERS, OUT_CHANNELS, DEPTH, HEIGHT, WIDTH]