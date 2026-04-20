import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Complex model that extracts sliding patches from a 2D image, applies per-instance
    normalization on patch vectors, processes them with 1D convolutions, computes
    attention over spatial patches, and reconstructs an output feature map by folding
    the refined patches back. Uses nn.Unfold, nn.InstanceNorm1d and nn.SyncBatchNorm.
    """
    def __init__(
        self,
        kernel_size: int,
        stride: int,
        padding: int,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        img_h: int,
        img_w: int,
    ):
        """
        Initializes the model components.

        Args:
            kernel_size (int): Size of the sliding window (square).
            stride (int): Stride for unfolding.
            padding (int): Padding for unfolding.
            in_channels (int): Number of input image channels.
            hidden_channels (int): Hidden channel dimension after projecting patch vectors.
            out_channels (int): Desired number of output channels after folding.
            img_h (int): Height of input images (used to configure Fold).
            img_w (int): Width of input images (used to configure Fold).
        """
        super(Model, self).__init__()

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.img_h = img_h
        self.img_w = img_w

        # Unfold to extract sliding local blocks (patch vectors)
        self.unfold = nn.Unfold(kernel_size=kernel_size, stride=stride, padding=padding)

        # Each patch vector dimension
        self.patch_dim = in_channels * kernel_size * kernel_size

        # Instance normalization across the sequence length (per-instance, per-channel)
        # Input to InstanceNorm1d will be shaped (N, patch_dim, L)
        self.inst_norm = nn.InstanceNorm1d(self.patch_dim, affine=True)

        # Project patch vectors into a hidden embedding (1x conv along the patch sequence)
        self.proj = nn.Conv1d(self.patch_dim, hidden_channels, kernel_size=1, bias=False)

        # SyncBatchNorm over hidden channels (works as BatchNorm in single-process)
        self.sync_bn = nn.SyncBatchNorm(hidden_channels)

        # Scoring head to produce attention weights per spatial patch
        self.scorer = nn.Conv1d(hidden_channels, 1, kernel_size=1, bias=True)

        # Decoder head to map hidden embeddings back into 'out_channels * k * k' patch vectors
        self.decoder = nn.Conv1d(hidden_channels, out_channels * kernel_size * kernel_size, kernel_size=1, bias=True)

        # Fold to reconstruct image from patches; configured with known output size
        self.fold = nn.Fold(output_size=(img_h, img_w), kernel_size=kernel_size, padding=padding, stride=stride)

        # Small epsilon for numerical stability when normalizing by overlap counts
        self.eps = 1e-6

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Steps:
            1. Extract sliding patches (N, C*k*k, L)
            2. Apply InstanceNorm1d over channels (patch vector channels)
            3. Project to hidden embedding per patch (N, hidden, L)
            4. Apply SyncBatchNorm and ReLU
            5. Compute attention scores per spatial location and softmax over L
            6. Weight hidden embeddings by attention (emphasize important patches)
            7. Decode weighted embeddings into output patch vectors (N, out*CkK, L)
            8. Fold patches back to spatial map and normalize by the overlap count

        Args:
            x (torch.Tensor): Input tensor of shape (N, in_channels, H, W)

        Returns:
            torch.Tensor: Output tensor of shape (N, out_channels, H, W)
        """
        # 1. Unfold -> (N, patch_dim, L)
        patches = self.unfold(x)

        # 2. Instance normalization (normalizes per-instance across the length dimension)
        # InstanceNorm1d expects (N, C, L) where C == patch_dim
        patches_norm = self.inst_norm(patches)

        # 3. Project to hidden embedding -> (N, hidden, L)
        hidden = self.proj(patches_norm)

        # 4. SyncBatchNorm + activation
        hidden = self.sync_bn(hidden)
        hidden = F.relu(hidden)

        # 5. Attention scores and softmax over spatial patches (L dimension)
        scores = self.scorer(hidden)  # (N, 1, L)
        attn = torch.softmax(scores, dim=-1)  # (N, 1, L)

        # 6. Apply attention: broadcast multiply -> (N, hidden, L)
        hidden_weighted = hidden * attn

        # 7. Decode to output patch vectors -> (N, out_channels * k*k, L)
        decoded_patches = self.decoder(hidden_weighted)

        # 8. Fold back to (N, out_channels, H, W)
        out = self.fold(decoded_patches)

        # Normalize by overlap counts: fold of ones to know how many times each pixel was added
        ones_input = torch.ones_like(decoded_patches)
        overlap_count = self.fold(ones_input)
        out = out / (overlap_count + self.eps)

        # Final non-linearity
        out = torch.tanh(out)
        return out

# Configuration / default parameters (module-level)
batch_size = 8
in_channels = 3
out_channels = 4
hidden_channels = 64
img_h = 32
img_w = 32
kernel_size = 3
stride = 2
padding = 1

def get_inputs():
    """
    Returns a list with a single input tensor matching the configured shape.
    """
    x = torch.randn(batch_size, in_channels, img_h, img_w)
    return [x]

def get_init_inputs():
    """
    Returns the initialization arguments for the Model constructor in order.
    """
    return [kernel_size, stride, padding, in_channels, hidden_channels, out_channels, img_h, img_w]