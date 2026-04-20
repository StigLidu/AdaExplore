import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Patch-based vision encoder that:
      - Pads input images to a multiple of patch_size using ConstantPad2d
      - Extracts non-overlapping patches via unfold
      - Projects patches to an embedding space
      - Applies a learned gating mechanism
      - Adds learnable positional embeddings
      - Processes the token sequence with a single TransformerEncoderLayer
      - Pools token outputs and produces class logits followed by LogSoftmax

    This model demonstrates the combination of nn.ConstantPad2d, nn.TransformerEncoderLayer,
    and nn.LogSoftmax in a compact image-to-logits pipeline.
    """
    def __init__(self):
        super(Model, self).__init__()

        # Patch embedding projection
        patch_dim = CHANNELS * PATCH_SIZE * PATCH_SIZE
        self.proj = nn.Linear(patch_dim, EMBED_DIM, bias=True)

        # Learned gating per embedding dimension (applied element-wise)
        self.gate = nn.Parameter(torch.ones(EMBED_DIM))

        # Maximum number of patches computed from configured image size
        max_patches_h = (IMAGE_HEIGHT + PATCH_SIZE - 1) // PATCH_SIZE
        max_patches_w = (IMAGE_WIDTH + PATCH_SIZE - 1) // PATCH_SIZE
        max_patches = max_patches_h * max_patches_w

        # Positional embeddings for tokens (num_patches, embed_dim)
        self.pos_embed = nn.Parameter(torch.randn(max_patches, EMBED_DIM) * 0.02)

        # Single-layer Transformer encoder (seq_len, batch, embed_dim expected)
        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=EMBED_DIM,
            nhead=NHEAD,
            dim_feedforward=FEEDFORWARD_DIM,
            activation='relu',
            batch_first=False  # we'll provide (seq_len, batch, embed_dim)
        )

        # Final classifier and log-softmax
        self.classifier = nn.Linear(EMBED_DIM, NUM_CLASSES)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input image tensor of shape (B, C, H, W)

        Returns:
            torch.Tensor: Log-probabilities over classes, shape (B, NUM_CLASSES)
        """
        B, C, H, W = x.shape
        assert C == CHANNELS, f"Expected {CHANNELS} channels, got {C}"

        # Compute required padding to make H and W divisible by PATCH_SIZE
        pad_h = (PATCH_SIZE - (H % PATCH_SIZE)) % PATCH_SIZE
        pad_w = (PATCH_SIZE - (W % PATCH_SIZE)) % PATCH_SIZE

        # Use nn.ConstantPad2d to pad right and bottom (left/top = 0)
        pad_layer = nn.ConstantPad2d((0, pad_w, 0, pad_h), 0.0)
        x_padded = pad_layer(x)  # shape (B, C, H+pad_h, W+pad_w)

        Hp = H + pad_h
        Wp = W + pad_w
        # Extract non-overlapping patches: kernel_size=PATCH_SIZE, stride=PATCH_SIZE
        # F.unfold outputs (B, C*patch_h*patch_w, num_patches)
        patches = F.unfold(x_padded, kernel_size=PATCH_SIZE, stride=PATCH_SIZE)
        # Convert to (B, num_patches, patch_dim)
        patches = patches.transpose(1, 2)

        # Linear projection to embeddings: (B, num_patches, EMBED_DIM)
        token_embeddings = self.proj(patches)

        # Apply learned gating (shape broadcast to (B, num_patches, EMBED_DIM))
        gate = torch.sigmoid(self.gate)  # (EMBED_DIM,)
        token_embeddings = token_embeddings * gate.unsqueeze(0).unsqueeze(0)

        # Add positional embeddings (slice to actual number of patches)
        num_patches = token_embeddings.size(1)
        pos = self.pos_embed[:num_patches, :]  # (num_patches, EMBED_DIM)
        token_embeddings = token_embeddings + pos.unsqueeze(0)

        # Prepare for transformer: (seq_len, batch, embed_dim)
        trans_in = token_embeddings.transpose(0, 1)

        # Transformer encoding (self-attention + feedforward)
        trans_out = self.transformer_layer(trans_in)  # (seq_len, B, EMBED_DIM)

        # Pool across tokens (mean pooling) -> (B, EMBED_DIM)
        pooled = trans_out.mean(dim=0)

        # Classification and log-probabilities
        logits = self.classifier(pooled)
        log_probs = self.logsoftmax(logits)
        return log_probs

# Configuration variables (module-level)
BATCH_SIZE = 8
CHANNELS = 3
IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224
PATCH_SIZE = 16
EMBED_DIM = 512
NHEAD = 8
FEEDFORWARD_DIM = 2048
NUM_CLASSES = 1000

def get_inputs():
    """
    Generates a batch of random images for testing.

    Returns:
        list: single-element list containing an input tensor of shape (BATCH_SIZE, CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH)
    """
    x = torch.randn(BATCH_SIZE, CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH)
    return [x]

def get_init_inputs():
    """
    No external initialization parameters required for the model; it uses module-level configuration.

    Returns:
        list: Empty list.
    """
    return []