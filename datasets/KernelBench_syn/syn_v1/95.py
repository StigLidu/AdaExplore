import torch
import torch.nn as nn
import math

class Model(nn.Module):
    """
    Patch-based aggregation model that:
    - Extracts sliding patches from images using nn.Unfold
    - Projects each patch into a lower-dimensional embedding via a linear layer
    - Applies a LeakyReLU non-linearity
    - Builds a context vector by averaging patch embeddings
    - Computes attention logits between context and each patch embedding, normalizes them with LogSoftmax
    - Uses the attention distribution to aggregate patch embeddings into a global descriptor
    - Classifies the aggregated descriptor and returns log-probabilities via LogSoftmax

    This model demonstrates a non-trivial combination of nn.Unfold, nn.Linear, nn.LeakyReLU,
    and nn.LogSoftmax modules together with typical tensor reshaping and batched reductions.
    """
    def __init__(
        self,
        in_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        proj_dim: int,
        num_classes: int,
        negative_slope: float = 0.01
    ):
        super(Model, self).__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.proj_dim = proj_dim
        self.num_classes = num_classes

        # Extract local patches
        self.unfold = nn.Unfold(kernel_size=kernel_size, stride=stride, padding=padding)

        # Project flattened patch -> embedding
        patch_size = in_channels * kernel_size * kernel_size
        self.proj = nn.Linear(patch_size, proj_dim, bias=True)

        # Non-linearity
        self.activation = nn.LeakyReLU(negative_slope=negative_slope, inplace=False)

        # Final classifier from aggregated descriptor to class logits
        self.classifier = nn.Linear(proj_dim, num_classes, bias=True)

        # Two LogSoftmax modules:
        # - one to normalize attention logits across patches (dim = number of patches)
        # - one to produce log-probabilities over classes (dim = num_classes)
        self.log_softmax_patches = nn.LogSoftmax(dim=1)   # applied over patch dimension (L)
        self.log_softmax_classes = nn.LogSoftmax(dim=1)   # applied over class dimension

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, in_channels, height, width)

        Returns:
            log_probs: Tensor of shape (batch_size, num_classes) containing log-probabilities.
        """
        # x_unfold: (B, C*ks*ks, L) where L = number of patches per sample
        x_unfold = self.unfold(x)

        # Reformat to (B, L, patch_size)
        x_patches = x_unfold.transpose(1, 2)

        # Project each patch to an embedding: (B, L, proj_dim)
        embedded = self.proj(x_patches)

        # Non-linear activation
        activated = self.activation(embedded)

        # Context vector: average of patch embeddings per sample -> (B, proj_dim)
        context = activated.mean(dim=1)

        # Compute attention logits as scaled dot-product between each patch and context
        # attention_logits: (B, L)
        # scale by sqrt(dim) for stability
        scale = math.sqrt(self.proj_dim) if self.proj_dim > 0 else 1.0
        attention_logits = torch.einsum('bld,bd->bl', activated, context) / scale

        # Normalize attention logits with LogSoftmax across patches (dim=1 since shape is (B, L))
        log_attn = self.log_softmax_patches(attention_logits)

        # Convert to probabilities to form weighted sum (exp(log_softmax) == softmax)
        attn_weights = log_attn.exp()  # (B, L)

        # Aggregate patch embeddings into a single descriptor: weighted sum over L -> (B, proj_dim)
        aggregated = torch.einsum('bl,bld->bd', attn_weights, activated)

        # Classify aggregated descriptor
        logits = self.classifier(aggregated)  # (B, num_classes)

        # Return log-probabilities over classes
        log_probs = self.log_softmax_classes(logits)
        return log_probs

# Configuration / default initialization parameters
batch_size = 8
in_channels = 3
height = 32
width = 32

kernel_size = 4
stride = 4
padding = 0

proj_dim = 128
num_classes = 10
negative_slope = 0.1

def get_inputs():
    """
    Returns example input tensors for a forward pass.
    """
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]

def get_init_inputs():
    """
    Returns the initialization arguments for Model in the same order as Model.__init__ expects.
    """
    return [in_channels, kernel_size, stride, padding, proj_dim, num_classes, negative_slope]