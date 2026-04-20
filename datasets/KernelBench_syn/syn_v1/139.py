import torch
import torch.nn as nn

# Module-level configuration
batch_size = 8
in_channels = 32
hidden_channels = 64  # must be divisible by num_groups
height = 64
width = 64
kernel_size = 3
padding = 1
num_groups = 8
num_classes = 100
threshold_value = 1e-3  # small threshold for nn.Threshold (acts like a soft ReLU cutoff)

class Model(nn.Module):
    """
    Complex convolutional block demonstrating:
      - Initial 2D convolution
      - Group Normalization
      - Threshold non-linearity (nn.Threshold)
      - 1x1 pointwise convolution
      - Lazy BatchNorm2d that initializes on first forward
      - Residual connection via a 1x1 projection
      - Global average pooling and final linear classifier via einsum

    Forward computation steps:
      x -> conv1 (KxK) -> groupnorm -> threshold -> conv_pw (1x1) -> lazy_bn
      residual: x_proj = proj_conv(input)
      out = lazy_bn_output + x_proj
      pooled = out.mean(dim=(2,3))           # global average pool over H,W -> (B, C)
      logits = einsum("bc,ck->bk", pooled, classifier_weight) + classifier_bias
    """
    def __init__(
        self,
        in_ch: int,
        hidden_ch: int,
        kernel_size: int,
        padding: int,
        num_groups: int,
        num_classes: int,
        threshold_val: float = 0.0,
    ):
        super(Model, self).__init__()
        # Feature extractor
        self.conv1 = nn.Conv2d(in_channels=in_ch, out_channels=hidden_ch,
                               kernel_size=kernel_size, stride=1, padding=padding, bias=False)
        # GroupNorm requires (num_groups, num_channels)
        self.gn = nn.GroupNorm(num_groups=num_groups, num_channels=hidden_ch)
        # Threshold non-linearity; below threshold replaced with 0.0
        self.threshold = nn.Threshold(threshold_val, 0.0)
        # Pointwise conv to mix channels
        self.conv_pw = nn.Conv2d(in_channels=hidden_ch, out_channels=hidden_ch,
                                 kernel_size=1, stride=1, padding=0, bias=False)
        # LazyBatchNorm2d will infer num_features on the first forward
        self.lazy_bn = nn.LazyBatchNorm2d()
        # Projection for residual connection (project input channels -> hidden_ch)
        self.proj_conv = nn.Conv2d(in_channels=in_ch, out_channels=hidden_ch,
                                   kernel_size=1, stride=1, padding=0, bias=False)
        # Classifier weights and bias (used with einsum)
        self.classifier_weight = nn.Parameter(torch.randn(hidden_ch, num_classes))
        self.classifier_bias = nn.Parameter(torch.randn(num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (B, in_channels, H, W)

        Returns:
            logits: Tensor of shape (B, num_classes)
        """
        # Initial convolution -> normalization -> non-linearity
        t = self.conv1(x)               # (B, hidden_ch, H, W)
        t = self.gn(t)                  # GroupNorm
        t = self.threshold(t)           # Threshold non-linearity

        # Pointwise conv followed by lazy batch norm (will initialize on first call)
        t = self.conv_pw(t)             # (B, hidden_ch, H, W)
        t = self.lazy_bn(t)             # LazyBatchNorm2d

        # Residual connection (project input to match hidden channels)
        x_proj = self.proj_conv(x)      # (B, hidden_ch, H, W)
        out = t + x_proj                # Residual add

        # Global average pooling over spatial dims -> (B, hidden_ch)
        pooled = out.mean(dim=(2, 3))

        # Final linear classifier via einsum for clarity of contraction
        logits = torch.einsum("bc,ck->bk", pooled, self.classifier_weight) + self.classifier_bias
        return logits

def get_inputs():
    """
    Returns:
        list: [input_tensor]
    """
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]

def get_init_inputs():
    """
    Returns the initialization parameters for the Model constructor in the correct order.
    """
    return [in_channels, hidden_channels, kernel_size, padding, num_groups, num_classes, threshold_value]