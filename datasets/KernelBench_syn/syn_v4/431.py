import torch
import torch.nn as nn

class Model(nn.Module):
    """
    A moderately complex convolutional block that demonstrates a combination of lazy batch normalization,
    two distinct non-linearities (Mish and GELU), residual connectivity, and a global pooling + classifier head.

    Structure:
      - Conv2d(in_ch -> mid_ch, 3x3, padding=1)
      - LazyBatchNorm2d (initialized on first forward)
      - Mish activation
      - Conv2d(mid_ch -> mid_ch, 3x3, padding=1)
      - GELU activation
      - Pointwise Conv2d(mid_ch -> out_ch, 1x1)
      - Optional projection of the input to out_ch for residual addition
      - Adaptive average pooling (1x1), flatten, and Linear(out_ch -> num_classes)

    This pattern mixes normalization, two different activations, and a residual path to form a functionally
    distinct computation from the example files.
    """
    def __init__(self, in_ch: int, mid_ch: int, out_ch: int, num_classes: int, use_residual: bool = True):
        """
        Args:
            in_ch (int): Number of input channels.
            mid_ch (int): Number of channels in the intermediate feature maps.
            out_ch (int): Number of output channels produced by the block before pooling.
            num_classes (int): Size of the final classification head output.
            use_residual (bool): If True, adds the input (optionally projected) to the output as a residual.
        """
        super(Model, self).__init__()

        # First convolution to expand/transform the input feature space
        self.conv1 = nn.Conv2d(in_ch, mid_ch, kernel_size=3, stride=1, padding=1, bias=False)
        # LazyBatchNorm2d will infer num_features (mid_ch) on the first forward pass
        self.bn1 = nn.LazyBatchNorm2d()
        # Non-linearities
        self.mish = nn.Mish()
        self.gelu = nn.GELU()

        # Second convolution to process features in the mid channel space
        self.conv2 = nn.Conv2d(mid_ch, mid_ch, kernel_size=3, stride=1, padding=1, bias=False)

        # Pointwise conv to produce the desired output channels
        self.conv3 = nn.Conv2d(mid_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=False)

        # Optional projection for residual connection (input -> out_ch) if channels mismatch
        self.use_residual = use_residual
        if use_residual and in_ch != out_ch:
            self.proj = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=False)
        else:
            self.proj = None

        # Classification head after global pooling
        self.fc = nn.Linear(out_ch, num_classes)

        # Small weight initialization for stability
        nn.init.kaiming_normal_(self.conv1.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv2.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv3.weight, nonlinearity='linear')
        if self.proj is not None:
            nn.init.kaiming_normal_(self.proj.weight, nonlinearity='linear')
        nn.init.normal_(self.fc.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.fc.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass implementing the described computation graph.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_ch, H, W)

        Returns:
            torch.Tensor: Logits tensor of shape (batch_size, num_classes)
        """
        identity = x

        # conv -> lazy batchnorm -> Mish
        out = self.conv1(x)
        out = self.bn1(out)   # bn1 will lazily initialize its num_features to match out.shape[1]
        out = self.mish(out)

        # conv -> GELU
        out = self.conv2(out)
        out = self.gelu(out)

        # pointwise conv to adjust channels
        out = self.conv3(out)

        # residual addition (with optional projection)
        if self.use_residual:
            if self.proj is not None:
                identity = self.proj(identity)
            out = out + identity

        # global pooling + classifier
        out = nn.functional.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        logits = self.fc(out)
        return logits

# Configuration / default inputs for testing
batch_size = 8
in_channels = 3
mid_channels = 64
out_channels = 128
img_h = 32
img_w = 32
num_classes = 10

def get_inputs():
    """
    Create a random input batch compatible with the model's expected shape.

    Returns:
        list: [input_tensor] where input_tensor shape is (batch_size, in_channels, img_h, img_w)
    """
    x = torch.randn(batch_size, in_channels, img_h, img_w)
    return [x]

def get_init_inputs():
    """
    Provide constructor parameters for the Model.

    Returns:
        list: [in_ch, mid_ch, out_ch, num_classes, use_residual]
    """
    return [in_channels, mid_channels, out_channels, num_classes, True]