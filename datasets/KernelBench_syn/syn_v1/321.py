import torch
import torch.nn as nn

# Configuration variables
batch_size = 8
in_channels = 16
mid_channels = 32
height = 64
width = 64
pooled_h = 8
pooled_w = 8
num_classes = 10
alpha_dropout_p = 0.1

class Model(nn.Module):
    """
    Complex vision-style module that combines convolution, SyncBatchNorm,
    AdaptiveMaxPool2d, AlphaDropout, and a learned channel gating mechanism.

    Computation pattern:
    1. 3x3 convolution -> SyncBatchNorm -> ReLU
    2. AdaptiveMaxPool2d to (pooled_h, pooled_w)
    3. AlphaDropout
    4. 1x1 convolution (channel mixing)
    5. Global adaptive max pool to (1,1) -> Linear -> Sigmoid to create channel gates
    6. Apply gates to spatial feature map (channel-wise scaling)
    7. Flatten pooled feature map and apply final Linear to produce logits
    """
    def __init__(self):
        super(Model, self).__init__()
        # Initial spatial conv and normalization
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.SyncBatchNorm(mid_channels)

        # Spatial pooling and dropout
        self.pool = nn.AdaptiveMaxPool2d((pooled_h, pooled_w))
        self.alpha_drop = nn.AlphaDropout(p=alpha_dropout_p)

        # Channel mixing
        self.conv_pw = nn.Conv2d(mid_channels, mid_channels, kernel_size=1, bias=True)

        # Gate network: global pooled context -> channel gates
        self.global_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.fc_gate = nn.Linear(mid_channels, mid_channels)

        # Final classifier from flattened pooled feature map
        self.classifier = nn.Linear(mid_channels * pooled_h * pooled_w, num_classes)

        # Activation
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (B, in_channels, H, W)

        Returns:
            logits: Tensor of shape (B, num_classes)
        """
        # Conv + SyncBatchNorm + ReLU
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)

        # Spatial adaptive pooling + AlphaDropout
        x = self.pool(x)  # (B, mid_channels, pooled_h, pooled_w)
        x = self.alpha_drop(x)

        # Pointwise conv for channel mixing
        x = self.conv_pw(x)  # (B, mid_channels, pooled_h, pooled_w)

        # Compute channel-wise gates from global context
        ctx = self.global_pool(x)  # (B, mid_channels, 1, 1)
        ctx = ctx.view(ctx.size(0), -1)  # (B, mid_channels)
        gates = torch.sigmoid(self.fc_gate(ctx))  # (B, mid_channels)
        gates = gates.view(gates.size(0), gates.size(1), 1, 1)  # (B, mid_channels, 1, 1)

        # Apply gates (channel-wise scaling)
        x = x * gates

        # Flatten pooled feature map and classify
        x = x.view(x.size(0), -1)  # (B, mid_channels * pooled_h * pooled_w)
        logits = self.classifier(x)  # (B, num_classes)
        return logits

def get_inputs():
    """
    Returns a list containing a single input tensor shaped (batch_size, in_channels, height, width).
    """
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]

def get_init_inputs():
    """
    No special initialization parameters required; return empty list for compatibility.
    """
    return []