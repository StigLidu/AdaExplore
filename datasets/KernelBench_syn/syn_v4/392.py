import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    A moderately complex 1D processing module that demonstrates a small sequence
    of operations combining ReplicationPad1d, Conv1d blocks, Dropout3d and LogSoftmax.

    Computation pattern:
      1. Replication padding on the temporal dimension (ReplicationPad1d)
      2. Two stacked Conv1d + ReLU transforms to extract features
      3. Reshape to 5D and apply Dropout3d to drop whole channels across the spatial
         dimensions (acts as strong channel-wise regularization)
      4. Squeeze back to 3D, apply global average pooling over the temporal dimension
      5. Final linear projection to class logits followed by LogSoftmax
    """
    def __init__(
        self,
        in_channels: int = 4,
        hidden_channels: int = 32,
        kernel_size: int = 5,
        pad_size: int = 2,
        dropout_p: float = 0.25,
        num_classes: int = 7
    ):
        """
        Initializes the module.

        Args:
            in_channels (int): Number of input channels.
            hidden_channels (int): Number of channels in the hidden Conv1d layers.
            kernel_size (int): Kernel size for Conv1d layers.
            pad_size (int): Padding size for replication padding (applied on both sides).
            dropout_p (float): Dropout probability for Dropout3d.
            num_classes (int): Number of output classes for the final classification.
        """
        super(Model, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.pad_size = pad_size
        self.dropout_p = dropout_p
        self.num_classes = num_classes

        # Replication pad will pad both sides of the last dimension by pad_size
        self.pad = nn.ReplicationPad1d(self.pad_size)

        # Two small Conv1d feature extractors
        self.conv1 = nn.Conv1d(in_channels, hidden_channels, kernel_size=self.kernel_size)
        self.conv2 = nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, padding=1)

        # Dropout3d: expects a 5D tensor (N, C, D, H, W). We'll unsqueeze to fit that.
        self.drop3d = nn.Dropout3d(p=self.dropout_p)

        # Final classifier
        self.classifier = nn.Linear(hidden_channels, num_classes)

        # LogSoftmax for stable log-probabilities across classes
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (N, C_in, L)

        Returns:
            torch.Tensor: Log-probabilities of shape (N, num_classes)
        """
        # 1. Replication pad temporal dimension
        # Input: (N, C, L) -> Padded: (N, C, L + 2*pad_size)
        x = self.pad(x)

        # 2. Conv1d + ReLU -> produce hidden feature maps
        # conv1 reduces/increases channels and shortens length depending on kernel_size
        x = F.relu(self.conv1(x))
        # conv2 keeps same temporal dimension due to padding=1
        x = F.relu(self.conv2(x))

        # 3. Apply Dropout3d by temporarily viewing the data as 5D:
        #    (N, C, L) -> (N, C, D=1, H=1, W=L)
        x = x.unsqueeze(2).unsqueeze(3)  # add D and H dims
        x = self.drop3d(x)
        x = x.squeeze(3).squeeze(2)      # back to (N, C, L)

        # 4. Global average pooling over temporal dimension -> (N, C)
        x = x.mean(dim=2)

        # 5. Linear classifier -> (N, num_classes)
        logits = self.classifier(x)

        # 6. LogSoftmax -> log-probabilities
        log_probs = self.logsoftmax(logits)
        return log_probs

# Module-level configuration variables
batch_size = 8
in_channels = 4
length = 128

# Default initialization parameters that match the Model signature
hidden_channels = 32
kernel_size = 5
pad_size = 2
dropout_p = 0.25
num_classes = 7

def get_inputs():
    """
    Returns example input tensors for forward pass:
      - A random tensor of shape (batch_size, in_channels, length)
    """
    x = torch.randn(batch_size, in_channels, length)
    return [x]

def get_init_inputs():
    """
    Returns initialization arguments for the Model constructor:
      [in_channels, hidden_channels, kernel_size, pad_size, dropout_p, num_classes]
    """
    return [in_channels, hidden_channels, kernel_size, pad_size, dropout_p, num_classes]