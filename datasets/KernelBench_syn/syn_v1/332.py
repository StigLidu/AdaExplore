import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Complex model combining SyncBatchNorm, ZeroPad2d and an LSTM to process spatial
    feature maps as sequences. The input is expected to be a 4D tensor (N, C, H, W).
    
    Computation steps:
      1. Channel-wise synchronization batch normalization (nn.SyncBatchNorm)
      2. Zero padding around spatial dimensions (nn.ZeroPad2d)
      3. Rearrange spatial locations into a sequence: (N, H*W, C)
      4. Process the sequence with an LSTM (batch_first=True)
      5. Use the last-layer hidden state, project with a linear layer to class logits
      6. Apply log_softmax to produce log-probabilities
    """
    def __init__(self,
                 in_channels: int = 16,
                 hidden_size: int = 64,
                 num_layers: int = 2,
                 num_classes: int = 10,
                 pad: int = 1):
        super(Model, self).__init__()
        # Normalize across channels (works for N-D inputs)
        self.syncbn = nn.SyncBatchNorm(num_features=in_channels)
        # Pad spatial dimensions (left, right, top, bottom all equal when int provided)
        self.pad = nn.ZeroPad2d(pad)
        # LSTM processes flattened spatial tokens; batch_first=True for (N, seq, feature)
        self.lstm = nn.LSTM(input_size=in_channels,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True)
        # Final classifier projecting LSTM hidden state to classes
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape (N, C, H, W)

        Returns:
            torch.Tensor: Log-probabilities over classes with shape (N, num_classes)
        """
        # 1) Sync batch normalization over channels
        x = self.syncbn(x)

        # 2) Zero-pad spatial boundaries
        x = self.pad(x)  # -> (N, C, H + 2*pad, W + 2*pad)

        # 3) Rearrange to sequence of spatial tokens: (N, seq_len, C)
        N, C, H_pad, W_pad = x.shape
        x = x.permute(0, 2, 3, 1).contiguous()  # (N, H_pad, W_pad, C)
        seq = x.view(N, H_pad * W_pad, C)       # (N, seq_len, C)

        # 4) Process sequence through LSTM
        outputs, (h_n, c_n) = self.lstm(seq)    # outputs: (N, seq_len, hidden), h_n: (num_layers, N, hidden)

        # 5) Take the last layer's hidden state for each batch element
        last_layer_h = h_n[-1]                  # (N, hidden)

        # 6) Project to class logits and return log-probabilities
        logits = self.fc(last_layer_h)          # (N, num_classes)
        return F.log_softmax(logits, dim=-1)

# Configuration / default sizes
batch_size = 8
in_channels = 16
H = 8
W = 8
pad = 1
lstm_hidden = 64
lstm_layers = 2
num_classes = 10

def get_inputs():
    """
    Returns:
        List containing a single 4D input tensor of shape (batch_size, in_channels, H, W).
    """
    x = torch.randn(batch_size, in_channels, H, W)
    return [x]

def get_init_inputs():
    """
    Returns initialization parameters for constructing the Model:
      [in_channels, hidden_size, num_layers, num_classes, pad]
    """
    return [in_channels, lstm_hidden, lstm_layers, num_classes, pad]