import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Complex 1D sequence module that demonstrates a small processing pipeline:
      - Constant padding on the temporal dimension
      - SELU non-linearity
      - Local Response Normalization across channels
      - Channel descriptor pooling and normalized gating
      - Channel-wise re-scaling of the normalized feature map
      - Per-sample channel covariance aggregation and projection to a compact descriptor

    Input shape: (batch, channels, seq_len)
    Output shape: (batch, channels) -- compact per-channel descriptors
    """
    def __init__(self,
                 pad: int = 3,
                 lrn_size: int = 5,
                 lrn_alpha: float = 1e-4,
                 lrn_beta: float = 0.75,
                 lrn_k: float = 2.0,
                 eps: float = 1e-6):
        """
        Initializes the processing layers and hyperparameters.

        Args:
            pad: Number of values to pad on each side of the sequence.
            lrn_size: Local window size for LocalResponseNorm.
            lrn_alpha: Alpha parameter for LocalResponseNorm.
            lrn_beta: Beta parameter for LocalResponseNorm.
            lrn_k: K parameter for LocalResponseNorm.
            eps: Small epsilon used for numerical stability.
        """
        super(Model, self).__init__()
        # Padding layer: pads both sides of the last dimension by `pad` with constant value 0.1
        self.pad = nn.ConstantPad1d(pad, 0.1)
        # SELU non-linearity applied at two points in the pipeline
        self.selu = nn.SELU()
        # Local response normalization operating across channels
        self.lrn = nn.LocalResponseNorm(lrn_size, alpha=lrn_alpha, beta=lrn_beta, k=lrn_k)
        # Numerical stability epsilon for channel normalization
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Steps:
          1. Pad the temporal dimension with a constant.
          2. Apply SELU activation.
          3. Apply Local Response Normalization across channels.
          4. Compute per-channel pooled descriptors (mean across time).
          5. Generate normalized channel gates with SELU and L2 normalization.
          6. Re-scale the feature map by the channel gates.
          7. Compute per-sample channel covariance (C x C), average across time,
             and reduce to a compact per-channel descriptor.

        Args:
            x: Input tensor of shape (N, C, L)

        Returns:
            Tensor of shape (N, C) representing compact per-channel descriptors.
        """
        # 1. Pad temporal dimension: result shape (N, C, L + 2*pad)
        x_padded = self.pad(x)

        # 2. Non-linearity
        x_act = self.selu(x_padded)

        # 3. Local response normalization across channels
        #    (Keeps shape (N, C, L_padded))
        x_norm = self.lrn(x_act)

        # 4. Per-channel pooling (global mean across temporal dimension)
        #    pool shape: (N, C)
        pool = torch.mean(x_norm, dim=2)

        # 5. Channel gating: SELU followed by L2 normalization per sample
        gates = self.selu(pool)  # (N, C)
        # L2 normalize along channels for each sample
        gates = gates / (torch.norm(gates, p=2, dim=1, keepdim=True) + self.eps)  # (N, C)

        # 6. Re-scale the normalized feature map by channel gates
        #    Expand gates to match temporal dimension
        gates_expanded = gates.unsqueeze(-1)  # (N, C, 1)
        x_scaled = x_norm * gates_expanded  # (N, C, L_padded)

        # 7. Compute per-sample channel covariance matrices and reduce them to descriptors
        #    Flatten temporal dimension length Ls
        N, C, Ls = x_scaled.shape
        # Compute channel covariance: (N, C, C) = x_scaled @ x_scaled^T / Ls
        # Use bmm by reshaping to (N, C, Ls) and (N, Ls, C)
        cov = torch.bmm(x_scaled, x_scaled.transpose(1, 2)) / float(Ls + self.eps)  # (N, C, C)

        # Reduce covariance to a compact per-channel descriptor: mean across cov columns
        # final_desc shape: (N, C)
        final_desc = cov.mean(dim=2)

        # Final non-linearity for stability / additional expressiveness
        out = self.selu(final_desc)

        return out


# Configuration variables (module level)
batch_size = 8
channels = 64
seq_len = 1024

def get_inputs():
    """
    Generates a random input tensor with shape (batch_size, channels, seq_len).
    """
    x = torch.randn(batch_size, channels, seq_len)
    return [x]

def get_init_inputs():
    """
    Returns initialization configuration used by the model constructor.
    Empty list as defaults are embedded in the module, but kept here to
    match example structure and allow external initialization if needed.
    """
    return []  # No additional initialization inputs required