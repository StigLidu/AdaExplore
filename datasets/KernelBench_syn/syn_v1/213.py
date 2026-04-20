import torch
import torch.nn as nn

# Configuration / shapes
BATCH = 4
C_IN = 64
C_MID = 128
C_OUT = 96
DEPTH = 8
HEIGHT = 16
WIDTH = 16

class Model(nn.Module):
    """
    A moderately complex 3D feature mixer that demonstrates:
      - circular 3D padding (nn.CircularPad3d)
      - channel projection via einsum-based tensor contractions
      - channel gating computed from spatial pooled context
      - nonlinearities: Softsign and LogSigmoid

    Computation summary:
      1. Circularly pad the input volume.
      2. Project input channels into a higher-dimensional channel space using W1.
      3. Apply Softsign nonlinearity.
      4. Compute a global context vector by spatial averaging.
      5. Produce per-output-channel gates using W2 and LogSigmoid.
      6. Project activated mid-channels to output channels using W2 (shared) and combine with gates.
      7. Add a shortcut projection from original channels via W3 and apply a final Softsign.
    """
    def __init__(self, pad_size=1):
        super(Model, self).__init__()
        # Circular padding on all three spatial dims (depth, height, width)
        # padding format: (pad_w_left, pad_w_right, pad_h_left, pad_h_right, pad_d_left, pad_d_right)
        self.pad = nn.CircularPad3d((pad_size, pad_size, pad_size, pad_size, pad_size, pad_size))
        self.softsign = nn.Softsign()
        self.logsigmoid = nn.LogSigmoid()

    def forward(self, X: torch.Tensor, W1: torch.Tensor, W2: torch.Tensor, W3: torch.Tensor) -> torch.Tensor:
        """
        Args:
            X: Input tensor of shape (B, C_IN, D, H, W)
            W1: Channel projection matrix of shape (C_IN, C_MID)
            W2: Channel projection/gating matrix of shape (C_MID, C_OUT)
            W3: Shortcut projection matrix of shape (C_IN, C_OUT)

        Returns:
            Tensor of shape (B, C_OUT, D', H', W') where spatial dims have been padded by pad_size on each side.
        """
        # 1) Circular 3D padding: increases spatial dims by 2*pad_size
        Xp = self.pad(X)  # shape: (B, C_IN, D+2p, H+2p, W+2p)

        # 2) Channel-wise projection to a mid-dimension using einsum
        #    F_mid[b, m, d, h, w] = sum_c Xp[b, c, d, h, w] * W1[c, m]
        F_mid = torch.einsum("bcdhw,cm->bmdhw", Xp, W1)

        # 3) Nonlinearity
        A = self.softsign(F_mid)

        # 4) Global context: average over spatial dimensions to get (B, C_MID)
        context = A.mean(dim=(2, 3, 4))  # shape: (B, C_MID)

        # 5) Produce gates per output channel using W2 and LogSigmoid
        #    gates[b, o] = LogSigmoid(context[b] @ W2[:, o])
        gates = self.logsigmoid(torch.matmul(context, W2))  # shape: (B, C_OUT)

        # 6) Project mid channels to output channels and apply gating
        #    P[b, o, d, h, w] = sum_m A[b, m, d, h, w] * W2[m, o]
        projected = torch.einsum("bmdhw,mo->bodhw", A, W2)  # shape: (B, C_OUT, D', H', W')

        # Broadcast gates across spatial dims and apply
        gated = projected * gates.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

        # 7) Shortcut projection from padded input channels directly to output channels
        shortcut = torch.einsum("bcdhw,co->bodhw", Xp, W3)  # shape: (B, C_OUT, D', H', W')

        # Combine with a scaled shortcut and final nonlinearity
        out = self.softsign(gated + 0.5 * shortcut)

        return out

def get_inputs():
    """
    Generates:
      - X: random input volume (BATCH, C_IN, DEPTH, HEIGHT, WIDTH)
      - W1: channel projection (C_IN, C_MID)
      - W2: mid->out projection and gating matrix (C_MID, C_OUT)
      - W3: shortcut projection (C_IN, C_OUT)
    """
    X = torch.randn(BATCH, C_IN, DEPTH, HEIGHT, WIDTH)
    W1 = torch.randn(C_IN, C_MID)
    W2 = torch.randn(C_MID, C_OUT)
    W3 = torch.randn(C_IN, C_OUT)
    return [X, W1, W2, W3]

def get_init_inputs():
    # No special initialization inputs required beyond the tensors returned by get_inputs.
    return []