import torch
import torch.nn as nn

"""
Complex PyTorch kernel module that:
- Processes a 4D image-like tensor with an RNNCell over spatial positions (treating H*W as a sequence)
- Projects per-position channels into an RNN input space
- Runs an Elman RNNCell recurrently across spatial positions and collects hidden states
- Reassembles hidden states into a feature map, upsamples it with bilinear interpolation
- Applies Hardswish nonlinearity and a final linear projection to produce a per-batch output vector

Structure matches the examples: Model class, get_inputs(), get_init_inputs(), and module-level config variables.
"""

# Configuration / sizes
BATCH = 8          # number of examples in a batch
IN_CHANNELS = 32   # input channels per spatial location
H = 16             # input height
W = 16             # input width
RNN_INPUT = 64     # input dimension to RNNCell after projection
HIDDEN_DIM = 128   # hidden dimension of RNNCell
OUT_DIM = 512      # final output dimension after pooling + projection

class Model(nn.Module):
    """
    Model that combines per-position projection, an RNNCell recurrence across spatial locations,
    bilinear upsampling of the reconstructed feature map, Hardswish activation, and final projection.

    forward inputs:
        x: Tensor of shape (BATCH, IN_CHANNELS, H, W)
        h0: Initial hidden state tensor of shape (BATCH, HIDDEN_DIM)

    returns:
        Tensor of shape (BATCH, OUT_DIM)
    """

    def __init__(self):
        super(Model, self).__init__()
        # Project channels -> RNN input dimension for each spatial location
        self.input_proj = nn.Linear(IN_CHANNELS, RNN_INPUT)

        # Elman RNN cell (tanh nonlinearity by default)
        self.rnn_cell = nn.RNNCell(RNN_INPUT, HIDDEN_DIM)

        # Upsample the assembled hidden-feature map by factor 2 in H and W using bilinear interpolation
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=(2, 2))

        # Non-linear activation applied after upsampling
        self.hardswish = nn.Hardswish()

        # Final projection from pooled hidden features to desired output dimension
        self.output_proj = nn.Linear(HIDDEN_DIM, OUT_DIM)

    def forward(self, x: torch.Tensor, h0: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) input tensor
            h0: (B, HIDDEN_DIM) initial hidden state for the RNNCell

        Returns:
            out: (B, OUT_DIM) output tensor
        """
        B, C, H_in, W_in = x.shape
        assert C == IN_CHANNELS and H_in == H and W_in == W, \
            f"Expected input shape (B, {IN_CHANNELS}, {H}, {W}), got {(B,C,H_in,W_in)}"
        assert h0.shape == (B, HIDDEN_DIM), f"h0 must be (BATCH, HIDDEN_DIM), got {h0.shape}"

        # 1) Flatten spatial dims and create sequence of length L = H * W, features = IN_CHANNELS
        # x_seq: (B, L, C)
        L = H * W
        x_seq = x.permute(0, 2, 3, 1).reshape(B, L, C)

        # 2) Project per-position channels to RNN input dimension
        # projected: (B, L, RNN_INPUT)
        projected = self.input_proj(x_seq)

        # 3) Run RNNCell across the spatial sequence, collecting hidden states
        hs = torch.zeros(B, L, HIDDEN_DIM, device=x.device, dtype=x.dtype)
        h = h0
        for t in range(L):
            inp_t = projected[:, t, :]      # (B, RNN_INPUT)
            h = self.rnn_cell(inp_t, h)     # (B, HIDDEN_DIM)
            hs[:, t, :] = h

        # 4) Reassemble hidden states into spatial feature map: (B, HIDDEN_DIM, H, W)
        hs_map = hs.reshape(B, H, W, HIDDEN_DIM).permute(0, 3, 1, 2)

        # 5) Upsample spatially (bilinear) to (B, HIDDEN_DIM, 2H, 2W)
        up_map = self.upsample(hs_map)

        # 6) Apply Hardswish non-linearity
        activated = self.hardswish(up_map)

        # 7) Global average pooling over spatial dims -> (B, HIDDEN_DIM)
        pooled = activated.mean(dim=(2, 3))

        # 8) Final linear projection -> (B, OUT_DIM)
        out = self.output_proj(pooled)
        return out

def get_inputs():
    """
    Returns:
        [x, h0] where
        x: (BATCH, IN_CHANNELS, H, W)
        h0: initial hidden state (BATCH, HIDDEN_DIM)
    """
    x = torch.randn(BATCH, IN_CHANNELS, H, W)
    h0 = torch.randn(BATCH, HIDDEN_DIM)
    return [x, h0]

def get_init_inputs():
    """
    No extra initialization parameters needed beyond random inputs;
    return an empty list to match the example conventions.
    """
    return []