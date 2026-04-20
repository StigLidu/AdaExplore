import torch
import torch.nn as nn

"""
Complex PyTorch kernel module combining:
- nn.LazyConvTranspose1d (lazy transposed convolution / upsampling)
- nn.Hardsigmoid activation
- nn.Fold to reconstruct overlapping 1D patches into a sequence

Structure follows examples:
- Model class inheriting from nn.Module
- get_inputs() returning runtime tensors
- get_init_inputs() returning initialization parameters (none needed here)

Computation pattern (forward):
1. Transposed convolution (upsample) with lazy in_channels inference
2. Hardsigmoid activation
3. Extract sliding 1D patches via tensor.unfold
4. Apply learnable per-channel-per-kernel weights to patches
5. Reconstruct overlapped sequence via nn.Fold
"""

# Module-level configuration
BATCH_SIZE = 8
IN_CHANNELS = 8
INPUT_LENGTH = 128

DECONV_OUT_CHANNELS = 16
DECONV_KERNEL = 4
DECONV_STRIDE = 2
DECONV_PADDING = 1
DECONV_OUTPUT_PADDING = 0  # keep zero for deterministic size

FOLD_KERNEL = 5  # kernel width for patch extraction and folding
FOLD_STRIDE = 1

# Compute the output length after ConvTranspose1d:
# L_out = (L_in - 1) * stride - 2*padding + kernel_size + output_padding
OUTPUT_LENGTH = (INPUT_LENGTH - 1) * DECONV_STRIDE - 2 * DECONV_PADDING + DECONV_KERNEL + DECONV_OUTPUT_PADDING

class Model(nn.Module):
    """
    Model that upsamples a 1D signal with a lazy transposed convolution,
    applies a Hardsigmoid non-linearity, extracts sliding patches, modulates
    them with learnable per-channel/per-offset weights, and reconstructs the
    overlapped output using nn.Fold.
    """
    def __init__(self):
        super(Model, self).__init__()
        # Lazy ConvTranspose1d: in_channels will be inferred at first forward pass
        # out_channels is fixed and known
        self.deconv = nn.LazyConvTranspose1d(
            out_channels=DECONV_OUT_CHANNELS,
            kernel_size=DECONV_KERNEL,
            stride=DECONV_STRIDE,
            padding=DECONV_PADDING,
            output_padding=DECONV_OUTPUT_PADDING,
            bias=True
        )

        # Non-linear activation
        self.act = nn.Hardsigmoid()

        # Learnable modulation weights applied to each (channel, kernel_offset)
        # Shape: (out_channels, FOLD_KERNEL)
        self.patch_weights = nn.Parameter(
            torch.randn(DECONV_OUT_CHANNELS, FOLD_KERNEL) * 0.1
        )

        # Fold to reconstruct a 1xOUTPUT_LENGTH "image" from patches:
        # - kernel_size is (1, FOLD_KERNEL) because we're treating the spatial
        #   dimension as height=1, width=length.
        # - The Fold expects input of shape (N, C * K, L_out) where K = kernel_area (1 * FOLD_KERNEL)
        self.fold = nn.Fold(
            output_size=(1, OUTPUT_LENGTH),
            kernel_size=(1, FOLD_KERNEL),
            stride=FOLD_STRIDE,
            padding=0
        )

        # Optional small pointwise conv to mix output channels after folding
        self.post_mix = nn.Conv1d(DECONV_OUT_CHANNELS, DECONV_OUT_CHANNELS, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, input_length)

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, OUTPUT_LENGTH)
        """
        # 1) Upsample / transpose convolution (will lazily infer in_channels)
        y = self.deconv(x)  # shape: (B, DECONV_OUT_CHANNELS, OUTPUT_LENGTH)

        # 2) Non-linearity
        y = self.act(y)  # elementwise Hardsigmoid

        # 3) Extract sliding 1D patches along the length dimension
        #    After unfold: (B, out_ch, L_out, FOLD_KERNEL) where
        #    L_out = OUTPUT_LENGTH - FOLD_KERNEL + 1
        patches = y.unfold(dimension=2, size=FOLD_KERNEL, step=FOLD_STRIDE)

        # 4) Modulate patches by learnable weights (per-channel, per-kernel-offset)
        #    Broadcast patch_weights: (1, out_ch, 1, FOLD_KERNEL)
        weights = self.patch_weights.view(1, DECONV_OUT_CHANNELS, 1, FOLD_KERNEL)
        modulated = patches * weights  # shape unchanged

        # 5) Prepare for Fold: permute to (B, out_ch * FOLD_KERNEL, L_out)
        B = modulated.shape[0]
        L_out = modulated.shape[2]
        modulated = modulated.permute(0, 1, 3, 2).contiguous().view(B, DECONV_OUT_CHANNELS * FOLD_KERNEL, L_out)

        # 6) Reconstruct the overlapped signal with Fold -> (B, out_ch, 1, OUTPUT_LENGTH)
        folded = self.fold(modulated)

        # 7) Squeeze the height dimension to get (B, out_ch, OUTPUT_LENGTH)
        folded = folded.squeeze(2)

        # 8) Optional post-processing: pointwise mixing across channels
        out = self.post_mix(folded)

        return out

def get_inputs():
    """
    Returns a list of input tensors for the model's forward method.
    Shape: (BATCH_SIZE, IN_CHANNELS, INPUT_LENGTH)
    """
    x = torch.randn(BATCH_SIZE, IN_CHANNELS, INPUT_LENGTH)
    return [x]

def get_init_inputs():
    """
    Returns any inputs required for initialization.
    This module uses lazy initialization for the ConvTranspose1d, so no extra
    initialization inputs are necessary.
    """
    return []