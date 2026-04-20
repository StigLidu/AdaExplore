import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Complex module that:
    - Applies a 3D transposed convolution (upsampling) to each timestep of a short sequence of 3D volumes.
    - Extracts the central depth slice from each upsampled volume and zero-pads it in 2D.
    - Flattens the padded slice, projects it into a compact vector, and feeds it into a GRUCell recurrently.
    - At the end of the sequence, decodes the final GRU hidden state back into a 2D slice and embeds it into
      a 3D volume at the central depth position (other depths are zero).
    This combines nn.ConvTranspose3d, nn.ZeroPad2d, and nn.GRUCell with additional linear layers and activations.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding_3d: int,
        output_padding: int,
        pad2d: tuple,
        gru_hidden_size: int,
        D_in: int,
        H_in: int,
        W_in: int,
        seq_len: int,
        gru_input_size: int = None
    ):
        """
        Initializes layers and computes intermediate dimensions needed for shaping tensors.

        Args:
            in_channels (int): Channels of input 3D volumes.
            out_channels (int): Channels produced by ConvTranspose3d.
            kernel_size (int): Kernel size for ConvTranspose3d (assumed cubic for simplicity).
            stride (int): Stride for ConvTranspose3d.
            padding_3d (int): Padding for ConvTranspose3d.
            output_padding (int): Output padding for ConvTranspose3d.
            pad2d (tuple): ZeroPad2d padding (left, right, top, bottom).
            gru_hidden_size (int): Hidden size for GRUCell.
            D_in, H_in, W_in (int): Input depth, height, width of 3D volumes.
            seq_len (int): Number of time steps in the input sequence.
            gru_input_size (int, optional): Size of GRUCell input; if None, defaults to gru_hidden_size.
        """
        super(Model, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding_3d = padding_3d
        self.output_padding = output_padding
        self.pad2d = pad2d  # tuple: (left, right, top, bottom)
        self.gru_hidden_size = gru_hidden_size
        self.D_in = D_in
        self.H_in = H_in
        self.W_in = W_in
        self.seq_len = seq_len
        self.gru_input_size = gru_input_size if gru_input_size is not None else gru_hidden_size

        # ConvTranspose3d upsamples the input 3D volume
        self.convT = nn.ConvTranspose3d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding_3d,
            output_padding=self.output_padding,
            bias=True
        )

        # Zero padding for the 2D central slice
        self.zpad2d = nn.ZeroPad2d(self.pad2d)

        # We'll project the flattened slice to a compact vector to feed into GRUCell
        # Need to compute output spatial dimensions after ConvTranspose3d
        # Formula: out = (in - 1) * stride - 2*padding + kernel_size + output_padding
        self.D_out = (self.D_in - 1) * self.stride - 2 * self.padding_3d + self.kernel_size + self.output_padding
        self.H_out = (self.H_in - 1) * self.stride - 2 * self.padding_3d + self.kernel_size + self.output_padding
        self.W_out = (self.W_in - 1) * self.stride - 2 * self.padding_3d + self.kernel_size + self.output_padding

        # After ZeroPad2d, height and width will be:
        left, right, top, bottom = self.pad2d
        self.H_pad = self.H_out + top + bottom
        self.W_pad = self.W_out + left + right

        # Flattened feature size from the padded central slice
        self.flattened_size = self.out_channels * self.H_pad * self.W_pad

        # Linear layers for encoding slice -> gru input and decoding hidden -> slice
        self.encoder = nn.Linear(self.flattened_size, self.gru_input_size, bias=True)
        self.decoder = nn.Linear(self.gru_hidden_size, self.flattened_size, bias=True)

        # GRUCell to process sequence of encoded slice vectors
        self.grucell = nn.GRUCell(input_size=self.gru_input_size, hidden_size=self.gru_hidden_size)

        # Non-linearities
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input sequence tensor of shape
                              (seq_len, batch_size, in_channels, D_in, H_in, W_in)

        Returns:
            torch.Tensor: Output 3D volume tensor of shape
                          (batch_size, out_channels, D_out, H_pad, W_pad)
                          where only the central depth slice is non-zero (reconstructed from final hidden).
        """
        # Validate input shape
        assert x.dim() == 6, "Input must be shape (seq_len, batch, channels, D, H, W)"
        seq_len, batch_size, _, D, H, W = x.shape
        assert seq_len == self.seq_len, f"Expected seq_len={self.seq_len}, got {seq_len}"
        # Initialize hidden state to zeros on same device/dtype as input
        h = torch.zeros(batch_size, self.gru_hidden_size, device=x.device, dtype=x.dtype)

        for t in range(seq_len):
            xt = x[t]  # shape: (batch, in_channels, D_in, H_in, W_in)
            # Upsample with ConvTranspose3d
            up = self.convT(xt)  # (batch, out_channels, D_out, H_out, W_out)
            # Extract central depth slice
            d_center = self.D_out // 2
            slice2d = up[:, :, d_center, :, :]  # (batch, out_channels, H_out, W_out)
            # Zero-pad in 2D
            padded = self.zpad2d(slice2d)  # (batch, out_channels, H_pad, W_pad)
            # Flatten spatial dimensions and channels
            flat = padded.reshape(batch_size, -1)  # (batch, flattened_size)
            # Encode to GRU input dimension
            enc = self.relu(self.encoder(flat))  # (batch, gru_input_size)
            # Recurrent update
            h = self.grucell(enc, h)  # (batch, gru_hidden_size)
            # Optional non-linearity on hidden (keeps it bounded)
            h = self.tanh(h)

        # Decode final hidden state to a 2D slice and embed into a 3D volume
        decoded_flat = self.decoder(h)  # (batch, flattened_size)
        decoded_slice = decoded_flat.reshape(batch_size, self.out_channels, self.H_pad, self.W_pad)  # (batch, out_ch, H_pad, W_pad)

        # Build output volume initialized with zeros and place decoded slice at central depth
        output = torch.zeros(batch_size, self.out_channels, self.D_out, self.H_pad, self.W_pad, device=x.device, dtype=x.dtype)
        output[:, :, self.D_out // 2, :, :] = decoded_slice

        return output

# Module-level configuration (used by get_inputs and get_init_inputs)
SEQ_LEN = 8
BATCH_SIZE = 4
IN_CHANNELS = 8
OUT_CHANNELS = 16
D_IN = 4
H_IN = 16
W_IN = 16
KERNEL_SIZE = 3
STRIDE = 2
PADDING_3D = 1
OUTPUT_PADDING = 1
PAD2D = (1, 2, 1, 2)  # (left, right, top, bottom)
GRU_HIDDEN_SIZE = 128
GRU_INPUT_SIZE = 128  # can be set to None to default to GRU_HIDDEN_SIZE in init

def get_inputs():
    """
    Returns:
        list: [x] where x has shape (SEQ_LEN, BATCH_SIZE, IN_CHANNELS, D_IN, H_IN, W_IN)
    """
    x = torch.randn(SEQ_LEN, BATCH_SIZE, IN_CHANNELS, D_IN, H_IN, W_IN)
    return [x]

def get_init_inputs():
    """
    Returns initialization parameters for Model.__init__ in the same order:
    in_channels, out_channels, kernel_size, stride, padding_3d, output_padding,
    pad2d, gru_hidden_size, D_in, H_in, W_in, seq_len, gru_input_size
    """
    return [
        IN_CHANNELS,
        OUT_CHANNELS,
        KERNEL_SIZE,
        STRIDE,
        PADDING_3D,
        OUTPUT_PADDING,
        PAD2D,
        GRU_HIDDEN_SIZE,
        D_IN,
        H_IN,
        W_IN,
        SEQ_LEN,
        GRU_INPUT_SIZE
    ]