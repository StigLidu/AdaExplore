import torch
import torch.nn as nn

# Configuration (module-level)
batch_size = 8
src_len = 32
tgt_len = 16
d_model = 128           # embedding dimension used by TransformerDecoder
rnn_input_size = 64     # input size per time-step for the RNNCell
rnn_hidden_size = 128   # hidden size of the RNNCell (we will project this to d_model)
nhead = 8
num_decoder_layers = 2

class Model(nn.Module):
    """
    Composite module that:
      - Runs an nn.RNNCell recurrently over a target input sequence to produce step-wise hidden states.
      - Projects RNN hidden states into the Transformer model dimension and applies GELU.
      - Uses an nn.TransformerDecoder to attend the processed RNN outputs (as `tgt`) to an encoder-like `memory`
        built from a source sequence.
      - Returns the decoded sequence in (batch, tgt_len, d_model) layout.
    """
    def __init__(self,
                 rnn_input_size: int,
                 rnn_hidden_size: int,
                 d_model: int,
                 nhead: int,
                 num_decoder_layers: int):
        super(Model, self).__init__()

        # Recurrent cell processing the target-side inputs step-by-step
        self.rnn_cell = nn.RNNCell(input_size=rnn_input_size, hidden_size=rnn_hidden_size, nonlinearity='tanh')

        # Project RNN hidden state to Transformer d_model
        self.rnn_to_model = nn.Linear(rnn_hidden_size, d_model)

        # Optional nonlinear activation drawn from provided layers
        self.gelu = nn.GELU()

        # Build Transformer decoder stack. Use GELU activation internally as well to keep consistency.
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, activation='gelu')
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        # Final projection (identity-like) to emphasize we might produce some final logits or embeddings
        self.output_proj = nn.Linear(d_model, d_model)

        # Small layer norm for stability
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, src: torch.Tensor, tgt_input: torch.Tensor, h0: torch.Tensor) -> torch.Tensor:
        """
        Args:
            src (torch.Tensor): Source sequence / memory. Shape: (batch, src_len, d_model)
            tgt_input (torch.Tensor): Raw target inputs for the RNNCell. Shape: (batch, tgt_len, rnn_input_size)
            h0 (torch.Tensor): Initial hidden state for the RNNCell. Shape: (batch, rnn_hidden_size)

        Returns:
            torch.Tensor: Decoded output sequence. Shape: (batch, tgt_len, d_model)
        """
        batch, t_len, _ = tgt_input.shape

        # Recurrently compute hidden states for each time-step
        h = h0  # (batch, rnn_hidden_size)
        rnn_projected_steps = []
        for t in range(t_len):
            x_t = tgt_input[:, t, :]                          # (batch, rnn_input_size)
            h = self.rnn_cell(x_t, h)                        # (batch, rnn_hidden_size)
            proj = self.rnn_to_model(h)                      # (batch, d_model)
            activated = self.gelu(proj)                      # (batch, d_model)
            rnn_projected_steps.append(activated)

        # Stack to (tgt_len, batch, d_model) as expected by nn.Transformer modules
        tgt = torch.stack(rnn_projected_steps, dim=0)        # (tgt_len, batch, d_model)

        # Prepare memory: Transformer expects (src_len, batch, d_model)
        memory = src.permute(1, 0, 2).contiguous()          # (src_len, batch, d_model)

        # Let decoder attend: note we don't pass masks here for simplicity
        decoded = self.transformer_decoder(tgt=tgt, memory=memory)   # (tgt_len, batch, d_model)

        # Post-processing: (tgt_len, batch, d_model) -> (batch, tgt_len, d_model)
        decoded = decoded.permute(1, 0, 2).contiguous()     # (batch, tgt_len, d_model)
        decoded = self.layer_norm(decoded)
        out = self.output_proj(decoded)                     # (batch, tgt_len, d_model)
        return out

def get_inputs():
    """
    Returns:
        [src, tgt_input, h0]
          src: (batch_size, src_len, d_model)
          tgt_input: (batch_size, tgt_len, rnn_input_size)
          h0: (batch_size, rnn_hidden_size)
    """
    src = torch.randn(batch_size, src_len, d_model)
    tgt_input = torch.randn(batch_size, tgt_len, rnn_input_size)
    h0 = torch.zeros(batch_size, rnn_hidden_size)
    return [src, tgt_input, h0]

def get_init_inputs():
    """
    Returns initialization parameters required to instantiate Model in the following order:
      [rnn_input_size, rnn_hidden_size, d_model, nhead, num_decoder_layers]
    """
    return [rnn_input_size, rnn_hidden_size, d_model, nhead, num_decoder_layers]