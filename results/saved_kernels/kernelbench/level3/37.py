import torch
import torch.nn as nn


class ModelNew(nn.Module):
    """
    Optimized replacement of the original Model that leverages cuDNN's fused LSTM
    while minimizing per-forward overhead.

    Key changes:
      - Cache a single LSTM parameter reference during initialization and refresh it
        only when the module is moved (overrides to / cuda / cpu). This removes
        per-forward next(self.lstm.parameters()) lookups and conditional flattening.
      - Provide fast_forward(...) for inference: wraps the LSTM call in torch.no_grad()
        by default and avoids redundant copies when inputs are already on the correct
        device/dtype/contiguity.
      - Remove unnecessary global backend flag toggles and the unused final linear
        layer to reduce module state and scanning costs.
    The forward() method preserves autograd behavior and returns state[1] as before.
    """
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.0):
        super(ModelNew, self).__init__()
        # Create a single fused LSTM layer stack (num_layers), leveraging cuDNN when available.
        # batch_first=True to match the original Model's API.
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout, bidirectional=False)

        # Cache a single parameter reference and its device/dtype to avoid per-forward lookups.
        self._refresh_lstm_param()

    def _refresh_lstm_param(self):
        # Refresh cached parameter reference and metadata, and flatten parameters once.
        try:
            self._lstm_parm = next(self.lstm.parameters())
            self._lstm_device = self._lstm_parm.device
            self._lstm_dtype = self._lstm_parm.dtype
            try:
                # One-time flatten to prepare cuDNN internal layout when appropriate.
                if self._lstm_device.type == 'cuda':
                    self.lstm.flatten_parameters()
            except Exception:
                pass
        except StopIteration:
            # Defensive: no parameters present
            self._lstm_parm = None
            self._lstm_device = torch.device('cpu')
            self._lstm_dtype = torch.float32

    # Override device-move helpers to refresh cached parameter metadata and flatten once.
    def to(self, *args, **kwargs):
        module = super(ModelNew, self).to(*args, **kwargs)
        self._refresh_lstm_param()
        return module

    def cuda(self, device=None):
        module = super(ModelNew, self).cuda(device)
        self._refresh_lstm_param()
        return module

    def cpu(self):
        module = super(ModelNew, self).cpu()
        self._refresh_lstm_param()
        return module

    def _prepare_tensor(self, t: torch.Tensor, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        # Fast path: if already on correct device/dtype and contiguous, return as-is.
        if t.device == device and t.dtype == dtype and t.is_contiguous():
            return t
        # Otherwise, perform a single-shot conversion and ensure contiguity to avoid temporaries.
        non_blocking = False
        try:
            if t.device.type == 'cpu' and device.type == 'cuda' and t.is_pinned():
                non_blocking = True
        except Exception:
            non_blocking = False
        return t.to(device=device, dtype=dtype, non_blocking=non_blocking).contiguous()

    def forward(self, x, h0, c0):
        """
        Forward pass (keeps autograd):
          x: (batch_size, seq_length, input_size)
          h0, c0: (num_layers, batch_size, hidden_size)
        returns:
          c (num_layers, batch_size, hidden_size)  -- the final cell states (state[1])
        Notes:
          - This forward avoids per-call parameter lookups by relying on cached
            device/dtype values maintained by _refresh_lstm_param (triggered on module moves).
          - Only performs tensor preparation (to/contiguous) when necessary.
        """
        device = self._lstm_device
        dtype = self._lstm_dtype

        # If all tensors already match device/dtype/contiguity, skip preparation.
        if (x.device == device and x.dtype == dtype and x.is_contiguous() and
            h0.device == device and h0.dtype == dtype and h0.is_contiguous() and
            c0.device == device and c0.dtype == dtype and c0.is_contiguous()):
            x_in, h_in, c_in = x, h0, c0
        else:
            x_in = self._prepare_tensor(x, device, dtype)
            h_in = self._prepare_tensor(h0, device, dtype)
            c_in = self._prepare_tensor(c0, device, dtype)

        # Run the fused cuDNN LSTM (keeps autograd)
        _, state = self.lstm(x_in, (h_in, c_in))

        # state is (h_n, c_n); return c_n to match original Model's return of state[1]
        return state[1]

    def fast_forward(self, x, h0, c0, prepared=False, no_grad=True):
        """
        Fast inference path:
          - By default runs under torch.no_grad() to avoid autograd overhead.
          - If prepared=True the caller promises inputs are already on the
            correct device/dtype/contiguous layout and preparation is skipped.
          - Returns state[1] (c_n) like forward().
        """
        device = self._lstm_device
        dtype = self._lstm_dtype

        if prepared:
            x_in, h_in, c_in = x, h0, c0
        else:
            if (x.device == device and x.dtype == dtype and x.is_contiguous() and
                h0.device == device and h0.dtype == dtype and h0.is_contiguous() and
                c0.device == device and c0.dtype == dtype and c0.is_contiguous()):
                x_in, h_in, c_in = x, h0, c0
            else:
                x_in = self._prepare_tensor(x, device, dtype)
                h_in = self._prepare_tensor(h0, device, dtype)
                c_in = self._prepare_tensor(c0, device, dtype)

        if no_grad:
            with torch.no_grad():
                _, state = self.lstm(x_in, (h_in, c_in))
        else:
            _, state = self.lstm(x_in, (h_in, c_in))

        return state[1]