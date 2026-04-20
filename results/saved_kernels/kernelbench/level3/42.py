import torch
import torch.nn as nn
import triton
import triton.language as tl

# Autotuning configs tuned for Ampere (A6000). Try a few BLOCK sizes and warps/stages.
AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK": 256},  num_warps=2, num_stages=2),
    triton.Config({"BLOCK": 512},  num_warps=4, num_stages=2),
    triton.Config({"BLOCK": 1024}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK": 2048}, num_warps=8, num_stages=3),
]

# Reduction tile size (constexpr). 32 is a good compromise on Ampere for register/shared usage.
KBLOCK = 32


@triton.autotune(configs=AUTOTUNE_CONFIGS, key=['n_elements', 'H'])
@triton.jit
def _gru_fused_step_kernel_transposed(
    gate_x_ptr,      # pointer to gate_x flattened: (batch * 3H)
    w_hh_t_ptr,      # pointer to W_hh transposed flattened: (H * 3H) where w_hh_t[c, r] = W_hh[r, c]
    b_hh_ptr,        # pointer to bias_hh: (3H,)
    h_prev_ptr,      # pointer to h_prev flattened: (batch * H)
    out_ptr,         # pointer to out flattened: (batch * H)
    n_elements,      # total elements = batch * H
    H,               # hidden size
    KBLOCK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """
    Fused GRU step kernel that assumes W_hh has been transposed on the host to shape (H, 3H).
    This layout allows coalesced loads when performing the reduction across columns of W_hh.
    Each program processes BLOCK elements (each element corresponds to a (batch, hidden_index) pair).
    KBLOCK is a constexpr tile size used for the reduction over H.
    """
    start = tl.program_id(0) * BLOCK
    offs = start + tl.arange(0, BLOCK)
    mask = offs < n_elements

    idx = offs  # flattened index over batch * H
    b = idx // H           # batch index (BLOCK,)
    k = idx - b * H        # hidden index within [0, H)

    # gate stride per batch: 3H
    gate_stride = 3 * H
    pos_r = b * gate_stride + k
    pos_z = b * gate_stride + H + k
    pos_n = b * gate_stride + 2 * H + k

    # load precomputed x contributions (x @ W_ih^T + b_ih)
    x_r = tl.load(gate_x_ptr + pos_r, mask=mask, other=0.0)
    x_z = tl.load(gate_x_ptr + pos_z, mask=mask, other=0.0)
    x_n = tl.load(gate_x_ptr + pos_n, mask=mask, other=0.0)

    # accumulators for contributions from h_prev @ W_hh.T for the three gate rows
    acc_r = tl.zeros_like(x_r)
    acc_z = tl.zeros_like(x_z)
    acc_n = tl.zeros_like(x_n)

    # reduction over H dimension in tiles of KBLOCK (KBLOCK is a constexpr kernel arg)
    arange_k = tl.arange(0, KBLOCK)  # constexpr-sized arange
    base_h = b[:, None] * H  # shape (BLOCK, 1)

    # W_hh_t layout: (H, 3H) flattened row-major -> offset = row_idx * (3H) + col_idx
    # row_idx corresponds to original column (c in [0..H)), col_idx corresponds to original row (r in [0..3H))
    for q in range(0, H, KBLOCK):
        valid_k = (q + arange_k) < H  # (KBLOCK,)

        # load h_prev tiles: shape (BLOCK, KBLOCK)
        offs_h = base_h + (q + arange_k)[None, :]  # shape (BLOCK, KBLOCK)
        hv = tl.load(h_prev_ptr + offs_h, mask=mask[:, None] & valid_k[None, :], other=0.0)

        # For each gate row, compute indices into the transposed W_hh_t:
        # We need W_hh[row, col] where row in {k, H+k, 2H+k} and col in {q .. q+KBLOCK-1}
        # In w_hh_t, element is at [col, row] -> flattened offset = col * (3H) + row
        # Compute a (BLOCK, KBLOCK) matrix of offsets for each gate
        # col_base: (1, KBLOCK)
        col_base = (q + arange_k)[None, :] * (3 * H)  # shape (1, KBLOCK)

        row_r = k[:, None]           # shape (BLOCK, 1) -> original row index for r-gate
        row_z = (H + k)[:, None]     # z-gate
        row_n = (2 * H + k)[:, None] # n-gate

        off_r = col_base + row_r     # shape (BLOCK, KBLOCK)
        off_z = col_base + row_z
        off_n = col_base + row_n

        # load corresponding W_hh_t tiles (coalesced by columns)
        wv_r = tl.load(w_hh_t_ptr + off_r, mask=mask[:, None] & valid_k[None, :], other=0.0)
        wv_z = tl.load(w_hh_t_ptr + off_z, mask=mask[:, None] & valid_k[None, :], other=0.0)
        wv_n = tl.load(w_hh_t_ptr + off_n, mask=mask[:, None] & valid_k[None, :], other=0.0)

        # accumulate dot-products along KBLOCK
        acc_r += tl.sum(hv * wv_r, axis=1)
        acc_z += tl.sum(hv * wv_z, axis=1)
        acc_n += tl.sum(hv * wv_n, axis=1)

    # load biases (shared across batch)
    b_idx_r = k
    b_idx_z = H + k
    b_idx_n = 2 * H + k
    b_r = tl.load(b_hh_ptr + b_idx_r, mask=mask, other=0.0)
    b_z = tl.load(b_hh_ptr + b_idx_z, mask=mask, other=0.0)
    b_n = tl.load(b_hh_ptr + b_idx_n, mask=mask, other=0.0)

    # combine contributions
    v_r = x_r + acc_r + b_r
    v_z = x_z + acc_z + b_z

    # sigmoid gates
    r_gate = 1.0 / (1.0 + tl.exp(-v_r))
    z_gate = 1.0 / (1.0 + tl.exp(-v_z))

    # candidate hidden computation
    v_n = x_n + r_gate * (acc_n + b_n)
    s = 1.0 / (1.0 + tl.exp(-2.0 * v_n))
    n_val = 2.0 * s - 1.0

    # load previous hidden
    h_prev_val = tl.load(h_prev_ptr + idx, mask=mask, other=0.0)

    one_minus_z = 1.0 - z_gate
    h_new = one_minus_z * n_val + z_gate * h_prev_val

    tl.store(out_ptr + idx, h_new, mask=mask)


def gru_step_triton_transposed(gate_x: torch.Tensor, w_hh_t: torch.Tensor, b_hh: torch.Tensor, h_prev: torch.Tensor):
    """
    Wrapper for the Triton fused GRU step kernel using transposed W_hh layout.
    Inputs:
      gate_x: (batch, 3H) precomputed x @ W_ih.T + b_ih
      w_hh_t: (H, 3H) which is w_hh.t() on the host
      b_hh: (3H,)
      h_prev: (batch, H)
    Returns:
      h_next: (batch, H)
    """
    assert gate_x.is_cuda and w_hh_t.is_cuda and b_hh.is_cuda and h_prev.is_cuda, "All tensors must be on CUDA."
    assert gate_x.dtype == torch.float32 and w_hh_t.dtype == torch.float32 and b_hh.dtype == torch.float32 and h_prev.dtype == torch.float32

    batch = h_prev.shape[0]
    H = h_prev.shape[1]
    n_elements = batch * H

    gate_x_c = gate_x.contiguous()
    w_hh_t_c = w_hh_t.contiguous()
    b_hh_c = b_hh.contiguous()
    h_prev_c = h_prev.contiguous()

    out = torch.empty_like(h_prev_c)

    gate_x_flat = gate_x_c.view(-1)
    w_hh_t_flat = w_hh_t_c.view(-1)
    b_hh_flat = b_hh_c.view(-1)
    h_prev_flat = h_prev_c.view(-1)
    out_flat = out.view(-1)

    grid = lambda meta: ((n_elements + meta['BLOCK'] - 1) // meta['BLOCK'],)

    _gru_fused_step_kernel_transposed[grid](gate_x_flat, w_hh_t_flat, b_hh_flat, h_prev_flat, out_flat, n_elements, H, KBLOCK=KBLOCK)
    return out


class ModelNew(nn.Module):
    """
    Optimized multi-layer bidirectional GRU using:
      - Precomputation of gate_x = x @ W_ih.T + b_ih via a single GEMM per direction.
      - A Triton fused kernel that computes h_next = GRU_update(...) using W_hh transposed layout
        to enable coalesced memory access during the reduction, improving bandwidth utilization on Ampere.
      - The kernel performs the reduction over H in tiled fashion (KBLOCK), summing contributions for three gates.
    Behavior matches torch.nn.GRU forward for the h_n output.
    """
    def __init__(self, input_size, hidden_size, num_layers=3, bias=True, batch_first=False):
        super(ModelNew, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.num_directions = 2  # bidirectional

        # parameters stored per (layer, direction)
        self.weight_ih = nn.ParameterList()
        self.weight_hh = nn.ParameterList()
        self.bias_ih = nn.ParameterList()
        self.bias_hh = nn.ParameterList()

        for layer in range(num_layers):
            for direction in range(self.num_directions):
                in_dim = input_size if layer == 0 else hidden_size * self.num_directions
                w_ih = nn.Parameter(torch.empty(3 * hidden_size, in_dim, dtype=torch.float32))
                w_hh = nn.Parameter(torch.empty(3 * hidden_size, hidden_size, dtype=torch.float32))
                if bias:
                    b_ih = nn.Parameter(torch.empty(3 * hidden_size, dtype=torch.float32))
                    b_hh = nn.Parameter(torch.empty(3 * hidden_size, dtype=torch.float32))
                else:
                    b_ih = nn.Parameter(torch.zeros(3 * hidden_size, dtype=torch.float32), requires_grad=False)
                    b_hh = nn.Parameter(torch.zeros(3 * hidden_size, dtype=torch.float32), requires_grad=False)

                stdv = 1.0 / (hidden_size ** 0.5)
                for p in (w_ih, w_hh):
                    nn.init.uniform_(p, -stdv, stdv)
                if bias:
                    nn.init.uniform_(b_ih, -stdv, stdv)
                    nn.init.uniform_(b_hh, -stdv, stdv)

                self.weight_ih.append(w_ih)
                self.weight_hh.append(w_hh)
                self.bias_ih.append(b_ih)
                self.bias_hh.append(b_hh)

    def forward(self, x, h0):
        """
        x: (seq_len, batch, input_size) if batch_first=False
        h0: (num_layers * num_directions, batch, hidden_size)
        returns: h_n (num_layers * num_directions, batch, hidden_size)
        """
        if self.batch_first:
            x = x.transpose(0, 1)  # to (seq_len, batch, input_size)

        seq_len, batch_size, _ = x.shape
        H = self.hidden_size
        outputs = x  # input to first layer

        h_n = x.new_zeros((self.num_layers * self.num_directions, batch_size, H), dtype=torch.float32)

        param_idx = 0
        for layer in range(self.num_layers):
            # output buffer for this layer: seq_len x batch x (H * num_directions)
            layer_out = x.new_zeros((seq_len, batch_size, H * self.num_directions), dtype=torch.float32)

            for direction in range(self.num_directions):
                w_ih = self.weight_ih[param_idx]
                w_hh = self.weight_hh[param_idx]
                b_ih = self.bias_ih[param_idx]
                b_hh = self.bias_hh[param_idx]

                # input to this layer
                input_seq = outputs
                in_dim = input_seq.shape[2]

                # Precompute gate_x = x @ W_ih.T + b_ih for all timesteps (seq_len, batch, 3H)
                x_flat = input_seq.contiguous().view(seq_len * batch_size, in_dim)
                gate_x_flat = x_flat.matmul(w_ih.t())  # (seq_len*batch, 3H)
                if self.bias:
                    gate_x_flat = gate_x_flat + b_ih.unsqueeze(0)
                gate_x = gate_x_flat.view(seq_len, batch_size, 3 * H).contiguous()

                # prepare transposed W_hh once on the device to allow coalesced loads in the Triton kernel
                # w_hh: (3H, H) -> transpose to (H, 3H)
                # Move to same device as gate_x for kernel
                device = gate_x.device
                w_hh_dev = w_hh
                b_hh_dev = b_hh
                if w_hh_dev.device != device:
                    w_hh_dev = w_hh_dev.to(device)
                if b_hh_dev.device != device:
                    b_hh_dev = b_hh_dev.to(device)
                w_hh_t = w_hh_dev.t().contiguous()  # (H, 3H)

                # initial hidden for this (layer,direction)
                h_prev = h0[param_idx].contiguous()
                if h_prev.device != device:
                    h_prev = h_prev.to(device)

                # choose time iteration order based on direction
                if direction == 0:
                    time_range = range(0, seq_len)
                else:
                    time_range = range(seq_len - 1, -1, -1)

                # For each timestep, call the fused Triton kernel that computes the GRU update using transposed W_hh
                for t in time_range:
                    gx = gate_x[t]  # (batch, 3H)
                    # ensure tensors are on CUDA (kernel requires CUDA tensors)
                    if gx.is_cuda:
                        h_next = gru_step_triton_transposed(gx, w_hh_t, b_hh_dev, h_prev)
                    else:
                        # fallback CPU path
                        gate_h = h_prev.matmul(w_hh.t())
                        if self.bias:
                            gate_h = gate_h + b_hh.unsqueeze(0)
                        x_r = gx[:, :H]
                        x_z = gx[:, H:2*H]
                        x_n = gx[:, 2*H:3*H]
                        h_r = gate_h[:, :H]
                        h_z = gate_h[:, H:2*H]
                        h_n = gate_h[:, 2*H:3*H]
                        r = torch.sigmoid(x_r + h_r)
                        z = torch.sigmoid(x_z + h_z)
                        n_val = torch.tanh(x_n + r * h_n)
                        h_next = (1 - z) * n_val + z * h_prev

                    layer_out[t, :, direction * H:(direction + 1) * H] = h_next
                    h_prev = h_next

                # store final hidden
                h_n[param_idx, :, :] = h_prev

                param_idx += 1

            outputs = layer_out

        return h_n


# Preserve original test input helpers for API compatibility
batch_size = 10
seq_len = 512
input_size = 128
hidden_size = 256
num_layers = 6

def get_inputs():
    return [torch.rand(seq_len, batch_size, input_size), torch.rand((num_layers*2, batch_size, hidden_size))]

def get_init_inputs():
    return [input_size, hidden_size, num_layers]