import torch
import torch.nn as nn
import triton
import triton.language as tl

# Tunable tile sizes chosen for NVIDIA A6000 (Ampere)
BLOCK_H = 256   # hidden channels per program (tile over H) - larger tile to reduce kernel launch overhead
BLOCK_K = 64    # reduction tile over the H dimension (K)

@triton.jit
def _gru_step_fused_kernel(
    x_ptr,          # pointer to x_proj flattened (SEQ_LEN, B_comb, 3H) in fp16
    W_f_ptr,        # pointer to W_hh forward (3H, H) in fp16
    W_b_ptr,        # pointer to W_hh backward (3H, H) in fp16
    b_f_ptr,        # pointer to b_hh forward (3H,) in fp32
    b_b_ptr,        # pointer to b_hh backward (3H,) in fp32
    h_ptr,          # pointer to initial hidden flattened (B_comb, H) in fp32
    out_ptr,        # pointer to output sequence flattened (SEQ_LEN, B_comb, H) in fp32
    H,              # hidden size
    SEQ_LEN,        # sequence length
    BATCH,          # original batch size (not combined)
    BLOCK_H: tl.constexpr = BLOCK_H,
    BLOCK_K: tl.constexpr = BLOCK_K,
):
    """
    Kernel iterates over timesteps internally and computes the recurrence for one combined-batch entry
    (forward or backward stacked) and one hidden block.
    Grid: (B_combined, num_hidden_blocks)
    """
    pid_b = tl.program_id(0)  # combined batch index (0..2*BATCH-1)
    pid_h = tl.program_id(1)  # hidden-block index

    # determine direction and sample index
    dir_flag = 0
    if pid_b >= BATCH:
        dir_flag = 1
    # sample in original batch
    sample_idx = pid_b - dir_flag * BATCH

    # hidden offsets handled by this program
    offs = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    mask = offs < H

    # pick direction-specific pointers
    W_ptr_dir = W_b_ptr if dir_flag == 1 else W_f_ptr
    b_ptr_dir = b_b_ptr if dir_flag == 1 else b_f_ptr

    # load initial hidden state for this combined sample (fp32)
    h = tl.load(h_ptr + pid_b * H + offs, mask=mask, other=0.0)

    # precompute row bases for W (3H x H) layout
    row_base_r = offs                # 0..H-1
    row_base_z = offs + H            # H..2H-1
    row_base_n = offs + 2 * H        # 2H..3H-1

    k_range = tl.arange(0, BLOCK_K)

    # iterate over sequence timesteps inside kernel to reduce host launches
    for t in range(SEQ_LEN):
        # bases for x and out for this timestep and this combined-batch entry
        x_base = x_ptr + t * (BATCH * 2) * (3 * H) + pid_b * (3 * H)
        out_base = out_ptr + t * (BATCH * 2) * H + pid_b * H

        # accumulators
        acc_r = tl.zeros([BLOCK_H], dtype=tl.float32)
        acc_z = tl.zeros([BLOCK_H], dtype=tl.float32)
        acc_n = tl.zeros([BLOCK_H], dtype=tl.float32)

        # reduction over K (H dimension)
        for k_start in range(0, H, BLOCK_K):
            k_offs = k_start + k_range
            mask_k = k_offs < H

            # load the relevant slice of the current hidden (fp32) for reduction
            # current hidden for reductions is in h buffer (which is kept updated)
            # build pointer to load from global h_ptr for correct values across programs
            # (we use the global h buffer as source of truth)
            h_vals = tl.load(h_ptr + pid_b * H + k_offs, mask=mask_k, other=0.0)
            hv = h_vals[None, :]  # (1, BLOCK_K)

            # addresses for W tiles: rows * H + k_offs
            row_r_exp = row_base_r[:, None] * H
            row_z_exp = row_base_z[:, None] * H
            row_n_exp = row_base_n[:, None] * H
            k_exp = k_offs[None, :]

            mask2d = mask[:, None] & mask_k[None, :]

            # load direction-specific W tiles (stored fp16), cast to fp32 for accumulation
            W_r_fp16 = tl.load(W_ptr_dir + row_r_exp + k_exp, mask=mask2d, other=0.0)
            W_z_fp16 = tl.load(W_ptr_dir + row_z_exp + k_exp, mask=mask2d, other=0.0)
            W_n_fp16 = tl.load(W_ptr_dir + row_n_exp + k_exp, mask=mask2d, other=0.0)

            W_r = tl.cast(W_r_fp16, tl.float32)
            W_z = tl.cast(W_z_fp16, tl.float32)
            W_n = tl.cast(W_n_fp16, tl.float32)

            acc_r += tl.sum(W_r * hv, axis=1)
            acc_z += tl.sum(W_z * hv, axis=1)
            acc_n += tl.sum(W_n * hv, axis=1)

        # load x_proj (fp16) and cast
        x_r_fp16 = tl.load(x_base + offs, mask=mask, other=0.0)
        x_z_fp16 = tl.load(x_base + offs + H, mask=mask, other=0.0)
        x_n_fp16 = tl.load(x_base + offs + 2 * H, mask=mask, other=0.0)

        x_r = tl.cast(x_r_fp16, tl.float32)
        x_z = tl.cast(x_z_fp16, tl.float32)
        x_n = tl.cast(x_n_fp16, tl.float32)

        # load bias for this direction
        b_r = tl.load(b_ptr_dir + offs, mask=mask, other=0.0)
        b_z = tl.load(b_ptr_dir + offs + H, mask=mask, other=0.0)
        b_n = tl.load(b_ptr_dir + offs + 2 * H, mask=mask, other=0.0)

        hh_r = acc_r + b_r
        hh_z = acc_z + b_z
        hh_n = acc_n + b_n

        # gates
        r = 1.0 / (1.0 + tl.exp(-(x_r + hh_r)))
        z = 1.0 / (1.0 + tl.exp(-(x_z + hh_z)))

        tmp = x_n + r * hh_n
        e = tl.exp(-2.0 * tmp)
        n = (1.0 - e) / (1.0 + e)

        # update hidden: use local h (loaded at start and updated in-place)
        h = (1.0 - z) * n + z * h

        # store new hidden to output for this timestep
        tl.store(out_base + offs, h, mask=mask)

        # also write updated h back to global h_ptr so next timestep's reductions see updated values
        tl.store(h_ptr + pid_b * H + offs, h, mask=mask)


def triton_gru_step(x_proj_seq: torch.Tensor, W_hh_f: torch.Tensor, W_hh_b: torch.Tensor, b_hh_f: torch.Tensor, b_hh_b: torch.Tensor, h_prev: torch.Tensor):
    """
    Wrapper to launch the Triton GRU kernel for the full sequence for both directions combined.
    x_proj_seq: (seq_len, B_comb, 3H) in fp16 (combined forward+backward along batch dim)
    W_hh_f, W_hh_b: (3H, H) in fp16
    b_hh_f, b_hh_b: (3H,) in fp32
    h_prev: (B_comb, H) in fp32 (combined forward/backward initial states)
    returns out: (seq_len, B_comb, H) in fp32
    """
    assert x_proj_seq.is_cuda and W_hh_f.is_cuda and W_hh_b.is_cuda and b_hh_f.is_cuda and b_hh_b.is_cuda and h_prev.is_cuda, "All tensors must be CUDA tensors"
    seq_len, B_comb, threeH = x_proj_seq.shape
    H = threeH // 3
    assert W_hh_f.shape == (3 * H, H) and W_hh_b.shape == (3 * H, H)

    # combined batch must be even (forward + backward)
    assert (B_comb % 2) == 0, "Combined batch dimension must be even (forward + backward)."
    BATCH = B_comb // 2  # original (per-direction) batch size

    # convert heavy tensors to efficient device dtypes/layouts
    x_proj_seq_h = x_proj_seq.contiguous().half()       # (seq_len, B_comb, 3H) in fp16
    W_f_h = W_hh_f.contiguous().half()
    W_b_h = W_hh_b.contiguous().half()
    b_f_f32 = b_hh_f.contiguous().float()
    b_b_f32 = b_hh_b.contiguous().float()
    h_prev_f32 = h_prev.contiguous().float()  # (B_comb, H)

    out = torch.empty((seq_len, B_comb, H), device=x_proj_seq.device, dtype=torch.float32)

    num_hidden_blocks = (H + BLOCK_H - 1) // BLOCK_H

    grid = (B_comb, num_hidden_blocks)

    # Launch single kernel which iterates timesteps internally
    # Note: pass the original per-direction batch size (BATCH) to the kernel so it can determine direction
    _gru_step_fused_kernel[grid](
        x_proj_seq_h, W_f_h, W_b_h, b_f_f32, b_b_f32, h_prev_f32, out, H, seq_len, BATCH,
        BLOCK_H=BLOCK_H, BLOCK_K=BLOCK_K
    )

    return out


class ModelNew(nn.Module):
    """
    Optimized bidirectional multi-layer GRU using Triton fused kernel that iterates timesteps internally.
    Key optimizations:
      - Precompute x @ W_ih^T + b_ih for forward and backward directions in fp16 to reduce matmul bandwidth.
      - Run forward and backward directions simultaneously by stacking them in the batch dimension and
        providing small direction-specific W_hh/b vectors to the kernel.
      - Kernel iterates over sequence timesteps internally to minimize host launch overhead.
    """
    def __init__(self, input_size, hidden_size, num_layers=3, bias=True, batch_first=False):
        super(ModelNew, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first

        # Initialize parameters by copying a reference GRU (ensures correct naming/layout)
        _ref = nn.GRU(input_size=input_size,
                      hidden_size=hidden_size,
                      num_layers=num_layers,
                      bias=bias,
                      batch_first=batch_first,
                      bidirectional=True)
        for name, tensor in _ref.state_dict().items():
            if 'weight' in name or 'bias' in name:
                # register as parameter
                self.register_parameter(name, nn.Parameter(tensor.clone()))
            else:
                self.register_buffer(name, tensor.clone())

    def forward(self, x, h0):
        """
        x: (seq_len, batch, input_size) if batch_first=False else (batch, seq_len, input_size)
        h0: (num_layers * 2, batch, hidden_size)
        returns:
          output: (seq_len, batch, 2*hidden_size)
        """
        if self.batch_first:
            x = x.permute(1, 0, 2).contiguous()

        seq_len, batch, _ = x.shape
        device = x.device
        dtype = x.dtype
        H = self.hidden_size
        num_directions = 2  # bidirectional

        # prepare initial hidden states (ensure on same device as input)
        if h0 is None:
            h_prev_all = x.new_zeros((self.num_layers * num_directions, batch, H))
        else:
            # make contiguous and move to the same device as the input to avoid host/device mismatches
            h_prev_all = h0.contiguous().to(device=device)

        layer_input = x  # (seq_len, batch, input_dim)

        for layer in range(self.num_layers):
            if layer == 0:
                input_dim = self.input_size
            else:
                input_dim = H * 2

            # fetch per-direction parameters
            w_ih_fwd = getattr(self, f"weight_ih_l{layer}", None)
            w_hh_fwd = getattr(self, f"weight_hh_l{layer}", None)
            b_ih_fwd = getattr(self, f"bias_ih_l{layer}", None) if self.bias else None
            b_hh_fwd = getattr(self, f"bias_hh_l{layer}", None) if self.bias else None

            w_ih_bwd = getattr(self, f"weight_ih_l{layer}_reverse", None)
            w_hh_bwd = getattr(self, f"weight_hh_l{layer}_reverse", None)
            b_ih_bwd = getattr(self, f"bias_ih_l{layer}_reverse", None) if self.bias else None
            b_hh_bwd = getattr(self, f"bias_hh_l{layer}_reverse", None) if self.bias else None

            # Move weights/bias once to device
            W_ih_f = w_ih_fwd.to(device=device) if w_ih_fwd is not None else None
            W_hh_f = w_hh_fwd.to(device=device) if w_hh_fwd is not None else None
            b_ih_f = b_ih_fwd.to(device=device) if (self.bias and b_ih_fwd is not None) else torch.zeros((3 * H,), device=device, dtype=dtype)
            b_hh_f = b_hh_fwd.to(device=device) if (self.bias and b_hh_fwd is not None) else torch.zeros((3 * H,), device=device, dtype=torch.float32)

            W_ih_b = w_ih_bwd.to(device=device) if (w_ih_bwd is not None) else W_ih_f
            W_hh_b = w_hh_bwd.to(device=device) if (w_hh_bwd is not None) else W_hh_f
            b_ih_b = b_ih_bwd.to(device=device) if (self.bias and b_ih_bwd is not None) else b_ih_f
            b_hh_b = b_hh_bwd.to(device=device) if (self.bias and b_hh_bwd is not None) else b_hh_f

            # Precompute input projections for both directions in fp16 to reduce matmul bandwidth
            inp = layer_input  # (seq_len, batch, input_dim)
            inp_flat = inp.view(seq_len * batch, input_dim)

            # cast inputs and weights to fp16 for matmul (faster on Ampere)
            inp_flat_h = inp_flat.contiguous().half()
            W_ih_f_h = W_ih_f.contiguous().half()
            x_proj_flat_f_h = torch.matmul(inp_flat_h, W_ih_f_h.t())  # (seq*batch, 3H) in fp16
            # add bias in fp16
            b_ih_f_h = b_ih_f.contiguous().half()
            x_proj_flat_f_h = x_proj_flat_f_h + b_ih_f_h.unsqueeze(0)
            x_proj_f = x_proj_flat_f_h.view(seq_len, batch, 3 * H).contiguous()  # fp16

            W_ih_b_h = W_ih_b.contiguous().half()
            x_proj_flat_b_h = torch.matmul(inp_flat_h, W_ih_b_h.t())
            b_ih_b_h = b_ih_b.contiguous().half()
            x_proj_flat_b_h = x_proj_flat_b_h + b_ih_b_h.unsqueeze(0)
            x_proj_b = x_proj_flat_b_h.view(seq_len, batch, 3 * H).contiguous()  # fp16

            # prepare combined x_proj by stacking forward and reversed backward along batch axis
            x_proj_b_rev = x_proj_b.flip(0).contiguous()
            x_proj_combined = torch.cat([x_proj_f, x_proj_b_rev], dim=1).contiguous()  # (seq_len, 2*batch, 3H) in fp16

            # Prepare combined W_hh for both directions as fp16 and biases as fp32
            W_hh_f_h = W_hh_f.contiguous().half()
            W_hh_b_h = W_hh_b.contiguous().half()
            b_hh_f_f32 = b_hh_f.contiguous().float()
            b_hh_b_f32 = b_hh_b.contiguous().float()

            # Prepare combined initial hidden states: forward then backward
            idx_f = layer * num_directions + 0
            idx_b = layer * num_directions + 1
            h_prev_f = h_prev_all[idx_f].contiguous()
            h_prev_b = h_prev_all[idx_b].contiguous()
            # For backward we will feed reversed-time inputs, but the initial hidden ordering aligns with batch stacking
            h_prev_combined = torch.cat([h_prev_f, h_prev_b], dim=0).contiguous()  # (2*batch, H)

            # Call Triton fused kernel that processes both directions combined
            h_seq_combined = triton_gru_step(x_proj_combined, W_hh_f_h, W_hh_b_h, b_hh_f_f32, b_hh_b_f32, h_prev_combined)  # (seq_len, 2*batch, H)

            # Split outputs back into forward and backward (backward outputs are in reversed time order)
            h_seq_f = h_seq_combined[:, 0:batch, :].contiguous()
            h_seq_b_rev = h_seq_combined[:, batch:batch * 2, :].contiguous()
            h_seq_b = h_seq_b_rev.flip(0).contiguous()  # restore time order

            # update final hidden states for next layer / return
            h_prev_all[idx_f] = h_seq_f[-1].contiguous()
            h_prev_all[idx_b] = h_seq_b[-1].contiguous()

            # set layer input to concatenation of forward and backward outputs for next layer
            layer_output = torch.cat([h_seq_f, h_seq_b], dim=2).contiguous()  # (seq_len, batch, 2H)
            layer_input = layer_output

        output = layer_input
        if self.batch_first:
            output = output.permute(1, 0, 2).contiguous()

        return output