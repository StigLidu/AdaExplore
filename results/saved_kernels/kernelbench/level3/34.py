import torch
import torch.nn as nn
import triton
import triton.language as tl

# Autotune configurations tuned for A6000-like hardware and the problem sizes used in tests.
AUTOTUNE_CONFIGS = [
    triton.Config(
        {
            "BLOCK_B": 8,
            "BLOCK_H": 128,
            "BLOCK_K": 32,
            "BLOCK_O": 128,
        },
        num_warps=8,
        num_stages=3,
    ),
    triton.Config(
        {
            "BLOCK_B": 8,
            "BLOCK_H": 64,
            "BLOCK_K": 32,
            "BLOCK_O": 128,
        },
        num_warps=8,
        num_stages=3,
    ),
    triton.Config(
        {
            "BLOCK_B": 16,
            "BLOCK_H": 128,
            "BLOCK_K": 32,
            "BLOCK_O": 64,
        },
        num_warps=8,
        num_stages=3,
    ),
    triton.Config(
        {
            "BLOCK_B": 16,
            "BLOCK_H": 64,
            "BLOCK_K": 32,
            "BLOCK_O": 128,
        },
        num_warps=8,
        num_stages=3,
    ),
]


@triton.autotune(configs=AUTOTUNE_CONFIGS, key=['B', 'H', 'O'])
@triton.jit
def _rnn_fused_allsteps_kernel(
    # pointers
    hidden_seq_ptr,  # (T+1, B, H)
    W_h_ptr,         # (H, H) -- this should be W_h.t() as prepared by caller (so layout K,H)
    W_o_ptr,         # (H, O) -- prepared as transposed (H, O)
    input_ptr,       # (T, B, H) precomputed input contributions
    bias_ih_ptr,     # (H,)
    bias_o_ptr,      # (O,)
    out_ptr,         # (T, B, O)
    # sizes
    T, B, H, O,
    # strides (in elements)
    stride_hidden_0, stride_hidden_1, stride_hidden_2,
    stride_Wh_0, stride_Wh_1,
    stride_Wo_0, stride_Wo_1,
    stride_in_0, stride_in_1, stride_in_2,
    stride_out_0, stride_out_1, stride_out_2,
    # constexpr tuning params
    BLOCK_B: tl.constexpr, BLOCK_H: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_O: tl.constexpr
):
    """
    Fused kernel that computes the whole sequence for a tile of (batch rows, output cols).
    Grid layout:
      program_id(0) -> batch-block index
      program_id(1) -> output-block index
    The kernel iterates over timesteps t in [0..T) on-device, reading hidden_seq[t] and
    writing hidden_seq[t+1], and writes outputs out[t] for its output block.
    """

    # Program's batch and output block starts
    row_block = tl.program_id(0)
    out_block = tl.program_id(1)

    row_start = row_block * BLOCK_B
    out_start = out_block * BLOCK_O

    row_offsets = row_start + tl.arange(0, BLOCK_B)            # [BLOCK_B]
    out_offsets = out_start + tl.arange(0, BLOCK_O)            # [BLOCK_O]

    row_mask = row_offsets < B
    out_mask = out_offsets < O

    # Helper ranges for hidden blocks and k-chunks
    h_range = tl.arange(0, BLOCK_H)    # for hidden column blocks
    k_range = tl.arange(0, BLOCK_K)    # for accumulation chunks

    # Local accumulators for outputs per timestep (BLOCK_B, BLOCK_O)
    # We'll reinitialize per timestep
    t = 0
    while t < T:
        # zero accumulator for this timestep & program's output window
        out_acc = tl.zeros((BLOCK_B, BLOCK_O), dtype=tl.float32)

        # For each hidden block compute hidden_new_block and accumulate contribution to out_acc
        for h_start in range(0, H, BLOCK_H):
            h_offsets = h_start + h_range  # [BLOCK_H]
            h_mask = h_offsets < H

            # accumulator for hidden block: shape (BLOCK_B, BLOCK_H)
            hidden_acc = tl.zeros((BLOCK_B, BLOCK_H), dtype=tl.float32)

            # Accumulate A @ W_h_sub over k-chunks: A is hidden_old[t, :, k_offsets]
            for k_start in range(0, H, BLOCK_K):
                k_offsets = k_start + k_range  # [BLOCK_K]
                k_mask = k_offsets < H

                # Load A subblock: hidden_old[t, row_offsets, k_offsets] -> (BLOCK_B, BLOCK_K)
                # base for hidden_old slice t:
                hidden_base = hidden_seq_ptr + t * stride_hidden_0
                A_ptrs = hidden_base + (row_offsets[:, None] * stride_hidden_1 + k_offsets[None, :] * stride_hidden_2)
                a = tl.load(A_ptrs, mask=(row_mask[:, None] & k_mask[None, :]), other=0.0)

                # Load W_h subblock: W_h[k_offsets, h_offsets] -> (BLOCK_K, BLOCK_H)
                Wh_ptrs = W_h_ptr + (k_offsets[:, None] * stride_Wh_0 + h_offsets[None, :] * stride_Wh_1)
                wh = tl.load(Wh_ptrs, mask=(k_mask[:, None] & h_mask[None, :]), other=0.0)

                # Multiply accumulate: (BLOCK_B, K) @ (K, H_block) -> (BLOCK_B, H_block)
                prod = a[:, :, None] * wh[None, :, :]
                hidden_acc += tl.sum(prod, axis=1)

            # Add input contribution and bias: input_ptr[t, row_offsets, h_offsets]
            in_ptrs = input_ptr + (t * stride_in_0) + (row_offsets[:, None] * stride_in_1 + h_offsets[None, :] * stride_in_2)
            input_sub = tl.load(in_ptrs, mask=(row_mask[:, None] & h_mask[None, :]), other=0.0)

            bias_ptrs = bias_ih_ptr + h_offsets  # (BLOCK_H,)
            bias_sub = tl.load(bias_ptrs, mask=h_mask, other=0.0)

            hidden_acc += input_sub + bias_sub[None, :]

            # Apply tanh via stable exp formulation: tanh(x) = (1 - e^{-2x})/(1 + e^{-2x})
            neg2x = -2.0 * hidden_acc
            e = tl.exp(neg2x)
            hidden_new_block = (1.0 - e) / (1.0 + e)  # (BLOCK_B, BLOCK_H)

            # Write hidden_new_block into hidden_seq[t+1, :, h_offsets]
            hidden_out_base = hidden_seq_ptr + (t + 1) * stride_hidden_0
            hidden_out_ptrs = hidden_out_base + (row_offsets[:, None] * stride_hidden_1 + h_offsets[None, :] * stride_hidden_2)
            tl.store(hidden_out_ptrs, hidden_new_block, mask=(row_mask[:, None] & h_mask[None, :]))

            # Compute contribution to outputs for this hidden block and the program's output window
            # Load W_o subblock: W_o[h_offsets, out_offsets] -> (BLOCK_H, BLOCK_O)
            Wo_ptrs = W_o_ptr + (h_offsets[:, None] * stride_Wo_0 + out_offsets[None, :] * stride_Wo_1)
            wo = tl.load(Wo_ptrs, mask=(h_mask[:, None] & out_mask[None, :]), other=0.0)

            # hidden_new_block: (BLOCK_B, BLOCK_H), wo: (BLOCK_H, BLOCK_O)
            prod2 = hidden_new_block[:, :, None] * wo[None, :, :]
            partial = tl.sum(prod2, axis=1)  # (BLOCK_B, BLOCK_O)
            out_acc += partial

        # After accumulating over all hidden blocks, add bias_o and write out[t, row_offsets, out_offsets]
        bias_o_vals = tl.load(bias_o_ptr + out_offsets, mask=out_mask, other=0.0)  # (BLOCK_O,)
        out_acc += bias_o_vals[None, :]

        out_ptrs = out_ptr + (t * stride_out_0) + (row_offsets[:, None] * stride_out_1 + out_offsets[None, :] * stride_out_2)
        tl.store(out_ptrs, out_acc, mask=(row_mask[:, None] & out_mask[None, :]))

        t += 1


def rnn_fused_allsteps(hidden_0: torch.Tensor, W_h_t: torch.Tensor, W_o_t: torch.Tensor,
                       input_contrib: torch.Tensor, bias_i2h: torch.Tensor, bias_h2o: torch.Tensor):
    """
    Wrapper to call the Triton fused all-steps kernel.

    hidden_0: (B, H)
    W_h_t: (H, H)  -- transposed i2h hidden->hidden weight (i.e., original W_h.t())
    W_o_t: (H, O)  -- transposed h2o weight (i.e., h2o.weight.t())
    input_contrib: (T, B, H)
    bias_i2h: (H,)
    bias_h2o: (O,)
    returns: out (T, B, O) and also writes hidden states into a working buffer hidden_seq (not returned)
    """
    assert hidden_0.is_cuda and W_h_t.is_cuda and W_o_t.is_cuda and input_contrib.is_cuda
    T, B_in, H_in = input_contrib.shape
    B, H = hidden_0.shape
    H2, H3 = W_h_t.shape
    H4, O = W_o_t.shape
    assert B == B_in and H == H_in and H == H2 == H3 == H4, "Dimension mismatch"

    device = hidden_0.device
    dtype = hidden_0.dtype

    # Prepare contiguous tensors on device
    W_h_c = W_h_t.contiguous()
    W_o_c = W_o_t.contiguous()
    input_c = input_contrib.contiguous()
    bias_i2h_c = bias_i2h.contiguous()
    bias_h2o_c = bias_h2o.contiguous()

    # Hidden sequence buffer: shape (T+1, B, H)
    hidden_seq = torch.empty((T + 1, B, H), device=device, dtype=dtype)
    hidden_seq[0].copy_(hidden_0)

    out = torch.empty((T, B, O), device=device, dtype=dtype)

    # Strides in elements
    stride_hidden_0 = hidden_seq.stride(0)
    stride_hidden_1 = hidden_seq.stride(1)
    stride_hidden_2 = hidden_seq.stride(2)

    stride_Wh_0 = W_h_c.stride(0)
    stride_Wh_1 = W_h_c.stride(1)
    stride_Wo_0 = W_o_c.stride(0)
    stride_Wo_1 = W_o_c.stride(1)

    stride_in_0 = input_c.stride(0)
    stride_in_1 = input_c.stride(1)
    stride_in_2 = input_c.stride(2)

    stride_out_0 = out.stride(0)
    stride_out_1 = out.stride(1)
    stride_out_2 = out.stride(2)

    # Grid: over batch-blocks and output-blocks
    grid = lambda meta: (
        (B + meta["BLOCK_B"] - 1) // meta["BLOCK_B"],
        (O + meta["BLOCK_O"] - 1) // meta["BLOCK_O"],
    )

    _rnn_fused_allsteps_kernel[grid](
        hidden_seq, W_h_c, W_o_c, input_c, bias_i2h_c, bias_h2o_c, out,
        T, B, H, O,
        stride_hidden_0, stride_hidden_1, stride_hidden_2,
        stride_Wh_0, stride_Wh_1,
        stride_Wo_0, stride_Wo_1,
        stride_in_0, stride_in_1, stride_in_2,
        stride_out_0, stride_out_1, stride_out_2,
    )
    # The kernel filled out and hidden_seq[1..T] with new hidden states.
    return out


class ModelNew(nn.Module):
    """
    Optimized Vanilla RNN using a highly fused Triton kernel that processes the entire
    sequence for blocks of (batch, output) on-GPU in a single kernel launch per grid.

    Optimizations:
      - Precompute input contributions x_t @ W_x^T for all timesteps with a single large matmul.
      - Use a fused Triton kernel that iterates over time on-GPU and for each timestep:
          * computes hidden_new = tanh(hidden_old @ W_h^T + input_contrib + bias)
          * accumulates output = hidden_new @ W_o^T + bias
        This reduces kernel launch overhead and improves data locality by keeping the
        recurrence on device for all timesteps.
    """
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super(ModelNew, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Keep parameterization identical to the reference model
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor, h0: torch.Tensor) -> torch.Tensor:
        """
        x: (seq_len, batch_size, input_size)
        h0: (batch_size, hidden_size)
        returns: (seq_len, batch_size, output_size)
        """
        assert x.dtype == torch.float32 and h0.dtype == torch.float32, "Only fp32 supported"
        seq_len, batch_size, input_dim = x.shape
        assert input_dim == self.input_size, "Input size mismatch"

        device = x.device
        hidden = h0.to(device).contiguous()

        # Split i2h weight into W_x and W_h (i2h.weight shape: (hidden, input+hidden))
        W_i2h = self.i2h.weight  # (hidden, input+hidden)
        W_x = W_i2h[:, :self.input_size]      # (hidden, input)
        W_h = W_i2h[:, self.input_size:]      # (hidden, hidden)

        # Prepare transposed contiguous versions on device for kernel layout:
        # For the kernel we expect W_h_t with layout (K, H) where K iterates in accumulation loops:
        W_h_t = W_h.t().contiguous().to(device)   # (H, H)
        # W_o_t: (H, O)
        W_o_t = self.h2o.weight.t().contiguous().to(device)  # (hidden, output)

        bias_i2h = self.i2h.bias.to(device)
        bias_h2o = self.h2o.bias.to(device)

        # Precompute input contribution for all timesteps in one large GEMM:
        x_flat = x.reshape(seq_len * batch_size, self.input_size).to(device).contiguous()
        W_x_t = W_x.t().contiguous().to(device)  # (input, hidden)
        # Use torch.matmul which is highly optimized for large GEMMs
        input_contrib_flat = torch.matmul(x_flat, W_x_t)  # (seq_len*batch, hidden)
        input_contrib = input_contrib_flat.view(seq_len, batch_size, self.hidden_size).contiguous()

        # Call fused Triton kernel that processes the entire sequence on-GPU
        out = rnn_fused_allsteps(hidden, W_h_t, W_o_t, input_contrib, bias_i2h, bias_h2o)

        return out