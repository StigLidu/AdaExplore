import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class ModelNew(nn.Module):
    def __init__(self, batch_size, seq_length, n_heads, d_head, d_state, block_len=64):
        """
        Optimized implementation of the provided Model.
        Key optimizations compared to the original:
          - Remove unused diagonal-block output (Y_diag) computation.
          - Fuse decay scaling into B before contraction to avoid extra einsums.
          - Use efficient broadcasting and cumsum-based computations without materializing large intermediate tensors.
        """
        super(ModelNew, self).__init__()

        assert seq_length % block_len == 0, "Sequence length must be divisible by block length"

        self.batch_size = batch_size
        self.seq_length = seq_length
        self.n_heads = n_heads
        self.d_head = d_head
        self.d_state = d_state
        self.block_len = block_len

        # Initialize parameters
        # Keep same parameter shapes as original so this is a drop-in replacement
        self.A = nn.Parameter(torch.randn(batch_size, seq_length, n_heads))
        self.B = nn.Parameter(torch.randn(batch_size, seq_length, n_heads, d_state))
        self.C = nn.Parameter(torch.randn(batch_size, seq_length, n_heads, d_state))

    def forward(self, X, initial_states=None):
        """
        Forward pass:
          - X: (batch, length, n_heads, d_head)
          - Returns final new_states for the entire sequence (same semantic as original).
        """
        # Split into blocks/chunks
        # X_blocks: b, c, l, h, p
        # A_blocks, B_blocks, C_blocks have matching block structure
        X_blocks, A_blocks, B_blocks, C_blocks = [
            rearrange(x, "b (c l) ... -> b c l ...", l=self.block_len)
            for x in (X, self.A, self.B, self.C)
        ]

        # Reorder A to b, h, c, l to compute cumulative sums per-head
        A_blocks = rearrange(A_blocks, "b c l h -> b h c l")
        # cumulative sum along time within each block
        A_cumsum = torch.cumsum(A_blocks, dim=-1)  # shape (b, h, c, l)

        # Compute decay factors for intra-block/state computation:
        # decay_states[b,h,c,l] = exp( A_cumsum[b,h,c,-1] - A_cumsum[b,h,c,l] )
        decay_states = torch.exp(A_cumsum[..., -1:] - A_cumsum)  # shape (b, h, c, l)

        # Prepare decay for elementwise scaling with B_blocks which is (b, c, l, h, n)
        # Rearrange decay to b, c, l, h so it broadcasts to B_blocks
        decay_perm = rearrange(decay_states, "b h c l -> b c l h")

        # Scale B by decay along the time axis (l) to fold the per-time decay into B
        # B_blocks: b c l h n
        B_scaled = B_blocks * decay_perm[..., None]  # b c l h n

        # Compute states by contracting over the time dimension l:
        # states[b, c, h, p, n] = sum_l B_scaled[b,c,l,h,n] * X_blocks[b,c,l,h,p]
        # Use einsum to perform efficient contraction
        states = torch.einsum("bclhn,bclhp->bchpn", B_scaled, X_blocks)  # b c h p n -> note ordering matches original

        # initial_states handling: if not provided, initialize zeros with proper shape
        if initial_states is None:
            # states shape is (b, c, h, p, n) -> create init shape (b, 1, h, p, n)
            init_shape = (states.size(0), 1, states.size(2), states.size(3), states.size(4))
            initial_states = torch.zeros(init_shape, device=states.device, dtype=states.dtype)

        # Prepend initial state along chunk dimension c -> z = c+1
        # states becomes shape (b, z, h, p, n)
        states = torch.cat([initial_states, states], dim=1)

        # Inter-chunk recurrence:
        # Build decay_chunk[b,h,z,c] where z indexes output chunk and c indexes input chunk positions
        # Compute using cumsum on the last values of A_cumsum (per block)
        # last_cumsum: b,h,c
        last_cumsum = A_cumsum[..., -1]  # shape (b, h, c)

        # Pad a zero at the left (representing initial state) -> shape (b, h, c+1)
        # Using explicit concatenation to avoid any ambiguity with F.pad on shapes
        zero_pad = torch.zeros_like(last_cumsum[..., :1])
        padded = torch.cat([zero_pad, last_cumsum], dim=-1)  # b, h, z (z = c+1)

        # Cumulative sum across chunk axis
        padded_cumsum = torch.cumsum(padded, dim=-1)  # b, h, z

        # Compute pairwise differences and mask upper triangle (only keep j <= i)
        # seg[b,h,i,j] = padded_cumsum[b,h,i] - padded_cumsum[b,h,j]
        seg = padded_cumsum[..., :, None] - padded_cumsum[..., None, :]  # b, h, z, z
        z = seg.size(-1)
        mask = torch.tril(torch.ones((z, z), dtype=torch.bool, device=seg.device), diagonal=0)
        seg = seg.masked_fill(~mask, float("-inf"))

        # Exponentiate to get decay_chunk
        decay_chunk = torch.exp(seg)  # b, h, z, z

        # Apply decay_chunk to states: new_states[b, z, h, p, n] = sum_c decay_chunk[b,h,z,c] * states[b,c,h,p,n]
        new_states = torch.einsum("bhzc,bchpn->bzhpn", decay_chunk, states)

        # Return final state's last chunk (the last z)
        return new_states[:, -1]


# Test parameters (kept for compatibility)
batch_size = 2048
seq_length = 128
n_heads = 8
d_head = 64
d_state = 16
block_len = 64

def get_inputs():
    # Return a CUDA tensor ready for benchmarking as in previous examples
    return [torch.rand(batch_size, seq_length, n_heads, d_head).cuda().float()]

def get_init_inputs():
    return [batch_size, seq_length, n_heads, d_head, d_state, block_len]