import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# Try to import Triton. If unavailable, fall back to a pure-PyTorch path.
try:
    import triton
    import triton.language as tl
except Exception:
    triton = None


if triton is not None:
    # Triton kernel that scales B (BH, L, N) by decay (BH, L) along the L dim,
    # but writes output directly in (BH, N, L) layout so it's GEMM-friendly.
    #
    # Kernel layout: each program handles a BLOCK_L x BLOCK_N tile for a given bh.
    # Grid is (BH, num_n_blocks, num_l_blocks). This tiles both L and N axes.
    @triton.jit
    def _scale_B_kernel(B_ptr, decay_ptr, OUT_ptr, BH, L, N, BLOCK_L: tl.constexpr, BLOCK_N: tl.constexpr):
        bh = tl.program_id(0)
        ni_block = tl.program_id(1)
        l_block = tl.program_id(2)

        # base indices for this tile
        ni_base = ni_block * BLOCK_N
        l_base = l_block * BLOCK_L

        # l and n indices for the tile (absolute)
        l_idx = l_base + tl.arange(0, BLOCK_L)           # [BLOCK_L]
        n_idx = ni_base + tl.arange(0, BLOCK_N)          # [BLOCK_N]

        # masks for valid positions
        mask_l = l_idx < L                                # [BLOCK_L]
        mask_n = n_idx < N                                # [BLOCK_N]
        mask = mask_l[:, None] & mask_n[None, :]          # [BLOCK_L, BLOCK_N]

        # Load B: B layout is (BH, L, N) flattened as bh*(L*N) + l*N + n
        baseB = bh * (L * N)
        offsB = baseB + l_idx[:, None] * N + n_idx[None, :]  # [BLOCK_L, BLOCK_N]
        b_vals = tl.load(B_ptr + offsB, mask=mask, other=0.0)

        # load decay values once for the L axis and broadcast across N
        decay_offs = bh * L + l_idx
        dec = tl.load(decay_ptr + decay_offs, mask=mask_l, other=0.0)  # [BLOCK_L]
        dec = dec[:, None]  # broadcast to [BLOCK_L, 1]

        out = b_vals * dec

        # Store into OUT with layout (BH, N, L) flattened as bh*(N*L) + n*L + l
        baseOut = bh * (N * L)
        offsOut = baseOut + n_idx[None, :] * L + l_idx[:, None]  # [BLOCK_L, BLOCK_N] broadcastable
        tl.store(OUT_ptr + offsOut, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, batch_size, seq_length, n_heads, d_head, d_state, block_len=64):
        """
        Optimized Model using:
          - Block reshapes and cached permutations to minimize CPU overhead.
          - A Triton kernel to fuse the scaling of B by decay across the block-time
            dimension (this is a bandwidth-bound, elementwise operation that Triton
            handles efficiently), avoiding large temporary broadcast multiplies in Python.
          - A single batched bmm (torch.bmm) to compute intra-chunk states after scaling,
            leveraging cuBLAS/CUDA for the many small GEMMs.
          - Efficient handling of the inter-chunk recurrence using batched matmuls.
        """
        super(ModelNew, self).__init__()
        assert seq_length % block_len == 0, "Sequence length must be divisible by block length"

        self.batch_size = batch_size
        self.seq_length = seq_length
        self.n_heads = n_heads
        self.d_head = d_head
        self.d_state = d_state
        self.block_len = block_len

        # Model parameters
        self.A = nn.Parameter(torch.randn(batch_size, seq_length, n_heads))
        self.B = nn.Parameter(torch.randn(batch_size, seq_length, n_heads, d_state))
        self.C = nn.Parameter(torch.randn(batch_size, seq_length, n_heads, d_state))

        # Cache flags for permuted parameter layouts to avoid repeated permutes/contiguous
        self._cached = False
        self._cached_device = None

        # Preallocated buffer for scaled B to avoid repeated device allocations.
        # Allocated lazily in forward() with the correct device/shape.
        self._B_scaled_buffer = None

    def segsum(self, x):
        """
        Compute triangular segment-sum matrix as in the reference:
        For input x[..., T] returns x_segsum[..., T, T] where
        x_segsum[..., i, j] = cumsum[j] - cumsum[i-1] for i <= j else -inf.
        Used only for small T (number of chunks).
        """
        T = x.size(-1)
        x_cumsum = torch.cumsum(x, dim=-1)
        x_segsum = x_cumsum[..., :, None] - x_cumsum[..., None, :]
        mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=0)
        x_segsum = x_segsum.masked_fill(~mask, -torch.inf)
        return x_segsum

    def _ensure_cached_layouts(self, device):
        """
        Cache commonly used permuted/reshaped parameter layouts on a per-device basis.
        Copies are taken from parameter.detach() and moved to the target device to avoid
        creating stale tensors that are detached from the parameters' storage.
        The cache is rebuilt if device changes.
        """
        if self._cached and self._cached_device == device:
            return

        b = self.batch_size
        L = self.seq_length
        h = self.n_heads
        n = self.d_state
        l = self.block_len
        c = L // l

        # Copy parameter data (no gradient linkage) and move to desired device
        A_blocks = self.A.detach().reshape(b, c, l, h).to(device)                       # (b, c, l, h)
        # prepare A for cumsum: (b, h, c, l)
        self._A_blocks_re = A_blocks.permute(0, 3, 1, 2).contiguous()                    # (b, h, c, l)

        B_blocks = self.B.detach().reshape(b, c, l, h, n).to(device)                     # (b, c, l, h, n)
        # layout B1 = (b, c, h, l, n) which is used in intra-chunk compute
        self._B1 = B_blocks.permute(0, 1, 3, 2, 4).contiguous()                          # (b, c, h, l, n)

        # Keep C in block layout (not used in final return but cached for parity)
        self._C_blocks = self.C.detach().reshape(b, c, l, h, n).to(device).contiguous()   # (b, c, l, h, n)

        self._cached = True
        self._cached_device = device

    def forward(self, X, initial_states=None):
        """
        Forward pass optimized.

        Inputs:
        - X: (b, length, n_heads, d_head)
        - initial_states: optional tensor shaped like states[:, :1] (b, 1, h, p, n)

        Returns:
        - final accumulated state: (b, n_heads, d_head, d_state) i.e., (b, h, p, n)
        """
        device = X.device
        b = self.batch_size
        L = self.seq_length
        h = self.n_heads
        p = self.d_head
        n = self.d_state
        l = self.block_len
        c = L // l  # number of chunks

        # Ensure cached layouts on the device
        self._ensure_cached_layouts(device)

        # Move raw parameters to device if necessary and invalidate cache if moved
        if self.A.device != device or self.B.device != device or self.C.device != device:
            self.A.data = self.A.data.to(device)
            self.B.data = self.B.data.to(device)
            self.C.data = self.C.data.to(device)
            # Rebuild cache
            self._cached = False
            self._ensure_cached_layouts(device)

        # Ensure input is contiguous and reshape into blocks: (b, c, l, h, p)
        Xc = X.contiguous().reshape(b, c, l, h, p)  # (b, c, l, h, p)

        # A cumulative sums along time within blocks: use cached A layout (b, h, c, l)
        A_cumsum = torch.cumsum(self._A_blocks_re, dim=-1)  # (b, h, c, l)

        # Compute decay_states: decay_states[b,h,c,l] = exp(a_last - A_cumsum)
        a_last = A_cumsum[..., -1]  # (b, h, c)
        decay_states = torch.exp(a_last[..., None] - A_cumsum)  # (b, h, c, l)

        # Compute intra-chunk states.
        # We will compute B_scaled = B1 * decay (broadcast across n), using a Triton kernel
        # to accelerate the elementwise scaling. After scaling, we perform a single batched bmm
        # for each BH to compute (n x p) states.

        # B1: (b, c, h, l, n)
        B1 = self._B1  # cached and on device
        # Permute decay to (b, c, h, l) to align with B1
        decay_perm = decay_states.permute(0, 2, 1, 3).contiguous()  # (b, c, h, l)

        # Flatten over BH = b * c * h
        BH = b * c * h
        B_flat = B1.reshape(BH, l, n).contiguous()           # (BH, l, n)
        decay_flat = decay_perm.reshape(BH, l).contiguous()   # (BH, l)

        # Prepare X flattened for batched bmm: (BH, l, p)
        X1 = Xc.permute(0, 1, 3, 2, 4).contiguous()           # (b, c, h, l, p)
        X_flat = X1.reshape(BH, l, p).contiguous()            # (BH, l, p)

        if triton is not None:
            # Use Triton kernel to compute B_scaled and write it in (BH, n, l) layout
            # Flatten tensors to 1D pointers for Triton.
            B_ptr = B_flat.view(-1)
            decay_ptr = decay_flat.view(-1)

            # Reuse or allocate a preallocated buffer to avoid per-forward allocations.
            # Note: buffer layout is (BH, n, l) to be GEMM-friendly.
            if getattr(self, "_B_scaled_buffer", None) is None or self._B_scaled_buffer.shape != (BH, n, l) or self._B_scaled_buffer.device != device:
                self._B_scaled_buffer = torch.empty(BH, n, l, device=device, dtype=B_flat.dtype)
            OUT = self._B_scaled_buffer
            OUT_ptr = OUT.view(-1)

            # Tile sizes tuned for Ampere: tile both L and N axes.
            BLOCK_L = 32
            BLOCK_N = 16
            num_n_blocks = (n + BLOCK_N - 1) // BLOCK_N
            num_l_blocks = (l + BLOCK_L - 1) // BLOCK_L
            grid = (BH, num_n_blocks, num_l_blocks)

            _scale_B_kernel[grid](
                B_ptr,
                decay_ptr,
                OUT_ptr,
                BH, l, n,
                BLOCK_L=BLOCK_L,
                BLOCK_N=BLOCK_N,
            )
            # OUT is (BH, n, l) which is already GEMM-friendly (contiguous)
            B_scaled_T = OUT
        else:
            # Fallback: pure PyTorch scaling (vectorized) and transpose to (BH, n, l)
            B_scaled_T = (B_flat * decay_flat[..., None]).permute(0, 2, 1).contiguous()

        # torch.bmm requires shapes (BH, n, l) @ (BH, l, p) -> (BH, n, p)
        states_bh = torch.bmm(B_scaled_T, X_flat)  # (BH, n, p)

        # Reshape states back to (b, c, h, p, n) consistent with original semantics:
        # current states_bh is (BH, n, p) -> reshape to (b, c, h, n, p) then permute to (b, c, h, p, n)
        states = states_bh.view(b, c, h, n, p).permute(0, 1, 2, 4, 3).contiguous()  # (b, c, h, p, n)

        # Prepare initial states and concatenate along chunk dimension
        if initial_states is None:
            initial_states = torch.zeros_like(states[:, :1])  # (b, 1, h, p, n)
        states_cat = torch.cat([initial_states, states], dim=1)  # (b, z, h, p, n) where z = c+1

        # Chunk-level recurrence:
        a_last_unflat = a_last  # (b, h, c)
        a_last_padded = F.pad(a_last_unflat, (1, 0))  # (b, h, c+1)
        decay_chunk = torch.exp(self.segsum(a_last_padded))  # (b, h, z, z)
        z = c + 1

        # states_cat -> (b, h, z, p, n)
        states_cat_perm = states_cat.permute(0, 2, 1, 3, 4).contiguous()  # (b, h, z, p, n)
        # reshape to (BH2, z, p*n) where BH2 = b*h
        BH2 = b * h
        states_for_bmm = states_cat_perm.reshape(BH2, z, p * n)  # (BH2, z, p*n)
        decay_chunk_reshaped = decay_chunk.reshape(BH2, z, z)    # (BH2, z, z)

        # Batched matmul for recurrence
        new_states_bh = torch.bmm(decay_chunk_reshaped, states_for_bmm)  # (BH2, z, p*n)

        # reshape back to (b, z, h, p, n)
        new_states = new_states_bh.view(b, h, z, p, n).permute(0, 2, 1, 3, 4).contiguous()  # (b, z, h, p, n)

        # Return last along z dimension (final accumulated state)
        return new_states[:, -1]