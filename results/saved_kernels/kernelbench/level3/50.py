import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

# Lightweight GELU retained for compatibility (not used in fused attention)
class NewGELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

# Triton kernel: apply causal mask + ReLU to a flattened att_all block for many elements per program.
# We choose a larger BLOCK to reduce kernel launch overhead on Ampere GPUs.
@triton.jit
def _mask_relu_tile_kernel(
    att_ptr,        # pointer to att_all (flattened per head)
    T,              # global sequence length (unused in logic but kept for compatibility)
    m_start,        # global start index for rows (queries) for this tile
    k_start,        # global start index for cols (keys) for this tile
    M,              # number of rows in this tile (queries)
    N,              # number of cols in this tile (keys)
    n_elements,     # total elements in this tile = M * N
    stride_head,    # number of elements per head for this tile = M * N (used to index into head)
    BLOCK: tl.constexpr
):
    head_id = tl.program_id(0)    # which head (H = B * n_head)
    block_id = tl.program_id(1)   # which flattened block in the M*N space

    block_start = block_id * BLOCK
    offs = block_start + tl.arange(0, BLOCK)
    mask_offs = offs < n_elements

    # compute local row and col within the tile
    row = offs // N                 # [0..M-1]
    col = offs - row * N           # [0..N-1]

    # compute global indices
    global_row = m_start + row
    global_col = k_start + col

    # element pointers for loading/storing (per head)
    ptrs = att_ptr + head_id * stride_head + offs

    # load values (guarded by mask_offs)
    vals = tl.load(ptrs, mask=mask_offs, other=0.0)

    # causal mask: allow when key index <= query index
    keep_mask = mask_offs & (global_col <= global_row)

    # zero out positions not in causal region then apply ReLU
    vals = tl.where(keep_mask, vals, 0.0)
    vals = tl.where(vals > 0.0, vals, 0.0)

    # store back
    tl.store(ptrs, vals, mask=mask_offs)


def triton_mask_relu_tile(att_block: torch.Tensor, m_start: int, k_start: int):
    """
    In-place causal mask + ReLU for att_block of shape (H, M, N), where H is head-batch (B*n_head).
    This launches a Triton kernel across heads and flattened M*N tiles with a larger BLOCK to reduce launches.
    """
    assert att_block.is_cuda, "att_block must be on CUDA"
    assert att_block.dtype in (torch.float16, torch.float32), "att_block dtype must be float16/float32"
    assert att_block.dim() == 3, "att_block must be 3D: (H, M, N)"

    H, M, N = att_block.shape
    device = att_block.device

    # number of elements per head in this tile
    n_elements = M * N
    # stride in elements to go from head 0 to head 1 in flattened memory
    stride_head = n_elements

    # Choose a larger BLOCK size tuned for Ampere to reduce kernel launch overhead.
    # 4096 is a good tradeoff for big contiguous regions while keeping per-program work reasonable.
    BLOCK = 4096
    num_blocks = (n_elements + BLOCK - 1) // BLOCK

    grid = (H, num_blocks)
    # Launch kernel
    # Note: pass T for compatibility though kernel logic does not use it.
    _mask_relu_tile_kernel[grid](att_block, M + 0, m_start, k_start, M, N, n_elements, stride_head, BLOCK=BLOCK)


class ModelNew(nn.Module):
    """
    Optimized Model using a fused high-throughput strategy:
      - Perform q@k^T and att@v in fp16 to leverage Tensor Cores.
      - Compute the full qk matrix per head-batch in one large bmm to maximize GEMM throughput.
      - Apply causal mask + ReLU in-place using a Triton kernel that processes large contiguous chunks
        (tuned BLOCK to reduce kernel launches).
      - Compute final att@v in fp16 and accumulate in fp32 for numerical stability.
    This reduces memory traffic and kernel-launch overhead compared to repeated small-block approaches.
    """
    def __init__(self, n_embd, n_head, max_seqlen):
        super().__init__()
        assert n_embd % n_head == 0
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        # c_proj kept for API compatibility (not used in this forward)
        self.c_proj = nn.Linear(n_embd, n_embd)
        # causal bias buffer retained for compatibility (not used directly; mask handled by Triton)
        self.register_buffer("bias", torch.tril(torch.ones(max_seqlen, max_seqlen))
                                     .view(1, 1, max_seqlen, max_seqlen))
        self.n_head = n_head
        self.n_embd = n_embd
        self.max_seqlen = max_seqlen
        self.head_dim = n_embd // n_head

    def forward(self, x):
        """
        x: (B, T, C)
        returns: y (B, T, C)
        """
        B, T, C = x.size()
        assert C == self.n_embd, "Input embedding dim must match model configuration"
        assert T <= self.max_seqlen, "Sequence length exceeds maximum sequence length"

        # compute q,k,v
        qkv = self.c_attn(x)  # (B, T, 3*C)
        q, k, v = qkv.split(self.n_embd, dim=2)

        # reshape to (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2).contiguous()
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2).contiguous()
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2).contiguous()

        # Flatten batch and head into H = B * n_head
        H = B * self.n_head
        HS = self.head_dim

        q_view = q.view(H, T, HS)
        k_view = k.view(H, T, HS)
        v_view = v.view(H, T, HS)

        # scale factor
        scale = 1.0 / math.sqrt(HS)

        # Move to fp16 once and pre-scale q to avoid extra multiplies
        # Keep contiguous layout for efficient bmm
        q_half = q_view.to(torch.float16).contiguous() * float(scale)
        k_half = k_view.to(torch.float16).contiguous()
        v_half = v_view.to(torch.float16).contiguous()

        # Accumulator in fp32 for numeric stability
        y = torch.zeros((H, T, HS), dtype=torch.float32, device=q_view.device)

        # Compute attention scores for ALL queries vs ALL keys in one large bmm:
        # att_all: (H, T, T) = q_half @ k_half^T  -> fp16
        # Keep contiguity to ensure efficient memory layout for the Triton kernel
        att_all = torch.bmm(q_half, k_half.transpose(1, 2)).contiguous()  # fp16 (H, T, T)

        # Apply causal mask + ReLU in-place using Triton for efficient blocked elementwise ops.
        # We pass m_start=0 (queries start at 0) and k_start=0 (keys start at 0) for full matrix.
        triton_mask_relu_tile(att_all, 0, 0)

        # Multiply with v_half to get contributions for all queries:
        # contrib: (H, T, HS) = att_all @ v_half   (fp16)
        contrib_fp16 = torch.bmm(att_all, v_half)  # fp16
        # Accumulate in fp32
        y += contrib_fp16.to(torch.float32)

        # reshape back to (B, n_head, T, HS) -> (B, T, C)
        y = y.view(B, self.n_head, T, HS).transpose(1, 2).contiguous().view(B, T, C)

        return y


# Model input helpers (kept for compatibility)
batch_size = 16
max_seqlen = 1024
n_embd = 768
n_head = 12

def get_inputs():
    # Return CUDA inputs to ensure Triton kernels operate on device tensors
    return [torch.rand(batch_size, max_seqlen, n_embd).cuda()]

def get_init_inputs():
    return [n_embd, n_head, max_seqlen]