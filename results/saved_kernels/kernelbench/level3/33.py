import torch
import torch.nn as nn
import triton
import triton.language as tl

# Global configuration variables (kept for compatibility with the harness)
batch_size = 256
input_size = 16384
hidden_size = 16384
output_size = 8192
sequence_length = 256

# Autotune configurations for the fused sum+cast+bias+tanh kernel.
# We tile across rows (BLOCK_M) and hidden columns (BLOCK_N).
AUTOTUNE_SUM_TANH = [
    triton.Config({"BLOCK_M": 64,  "BLOCK_N": 256}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 256}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 64,  "BLOCK_N": 512}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 512}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_M": 256, "BLOCK_N": 256}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_M": 256, "BLOCK_N": 512}, num_warps=8, num_stages=3),
]

@triton.autotune(configs=AUTOTUNE_SUM_TANH, key=['M', 'N'])
@triton.jit
def _sum_cast_bias_tanh_kernel(
    S_ptr,        # S_half pointer (fp16) -> (M, N)
    H_ptr,        # H_half pointer (fp16) -> (M, N)
    bias_ptr,     # bias pointer (fp32) -> (N,)
    OUT_ptr,      # OUT pointer (fp32) -> (M, N)
    M, N,
    stride_s0, stride_s1,
    stride_h0, stride_h1,
    stride_out0, stride_out1,
    stride_bias,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
):
    """
    Fuse: OUT = tanh( float32(S_half) + float32(H_half) + bias )
    Processes a BLOCK_M x BLOCK_N tile.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)   # (BLOCK_M,)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)   # (BLOCK_N,)

    # Masks
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    # Compute pointer offsets for S and H loads
    s_ptrs = S_ptr + (offs_m[:, None].to(tl.int32) * stride_s0) + (offs_n[None, :].to(tl.int32) * stride_s1)
    h_ptrs = H_ptr + (offs_m[:, None].to(tl.int32) * stride_h0) + (offs_n[None, :].to(tl.int32) * stride_h1)

    # Load as fp16, cast to fp32 for accumulation
    s_vals = tl.load(s_ptrs, mask=mask, other=tl.zeros((), tl.float16))
    h_vals = tl.load(h_ptrs, mask=mask, other=tl.zeros((), tl.float16))
    s_f32 = tl.cast(s_vals, tl.float32)
    h_f32 = tl.cast(h_vals, tl.float32)

    acc = s_f32 + h_f32  # (BLOCK_M, BLOCK_N)

    # Load bias for columns and broadcast
    bias_vals = tl.load(bias_ptr + offs_n, mask=(offs_n < N), other=0.0)
    acc = acc + bias_vals[None, :]

    # Compute tanh using stable exp formulation: tanh(x) = (e^{2x}-1)/(e^{2x}+1)
    # Avoid overflow by clamping input to a reasonable range for exponent (optional)
    # Use small clamp to keep numerical stability in extreme cases
    # Clamp to [-50, 50] which is safe for exp in fp32
    acc_clamped = tl.maximum(tl.minimum(acc, 50.0), -50.0)
    e2 = tl.exp(acc_clamped * 2.0)
    tanh = (e2 - 1.0) / (e2 + 1.0)

    # Store result to OUT (fp32)
    out_ptrs = OUT_ptr + (offs_m[:, None].to(tl.int32) * stride_out0) + (offs_n[None, :].to(tl.int32) * stride_out1)
    tl.store(out_ptrs, tanh, mask=mask)


def triton_sum_cast_bias_tanh(S_half: torch.Tensor, H_half: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """
    Wrapper around the Triton kernel. Inputs:
      - S_half: (M, N) fp16
      - H_half: (M, N) fp16
      - bias: (N,) fp32
    Returns:
      - hidden_new: (M, N) fp32 = tanh(cast(S_half) + cast(H_half) + bias)
    """
    assert S_half.is_cuda and H_half.is_cuda, "Inputs must be CUDA tensors for Triton kernel."
    assert S_half.shape == H_half.shape, "S_half and H_half must have same shape"
    M, N = S_half.shape

    S_contig = S_half.contiguous()
    H_contig = H_half.contiguous()
    bias_contig = bias.contiguous().to(torch.float32)

    out = torch.empty((M, N), device=S_half.device, dtype=torch.float32)

    # Strides
    stride_s0 = S_contig.stride(0); stride_s1 = S_contig.stride(1)
    stride_h0 = H_contig.stride(0); stride_h1 = H_contig.stride(1)
    stride_out0 = out.stride(0); stride_out1 = out.stride(1)
    stride_bias = bias_contig.stride(0)

    def grid(meta):
        return ((M + meta['BLOCK_M'] - 1) // meta['BLOCK_M'],
                (N + meta['BLOCK_N'] - 1) // meta['BLOCK_N'])

    _sum_cast_bias_tanh_kernel[grid](
        S_contig, H_contig, bias_contig, out,
        M, N,
        stride_s0, stride_s1,
        stride_h0, stride_h1,
        stride_out0, stride_out1,
        stride_bias
    )
    return out


class ModelNew(nn.Module):
    """
    Optimized Vanilla RNN that:
      - Caches fp16 transposed weight buffers for i2h (split as WxT, WhT) and WoT.
      - Performs the two large input->hidden and hidden->hidden matmuls in fp16 (Tensor Cores via cuBLAS).
      - Fuses cast-to-fp32, bias add, and tanh into a Triton kernel to avoid extra memory traffic
        and to overlap cast/add/tanh work on the GPU with better locality.
      - Performs final output projection as an fp16 matmul (Tensor Cores) and casts back to fp32.
    """
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super(ModelNew, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Persistent hidden buffer (fp32)
        self.hidden = torch.randn((batch_size, hidden_size), dtype=torch.float32)

        # Keep Linear modules for parameter/bias storage (compatibility)
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, output_size)

        # Cached transposed fp16 weight buffers for fast matmuls (Tensor Core friendly)
        with torch.no_grad():
            W = self.i2h.weight.detach()  # (hidden_size, input_size + hidden_size)
            W_x = W[:, :input_size]       # (hidden_size, input_size)
            W_h = W[:, input_size:]       # (hidden_size, hidden_size)
            # Store transposed as (K, N_hidden) fp16
            self.register_buffer('i2h_wx_T_h', W_x.t().half().contiguous())  # (input_size, hidden_size)
            self.register_buffer('i2h_wh_T_h', W_h.t().half().contiguous())  # (hidden_size, hidden_size)

            Wo = self.h2o.weight.detach()  # (output_size, hidden_size)
            self.register_buffer('h2o_w_T_h', Wo.t().half().contiguous())   # (hidden_size, output_size)

        # Reusable temporary buffers (lazily allocated per-device/shape)
        self._S_half = None
        self._H_half = None

    def _ensure_tmp_half(self, M: int, N_hidden: int, device: torch.device):
        if self._S_half is None or self._S_half.shape != (M, N_hidden) or self._S_half.device != device:
            self._S_half = torch.empty((M, N_hidden), device=device, dtype=torch.half)
        if self._H_half is None or self._H_half.shape != (M, N_hidden) or self._H_half.device != device:
            self._H_half = torch.empty((M, N_hidden), device=device, dtype=torch.half)

    def forward(self, x: torch.Tensor, initial_hidden=None) -> torch.Tensor:
        """
        Forward pass:
          1. Optionally update persistent hidden from initial_hidden.
          2. Move caches to x.device.
          3. Compute S_half = x.half() @ WxT  and H_half = hidden.half() @ WhT   (fp16 GEMMs).
          4. Fuse cast to fp32 + bias add + tanh via Triton kernel -> hidden_new (fp32).
          5. Compute output = hidden_new.half() @ WoT (fp16 GEMM) and cast to fp32.
        """
        device = x.device

        # Optionally set persistent hidden
        if initial_hidden is not None:
            if initial_hidden.shape != self.hidden.shape or initial_hidden.device != self.hidden.device:
                self.hidden = torch.empty_like(initial_hidden).to(initial_hidden.device)
            self.hidden.copy_(initial_hidden)

        # Move persistent hidden and cached weights to x.device if necessary
        if self.hidden.device != device:
            self.hidden = self.hidden.to(device)
        if self.i2h_wx_T_h.device != device:
            self.i2h_wx_T_h = self.i2h_wx_T_h.to(device)
        if self.i2h_wh_T_h.device != device:
            self.i2h_wh_T_h = self.i2h_wh_T_h.to(device)
        if self.h2o_w_T_h.device != device:
            self.h2o_w_T_h = self.h2o_w_T_h.to(device)

        # Ensure contiguity
        x_contig = x.contiguous()
        h_contig = self.hidden.contiguous()

        M = x_contig.shape[0]
        K1 = x_contig.shape[1]
        K2 = h_contig.shape[1]
        N_hidden = self.hidden_size

        # Prepare reusable fp16 temporaries
        self._ensure_tmp_half(M, N_hidden, device)

        # Convert inputs to fp16 for Tensor Core GEMMs
        x_half = x_contig.half()
        h_half = h_contig.half()

        WxT = self.i2h_wx_T_h  # (K1, N_hidden) fp16
        WhT = self.i2h_wh_T_h  # (K2, N_hidden) fp16

        # Compute the two large GEMMs in fp16 (cuBLAS/Tensor Cores) - these are the heavy ops
        # We keep the results in fp16
        # Note: torch.matmul will allocate temporaries; to reuse memory we copy into preallocated buffers.
        # Compute into temporaries and then copy to our persistent buffers to avoid keeping extra references.
        S_tmp = torch.matmul(x_half, WxT)       # (M, N_hidden) fp16
        H_tmp = torch.matmul(h_half, WhT)       # (M, N_hidden) fp16
        # Copy into preallocated buffers to reduce pressure on allocator and reuse buffers across calls
        self._S_half.copy_(S_tmp)
        self._H_half.copy_(H_tmp)

        # Prepare bias (fp32) on correct device
        bias = self.i2h.bias
        if bias is None:
            bias_t = torch.zeros((N_hidden,), device=device, dtype=torch.float32)
        else:
            bias_t = bias.to(device=device, dtype=torch.float32)

        # Fuse cast to fp32, bias add, and tanh via Triton kernel
        hidden_new = triton_sum_cast_bias_tanh(self._S_half, self._H_half, bias_t)  # (M, N_hidden) fp32

        # Update persistent hidden efficiently
        if hidden_new.shape == self.hidden.shape and hidden_new.device == self.hidden.device and hidden_new.dtype == self.hidden.dtype:
            self.hidden.copy_(hidden_new)
        else:
            self.hidden = hidden_new

        # Final output projection: use fp16 matmul for speed and cast back to fp32
        WoT = self.h2o_w_T_h  # (N_hidden, output_size) fp16
        output_half = torch.matmul(hidden_new.half(), WoT)  # (M, output_size) fp16
        output = output_half.float()

        return output


def get_inputs():
    # For maximum performance we provide CUDA tensors (the harness may also provide CPU tensors).
    return [torch.rand(batch_size, input_size, device='cuda', dtype=torch.float32),
            torch.rand(batch_size, hidden_size, device='cuda', dtype=torch.float32)]

def get_init_inputs():
    return [input_size, hidden_size, output_size]