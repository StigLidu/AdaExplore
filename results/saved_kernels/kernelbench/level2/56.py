import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class ModelNew(nn.Module):
    """
    Optimized model that performs block-wise fp16 GEMMs (leveraging Tensor Cores via cuBLAS)
    and uses fused PyTorch elementwise + reduction (sigmoid + sum) on the device to avoid
    Triton atomic overheads. The weight and bias are stored in fp16 for faster matmuls.
    """
    def __init__(self, input_size, hidden_size):
        super(ModelNew, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Store parameters in fp16 to accelerate matmuls via Tensor Cores
        self.weight = nn.Parameter(torch.empty(hidden_size, input_size, dtype=torch.float16))
        self.bias = nn.Parameter(torch.empty(hidden_size, dtype=torch.float16))

        # Initialize with same scheme as nn.Linear (uniform in [-bound, bound]) then cast to fp16
        bound = 1.0 / math.sqrt(input_size)
        tmp_w = torch.empty_like(self.weight, dtype=torch.float32)
        tmp_b = torch.empty_like(self.bias, dtype=torch.float32)
        nn.init.uniform_(tmp_w, -bound, bound)
        nn.init.uniform_(tmp_b, -bound, bound)
        with torch.no_grad():
            self.weight.copy_(tmp_w.to(dtype=torch.float16))
            self.bias.copy_(tmp_b.to(dtype=torch.float16))

    def forward(self, x):
        """
        Forward pass:
          - If input is on CUDA: cast to fp16 and perform blockwise F.linear (cuBLAS/Tensor Cores),
            then convert blocks to fp32, apply sigmoid and sum across output dim to accumulate per-row results.
          - If input is CPU: fallback to exact fp32 computation using torch.nn.functional.linear.
        """
        if not x.is_cuda:
            # CPU fallback: compute in fp32
            y = F.linear(x, self.weight.to(dtype=torch.float32), self.bias.to(dtype=torch.float32))
            y = torch.sigmoid(y)
            y = torch.sum(y, dim=1, keepdim=True)
            return y

        # Enable TF32 for large matmuls on Ampere to accelerate GEMMs (safe tradeoff)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        # Ensure contiguous inputs for efficient GEMMs
        x = x.contiguous()
        # Cast inputs to fp16 to use Tensor Cores
        x_fp16 = x.half()

        w_fp16 = self.weight
        b_fp16 = self.bias

        M = x.shape[0]
        N = self.hidden_size

        # Accumulator in fp32 for numerical stability
        out_acc = torch.zeros(M, device=x.device, dtype=torch.float32)

        # Choose an output-block size to trade off between GEMM efficiency and memory.
        # With hidden_size=32768, block_o=16384 -> 2 large GEMMs which is efficient on Ampere.
        # If memory constrained, reduce to 8192.
        block_o = 16384
        # Ensure block_o divides reasonably; fallback to smaller if it's larger than N
        block_o = min(block_o, N)

        # Perform blockwise GEMMs and accumulate sums of sigmoid across output blocks
        for i in range(0, N, block_o):
            w_block = w_fp16[i:i + block_o]     # shape (block_o, input_size)
            b_block = b_fp16[i:i + block_o]     # shape (block_o,)

            # Compute logits_block = x @ w_block.T + b_block in fp16 (efficient cuBLAS/Tensor Cores)
            # F.linear handles bias addition efficiently.
            logits_block = F.linear(x_fp16, w_block, b_block)  # (M, block_o), fp16

            # Convert to fp32 for more accurate sigmoid and reduction
            logits_block_fp32 = logits_block.float()

            # Apply sigmoid and sum across output dimension for this block, accumulate into out_acc
            # Using PyTorch's fused elementwise + reduction kernels is efficient and avoids atomic contention.
            out_acc.add_(torch.sigmoid(logits_block_fp32).sum(dim=1))

        # Return shape (M, 1) to match original model
        return out_acc.view(M, 1)


# Keep helper functions expected by the harness
batch_size = 128
input_size = 32768
hidden_size = 32768

def get_inputs():
    # Provide CUDA input to exercise the optimized GPU path
    return [torch.rand(batch_size, input_size).cuda()]

def get_init_inputs():
    return [input_size, hidden_size]