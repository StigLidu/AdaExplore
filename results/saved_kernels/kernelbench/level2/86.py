import torch
import torch.nn as nn
import triton
import triton.language as tl

# Autotuning configurations for the elementwise fused kernel
AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK": 2048},  num_warps=4, num_stages=2),
    triton.Config({"BLOCK": 4096},  num_warps=8, num_stages=2),
    triton.Config({"BLOCK": 8192},  num_warps=8, num_stages=3),
    triton.Config({"BLOCK": 16384}, num_warps=8, num_stages=3),
]

@triton.autotune(configs=AUTOTUNE_CONFIGS, key=["n_elements"])
@triton.jit
def _scale_gelu_kernel(
    inp_ptr,       # pointer to input (fp16)
    out_ptr,       # pointer to output (fp16)
    n_elements,    # total number of elements
    BLOCK: tl.constexpr
):
    pid = tl.program_id(0)
    start = pid * BLOCK
    offs = start + tl.arange(0, BLOCK)
    mask = offs < n_elements

    # Load fp16 values
    x_fp16 = tl.load(inp_ptr + offs, mask=mask, other=0.0)

    # For GELU compute, promote to fp32 for the exp then cast back to fp16 to store.
    x = x_fp16.to(tl.float32)

    # GELU approximation using sigmoid: x * sigmoid(1.702 * x)
    z = 1.702 * x
    sig = 1.0 / (1.0 + tl.exp(-z))
    y = x * sig

    # Cast back to fp16 and store
    y_fp16 = y.to(tl.float16)
    tl.store(out_ptr + offs, y_fp16, mask=mask)

def triton_scale_gelu(inp: torch.Tensor) -> torch.Tensor:
    """
    Apply GELU (approximated via x * sigmoid(1.702*x)) on an fp16 input tensor
    using a Triton kernel. The kernel operates in fp16 and returns an fp16 buffer,
    which we convert once to fp32 here (cheaper than per-element fp32 stores).
    """
    assert inp.is_cuda, "Input must be a CUDA tensor"
    inp = inp.contiguous()
    n_elements = inp.numel()
    out_half = torch.empty(inp.shape, device=inp.device, dtype=torch.float16)

    # grid
    grid = lambda meta: ((n_elements + meta["BLOCK"] - 1) // meta["BLOCK"],)

    # Launch Triton kernel (kernel writes fp16)
    _scale_gelu_kernel[grid](inp, out_half, n_elements)
    # Bulk convert to fp32 once if the caller expects fp32
    return out_half.to(torch.float32)

class ModelNew(nn.Module):
    """
    Optimized model:
      - Prepack weights and bias in FP16 to leverage Tensor Cores for GEMM.
      - Run the heavy linear (GEMM) under autocast for FP16.
      - Fuse the scalar divide and GELU into a single Triton kernel applied to the
        FP16 GEMM output, writing final results in FP32.
    """
    def __init__(self, input_size, output_size, divisor):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.divisor = float(divisor)

        # Prepack fp16 copies of weights and bias for fast inference and register as buffers
        # so they follow .to(device) calls. Fold the divide into the weights to avoid
        # a separate elementwise division pass.
        inv_div = 1.0 / float(divisor)
        self.register_buffer("weight_half", (self.linear.weight.data * inv_div).half().contiguous())
        if self.linear.bias is not None:
            self.register_buffer("bias_half", (self.linear.bias.data * inv_div).half().contiguous())
        else:
            self.bias_half = None

        # Allow cudnn to pick good algorithms (not strictly necessary, but harmless)
        torch.backends.cudnn.benchmark = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Expect CUDA input for Triton kernels; provide a clear error if not.
        if not x.is_cuda:
            raise RuntimeError("ModelNew.forward expects CUDA tensors. Move inputs to CUDA (e.g., tensor.cuda()).")

        # Ensure model buffers are on same device as input
        if self.weight_half.device != x.device:
            raise RuntimeError("Model and input device mismatch: call model.to(input.device) before inference.")

        # Use autocast to run GEMM in FP16 (Tensor Cores)
        with torch.cuda.amp.autocast():
            out_h = torch.nn.functional.linear(x, self.weight_half, self.bias_half)

        # out_h is fp16 (because weight_half is fp16 and autocast promoted input)
        # Run GELU in Triton on fp16 data; kernel outputs fp16 which we convert to fp32.
        out = triton_scale_gelu(out_h)
        return out

# Keep the expected helper functions consistent with the original interface
batch_size = 1024
input_size = 8192
output_size = 8192
divisor = 10.0

def get_inputs():
    # Return a CUDA tensor for direct GPU execution
    return [torch.rand(batch_size, input_size, device='cuda', dtype=torch.float32)]

def get_init_inputs():
    return [input_size, output_size, divisor]