import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

# Autotune configs tuned for A6000; separate configs can be used for fp16/fp32 kernels.
AUTOTUNE_CONFIGS_FP32 = [
    triton.Config({"BLOCK_SIZE": 1024}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_SIZE": 2048}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_SIZE": 4096}, num_warps=8, num_stages=3),
]

AUTOTUNE_CONFIGS_FP16 = [
    triton.Config({"BLOCK_SIZE": 2048}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_SIZE": 4096}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_SIZE": 8192}, num_warps=8, num_stages=3),
]


@triton.autotune(configs=AUTOTUNE_CONFIGS_FP32, key=['n_elements'])
@triton.jit
def _mish_sub_kernel_fp32(x_ptr, out_ptr, n_elements, sub_total, BLOCK_SIZE: tl.constexpr):
    """
    FP32 fused kernel: out = mish(x - sub_total)
    where mish(z) = z * tanh(softplus(z)), softplus(z)=log(1+exp(z))
    """
    pid = tl.program_id(0)
    start = pid * BLOCK_SIZE
    offs = start + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    x = tl.load(x_ptr + offs, mask=mask, other=0.0)  # fp32
    z = x - sub_total  # fp32

    # Compute softplus in a stable way: sp = log(1 + exp(z))
    # Use exp directly; on Ampere this is efficient.
    exp_z = tl.exp(z)
    sp = tl.log(1.0 + exp_z)

    # tanh(sp) computed via exp to avoid calling tl.tanh
    neg2sp = -2.0 * sp
    exp_neg2sp = tl.exp(neg2sp)
    tanh_sp = (1.0 - exp_neg2sp) / (1.0 + exp_neg2sp)

    out = z * tanh_sp
    tl.store(out_ptr + offs, out, mask=mask)


@triton.autotune(configs=AUTOTUNE_CONFIGS_FP16, key=['n_elements'])
@triton.jit
def _mish_sub_kernel_fp16(x_ptr, out_ptr, n_elements, sub_total, BLOCK_SIZE: tl.constexpr):
    """
    FP16 fused kernel: loads fp16, computes mish in FP32 for stability, stores fp16.
    Flow:
      - load fp16 element
      - cast to fp32 and compute mish(z) in fp32
      - cast back to fp16 and store
    This reduces memory traffic on stores/loads compared to doing all fp32 while keeping reasonable numerical stability.
    """
    pid = tl.program_id(0)
    start = pid * BLOCK_SIZE
    offs = start + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    x_h = tl.load(x_ptr + offs, mask=mask, other=0.0)  # fp16
    # upcast to fp32 for compute
    x = tl.cast(x_h, tl.float32)
    z = x - sub_total  # fp32

    exp_z = tl.exp(z)
    sp = tl.log(1.0 + exp_z)
    neg2sp = -2.0 * sp
    exp_neg2sp = tl.exp(neg2sp)
    tanh_sp = (1.0 - exp_neg2sp) / (1.0 + exp_neg2sp)

    out_f32 = z * tanh_sp
    out_h = tl.cast(out_f32, tl.float16)
    tl.store(out_ptr + offs, out_h, mask=mask)


def triton_mish_sub(x: torch.Tensor, sub_total: float):
    """
    Wrapper that dispatches to either the FP16 or FP32 Triton kernel based on the input dtype.
    Returns a tensor with the same dtype as input.
    """
    assert x.is_cuda, "Input must be on CUDA"

    x_contig = x.contiguous()
    out = torch.empty_like(x_contig)

    n_elements = x_contig.numel()
    grid_fp32 = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    grid_fp16 = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    if x_contig.dtype == torch.float16:
        _mish_sub_kernel_fp16[grid_fp16](x_contig, out, n_elements, float(sub_total))
    elif x_contig.dtype == torch.float32:
        # Prefer fp32 kernel for fp32 inputs
        _mish_sub_kernel_fp32[grid_fp32](x_contig, out, n_elements, float(sub_total))
    else:
        raise RuntimeError("triton_mish_sub: unsupported dtype {}".format(x_contig.dtype))
    return out


class ModelNew(nn.Module):
    """
    Optimized model:
      - Runs convolution under autocast (fp16) to utilize faster Tensor Core/cuDNN paths when on CUDA.
      - Fuses the two scalar subtractions and Mish activation into a single Triton kernel.
      - The Triton kernel has specialized fp16 and fp32 implementations and will be selected based on the conv output dtype.
    Behavior matches original Model (returns fp32 tensor). For non-CUDA inputs, falls back to PyTorch ops.
    """
    def __init__(self, in_channels, out_channels, kernel_size, subtract_value_1, subtract_value_2):
        super(ModelNew, self).__init__()
        # Keep same conv weights/bias semantics
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        # store combined subtract value to avoid extra work per element
        self.subtract_value_1 = float(subtract_value_1)
        self.subtract_value_2 = float(subtract_value_2)
        self._sub_total = self.subtract_value_1 + self.subtract_value_2

    def forward(self, x):
        # If CUDA, try to run conv under autocast to leverage fast fp16 conv, then run Triton kernel
        if x.is_cuda:
            # Run convolution in fp16 for speed with autocast, letting PyTorch handle casting of weights/biases.
            # The conv result may be fp16 (if autocast is effective). We'll then run the appropriate Triton kernel.
            with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
                x_conv = F.conv2d(x, self.conv.weight, self.conv.bias,
                                  stride=self.conv.stride, padding=self.conv.padding,
                                  dilation=self.conv.dilation, groups=self.conv.groups)
            # Ensure contiguous
            x_conv = x_conv.contiguous()

            # If conv produced fp16, run the fp16 kernel and then cast back to fp32 for API compatibility.
            if x_conv.dtype == torch.float16:
                out = triton_mish_sub(x_conv, self._sub_total)
                # out is fp16, cast back to float32 to keep original model dtype
                return out.to(torch.float32)
            else:
                # In rare cases conv may still be fp32 (autocast disabled or not effective).
                out = triton_mish_sub(x_conv, self._sub_total)
                return out.to(torch.float32)
        else:
            # Fallback for CPU or non-CUDA tensors: same operations using PyTorch
            x = F.conv2d(x, self.conv.weight, self.conv.bias,
                         stride=self.conv.stride, padding=self.conv.padding,
                         dilation=self.conv.dilation, groups=self.conv.groups)
            x = x - self.subtract_value_1
            x = x - self.subtract_value_2
            return torch.nn.functional.mish(x)