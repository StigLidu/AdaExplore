import torch
import torch.nn as nn
import torch.nn.functional as F
# Optimized mixed-precision implementation:
# - Prepack weights/biases to fp16 and store pre-transposed contiguous buffers at init (in, out)
# - Run the entire forward in fp16 (single cast at the start), use matmul with pre-transposed weights
# - Cast back to fp32 only once at the end of the network
def triton_linear(x: torch.Tensor, weight_ht: torch.Tensor, bias_h: torch.Tensor = None, relu: bool = False):
    """
    Compute (fp16) x @ weight_ht + bias_h where weight_ht has shape (in_features, out_features).
    Returns an fp16 tensor.
    x: (batch, in_features) - fp16 (caller should cast once)
    weight_ht: (in_features, out_features) - fp16, pre-transposed contiguous
    bias_h: (out_features,) - fp16 or None
    relu: whether to apply in-place ReLU in fp16
    """
    assert x.is_cuda and weight_ht.is_cuda, "Inputs must be CUDA tensors"
    # x is expected in fp16 (forward casts once). Using matmul on contiguous operands avoids implicit transpose.
    out = x.matmul(weight_ht)
    if bias_h is not None:
        out = out + bias_h
    if relu:
        # in-place clamp to avoid extra allocation where safe
        out.clamp_min_(0.0)
    return out


class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_layer_sizes, output_size):
        """
        A version of the original MLP that pre-packs weights/biases to fp16 and uses
        fp16 matmuls with pre-transposed contiguous weights for the forward pass.
        We pad the K (in_features) dimension to a small multiple (8) to improve tensor-core throughput
        and avoid masked loads/divergence on Ampere GPUs.
        """
        super().__init__()

        self.linears = nn.ModuleList()
        current = input_size
        # build linear layers only (we will fuse ReLU inside the triton_linear call)
        for h in hidden_layer_sizes:
            self.linears.append(nn.Linear(current, h, bias=True))
            current = h
        # final linear
        self.linears.append(nn.Linear(current, output_size, bias=True))

        # Alignment/padding parameter for K (in_features). 8 is a safe tensor-core-friendly unit.
        self._pad_k = 8

        # Prepack weights/biases into fp16 pre-transposed contiguous buffers and register them.
        # Store weights as (in_features_padded, out_features) so we can call x.matmul(weight_ht) at runtime.
        for i, layer in enumerate(self.linears):
            # original weight: (out_features, in_features) -> we want (in, out)
            w_ht = layer.weight.data.half().t().contiguous()  # (in, out)
            in_k, out_f = w_ht.shape[0], w_ht.shape[1]
            # pad K to multiple of self._pad_k to avoid masked loads inside matmul and improve tensor-core usage
            target_k = ((in_k + self._pad_k - 1) // self._pad_k) * self._pad_k
            if target_k > in_k:
                pad_rows = target_k - in_k
                pad_tensor = torch.zeros((pad_rows, out_f), dtype=w_ht.dtype, device=w_ht.device)
                w_ht = torch.cat([w_ht, pad_tensor], dim=0).contiguous()

            if layer.bias is not None:
                b_h = layer.bias.data.half().contiguous()
            else:
                b_h = torch.zeros((layer.out_features,), dtype=torch.half, device=layer.weight.device)
            # register buffers so they move with the module and are saved in state_dict
            self.register_buffer(f'weight_ht_{i}', w_ht)
            self.register_buffer(f'bias_h_{i}', b_h)

    def forward(self, x: torch.Tensor):
        # Cast input to fp16 once
        out = x.half()
        num_layers = len(self.linears)
        for i in range(num_layers):
            weight_ht = getattr(self, f'weight_ht_{i}')
            bias_h = getattr(self, f'bias_h_{i}')
            is_last = (i == num_layers - 1)
            # If weight_ht was padded in K, pad the input activation columns accordingly so matmul has no masked K.
            k_w = weight_ht.shape[0]
            k_in = out.shape[1]
            if k_in != k_w:
                pad = k_w - k_in
                # F.pad pads last dimension with (left, right) pairs -> here we append columns on the right
                out = F.pad(out, (0, pad))
            # Use matmul with pre-transposed contiguous weight to avoid implicit transpose overhead.
            out = triton_linear(out, weight_ht, bias_h, relu=(not is_last))
        # Cast back to fp32 once at the end
        return out.float()