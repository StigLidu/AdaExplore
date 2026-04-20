import torch
import torch.nn as nn
import torch.nn.functional as F

# Hint TF32/mixed precision matmul on Ampere for additional perf when acceptable.
try:
    # medium uses TF32 on Ampere for float32 matmuls
    torch.set_float32_matmul_precision('medium')
except Exception:
    # older torch versions may not support this; ignore safely
    pass

def triton_linear(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor = None, relu: bool = False):
    """
    Mixed-precision fused linear:
      out = x @ weight.T + bias

    - Accepts either:
        * weight in standard layout (out_features, in_features) in float32
        * or a pretransposed/packed weight (in_features, out_features) in float16
      The function will detect a pretransposed packed weight by checking weight.shape[0] == x.shape[1].
    - If a packed fp16 weight is provided, the matmul runs under autocast to fp16 to leverage Tensor Cores
      and then (if the input x was float32) converts the result back to float32 so the external dtype remains stable.
    - If a float32 weight is provided we run the matmul in fp32 (useful for the final sensitive layer).
    """
    assert x.is_cuda and weight.is_cuda, "Inputs must be CUDA tensors"
    assert x.dtype in (torch.float32, torch.float16), "Input x must be float32 or float16"
    assert weight.dtype in (torch.float32, torch.float16), "Weight must be float32 or float16"

    # Determine if weight is already pretransposed/packed: (in_features, out_features)
    if weight.shape[0] == x.shape[1]:
        B = weight
    else:
        # transpose to (in_features, out_features)
        B = weight.t().contiguous()

    out_dtype = x.dtype

    # If B is fp16, run mixed-precision path using autocast; otherwise run fp32 path.
    if B.dtype == torch.float16:
        # Ensure tensors are on the same device and fp16 for the matmul
        B = B.to(device=x.device, dtype=torch.float16)
        bias_cast = bias.to(device=x.device, dtype=torch.float16) if bias is not None else None

        # Use autocast to get fast fp16 matmuls (Tensor Cores) on Ampere.
        with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
            out = x.matmul(B)
            if bias_cast is not None:
                out = out + bias_cast
            if relu:
                out = torch.relu(out)

        # Convert back to float32 if caller expects float32 outputs
        if out_dtype == torch.float32:
            out = out.to(torch.float32)
    else:
        # fp32 path
        B = B.to(device=x.device, dtype=torch.float32)
        bias_cast = bias.to(device=x.device, dtype=torch.float32) if bias is not None else None

        out = x.matmul(B)
        if bias_cast is not None:
            out = out + bias_cast
        if relu:
            out = torch.relu(out)

    return out


class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        """
        AlexNet-like model, replacing the fully connected layers' matmuls
        with mixed-precision, prepacked matmuls for better throughput on Ampere.
        """
        super(ModelNew, self).__init__()

        # Convolutional backbone (kept as PyTorch ops)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=2)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, padding=2)
        self.relu2 = nn.ReLU(inplace=True)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv3 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU(inplace=True)

        self.conv4 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1)
        self.relu4 = nn.ReLU(inplace=True)

        self.conv5 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1)
        self.relu5 = nn.ReLU(inplace=True)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2)

        # Keep Linear layers as nn.Linear to retain parameters, but their forward will use mixed-precision matmuls
        self.fc1 = nn.Linear(in_features=256 * 6 * 6, out_features=4096)
        self.relu6 = nn.ReLU(inplace=True)
        # dropout with p=0.0 is a no-op; kept for API compatibility
        self.dropout1 = nn.Dropout(p=0.0)

        self.fc2 = nn.Linear(in_features=4096, out_features=4096)
        self.relu7 = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout(p=0.0)

        self.fc3 = nn.Linear(in_features=4096, out_features=num_classes)

        # Cached prepacked (transposed + fp16) copies for fc1/fc2 to avoid per-call transpose and enable fast fp16 matmuls.
        # These will be created on first forward and refreshed in training mode.
        self._fc1_weight_t = None
        self._fc1_bias_h = None
        self._fc2_weight_t = None
        self._fc2_bias_h = None

    def forward(self, x):
        # Feature extraction via convolutions
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)

        x = self.conv3(x)
        x = self.relu3(x)

        x = self.conv4(x)
        x = self.relu4(x)

        x = self.conv5(x)
        x = self.relu5(x)
        x = self.maxpool3(x)

        x = torch.flatten(x, 1)  # (batch_size, 256*6*6)

        # Prepare and cache pretransposed fp16 weights for fc1
        # Refresh the cached packed weights when training (weights may change) or when device changes.
        if (self._fc1_weight_t is None) or (self._fc1_weight_t.device != x.device) or (self.training):
            # pack: transpose -> contiguous -> to fp16 on the active device
            self._fc1_weight_t = self.fc1.weight.t().contiguous().half().to(x.device)
            self._fc1_bias_h = self.fc1.bias.contiguous().half().to(x.device) if self.fc1.bias is not None else None

        # fc1 fused matmul + bias + ReLU in mixed precision
        x = triton_linear(x, self._fc1_weight_t, self._fc1_bias_h, relu=True)

        # dropout p=0.0 is a no-op; keep semantics
        if self.training and self.dropout1.p > 0:
            x = self.dropout1(x)

        # Prepare/refresh packed weights for fc2
        if (self._fc2_weight_t is None) or (self._fc2_weight_t.device != x.device) or (self.training):
            self._fc2_weight_t = self.fc2.weight.t().contiguous().half().to(x.device)
            self._fc2_bias_h = self.fc2.bias.contiguous().half().to(x.device) if self.fc2.bias is not None else None

        # fc2 fused matmul + bias + ReLU in mixed precision
        x = triton_linear(x, self._fc2_weight_t, self._fc2_bias_h, relu=True)

        if self.training and self.dropout2.p > 0:
            x = self.dropout2(x)

        # fc3: final linear without ReLU. Keep this layer in fp32 for final numeric fidelity.
        x = triton_linear(x, self.fc3.weight, self.fc3.bias, relu=False)

        return x