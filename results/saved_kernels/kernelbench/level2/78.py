import torch
import torch.nn as nn
import triton
import triton.language as tl

# Fused Triton kernel: replaces MaxPool3d(kernel=2)->MaxPool3d(kernel=3) sequence
# by a single max-pooling with kernel=6, stride=6, and simultaneously sums across channels.
# This avoids creating intermediate tensors and an extra pass for the channel-sum,
# reducing memory traffic and improving performance.

@triton.jit
def _fused_pool6_partial_kernel(
    inp_ptr, partial_ptr,
    N, C,
    D_in, H_in, W_in,
    D_out, H_out, W_out,
    stride_d: tl.constexpr, stride_h: tl.constexpr, stride_w: tl.constexpr,
    BLOCK: tl.constexpr, Kd: tl.constexpr, Kh: tl.constexpr, Kw: tl.constexpr, BLOCK_C: tl.constexpr
):
    # 2D grid: axis 0 -> output blocks, axis 1 -> channel blocks (tiles)
    pid_out = tl.program_id(0)
    pid_chunk = tl.program_id(1)

    offs = pid_out * BLOCK + tl.arange(0, BLOCK)  # flattened indices into output (without channel dim)
    n_elements = N * D_out * H_out * W_out
    mask_out = offs < n_elements

    # decode multi-dimensional indices from flattened output index
    tmp = offs
    ow = tmp % W_out
    tmp = tmp // W_out
    oh = tmp % H_out
    tmp = tmp // H_out
    od = tmp % D_out
    n = tmp // D_out  # batch index (vector of length BLOCK)

    # Precompute the spatial base offsets for input for each output element (BLOCK-sized vectors)
    id_base = od * stride_d
    ih_base = oh * stride_h
    iw_base = ow * stride_w

    # partial accumulator for this program (over its BLOCK_C channels)
    partial_sum = tl.zeros((BLOCK,), dtype=tl.float32)

    # Strides to move between channels in flattened memory
    channel_stride = D_in * H_in * W_in
    batch_stride = C * channel_stride
    spatial_stride_h = W_in
    spatial_stride_d = H_in * W_in

    # Channel chunk handled by this program
    c_start = pid_chunk * BLOCK_C
    c_end = min(c_start + BLOCK_C, C)

    # Iterate over channels within this chunk; compute per-channel pooled max and add to partial_sum.
    for c in range(c_start, c_end):
        # per-channel accumulator: initialize to very low value so max works with invalid positions
        m = tl.full((BLOCK,), -1e30, dtype=tl.float32)

        # base offset for this channel per output in the block
        base_chan = n * batch_stride + c * channel_stride

        # iterate over pooling window (Kd,Kh,Kw are constexpr small ints; Triton will unroll)
        for kd in range(Kd):
            id_ = id_base + kd  # vector of length BLOCK
            valid_d = id_ < D_in
            for kh in range(Kh):
                ih = ih_base + kh
                valid_h = ih < H_in
                for kw in range(Kw):
                    iw = iw_base + kw
                    valid_w = iw < W_in

                    # valid positions mask per output slot
                    valid = mask_out & valid_d & valid_h & valid_w

                    # compute flattened input addresses for this channel and these spatial coords
                    addr = base_chan + id_ * spatial_stride_d + ih * spatial_stride_h + iw

                    # load values (use -inf for out-of-bounds so max ignores them)
                    vals = tl.load(inp_ptr + addr, mask=valid, other=-1e30)
                    # update per-channel max
                    m = tl.maximum(m, vals)

        # after scanning window for this channel, add per-channel max to partial sum
        partial_sum = partial_sum + m

    # compute output addresses (flattened for the output space)
    out_addr = n * (D_out * H_out * W_out) + od * (H_out * W_out) + oh * W_out + ow

    # index into the flattened partial tensor arranged as (num_chunks, N*D_out*H_out*W_out)
    n_output = N * D_out * H_out * W_out
    partial_idx = pid_chunk * n_output + out_addr

    # store partial sums (zero where out-of-range)
    vals_to_store = tl.where(mask_out, partial_sum, tl.zeros((BLOCK,), dtype=tl.float32))
    tl.store(partial_ptr + partial_idx, vals_to_store, mask=mask_out)


@triton.jit
def _fused_pool6_final_kernel(
    partial_ptr, out_ptr,
    num_chunks, N,
    D_out, H_out, W_out,
    BLOCK: tl.constexpr
):
    # 1D grid: each program owns a block of outputs and reduces across channel chunks
    pid_out = tl.program_id(0)

    offs = pid_out * BLOCK + tl.arange(0, BLOCK)
    n_elements = N * D_out * H_out * W_out
    mask_out = offs < n_elements

    # decode multi-dimensional indices from flattened output index
    tmp = offs
    ow = tmp % W_out
    tmp = tmp // W_out
    oh = tmp % H_out
    tmp = tmp // H_out
    od = tmp % D_out
    n = tmp // D_out  # batch index

    out_addr = n * (D_out * H_out * W_out) + od * (H_out * W_out) + oh * W_out + ow

    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    # accumulate partial sums across channel chunks
    for k in range(num_chunks):
        idx = k * n_elements + out_addr
        vals = tl.load(partial_ptr + idx, mask=mask_out, other=0.0)
        acc = acc + vals

    # store final reduced sums
    tl.store(out_ptr + out_addr, acc, mask=mask_out)


def triton_fused_pool6_sum(input_tensor: torch.Tensor):
    """
    Performs fused pooling (kernel=6, stride=6) and channel-sum using a two-pass channel-tiling approach.
    Input: (N, C, D_in, H_in, W_in)
    Output: (N, 1, D_out, H_out, W_out)
    """
    assert input_tensor.is_cuda and input_tensor.dtype == torch.float32
    x = input_tensor.contiguous()
    N, C, D_in, H_in, W_in = x.shape

    Kd = Kh = Kw = 6
    sD = sH = sW = 6

    D_out = (D_in - Kd) // sD + 1
    H_out = (H_in - Kh) // sH + 1
    W_out = (W_in - Kw) // sW + 1

    out = torch.empty((N, 1, D_out, H_out, W_out), device=x.device, dtype=x.dtype)

    n_output = N * D_out * H_out * W_out
    if n_output == 0:
        return out

    # Tunable parameters: BLOCK controls output-lane parallelism, BLOCK_C controls channel-tile size
    BLOCK = 128
    BLOCK_C = 16

    num_blocks = (n_output + BLOCK - 1) // BLOCK
    num_chunks = (C + BLOCK_C - 1) // BLOCK_C

    # Intermediate partials: shape (num_chunks, N, D_out, H_out, W_out) flattened by kernel into (num_chunks, n_output)
    partial = torch.empty((num_chunks, N, D_out, H_out, W_out), device=x.device, dtype=x.dtype)
    partial.zero_()  # ensure deterministic values for masked positions

    # launch partial kernel over (output-blocks, channel-chunks)
    grid_partial = (num_blocks, num_chunks)
    _fused_pool6_partial_kernel[grid_partial](
        x, partial,
        N, C,
        D_in, H_in, W_in,
        D_out, H_out, W_out,
        sD, sH, sW,
        BLOCK=BLOCK, Kd=Kd, Kh=Kh, Kw=Kw, BLOCK_C=BLOCK_C
    )

    # launch final kernel that reduces across channel chunks
    grid_final = (num_blocks,)
    _fused_pool6_final_kernel[grid_final](
        partial, out,
        num_chunks, N,
        D_out, H_out, W_out,
        BLOCK=BLOCK
    )
    return out


class ModelNew(nn.Module):
    """
    Optimized model: keeps PyTorch ConvTranspose3d (uses highly-optimized CuDNN/cuBLAS paths),
    replaces the two MaxPool3d calls and the channel-sum with a single fused Triton kernel.
    The two sequential pools (kernel=2, stride=2) then (kernel=3, stride=3) are equivalent
    to a single pool with kernel=6 and stride=6, so we compute that directly and sum channels
    in the same kernel to minimize memory traffic.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        # x: (N, in_channels, D, H, W)
        x = self.conv_transpose(x)                       # use PyTorch's efficient conv_transpose
        # fused pooling + channel sum
        x = triton_fused_pool6_sum(x)
        return x