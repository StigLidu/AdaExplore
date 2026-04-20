# Copyright 2018 Antoine Miech All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Optimized NetVLAD-style aggregation (ModelNew) with:
 - Mixed-precision GEMMs (FP16) to leverage Tensor Cores.
 - BN folding in eval-mode to avoid an extra BN pass when possible.
 - Cached FP16 copies of cluster parameters and clusters2 to reduce dtype/device churn.
 - Minimal casts and contiguous calls; reduces allocations and memory traffic.
 - Heuristic: for small cluster widths (<=64) rely on PyTorch's highly-optimized softmax,
   otherwise compute softmax in fp32 then cast to fp16 to preserve numerical stability.
This is a lightweight, allocation-conscious rework of the FP16-optimized pipeline.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch as th


class Model(nn.Module):
    def __init__(self, cluster_size, feature_size, ghost_clusters):
        super(Model, self).__init__()

        self.feature_size = feature_size
        self.cluster_size = cluster_size
        self.ghost_clusters = ghost_clusters

        init_sc = (1 / math.sqrt(feature_size))
        clusters = cluster_size + ghost_clusters

        # The `clusters` weights are the `(w,b)` in the paper
        self.clusters = nn.Parameter(init_sc * th.randn(feature_size, clusters))
        self.batch_norm = nn.BatchNorm1d(clusters)
        # The `clusters2` weights are the visual words `c_k` in the paper
        self.clusters2 = nn.Parameter(init_sc * th.randn(1, feature_size, cluster_size))
        self.out_dim = self.cluster_size * feature_size

    def forward(self, x, mask=None):
        max_sample = x.size()[1]
        x = x.view(-1, self.feature_size)  # B x N x D -> BN x D

        if x.device != self.clusters.device:
            msg = f"x.device {x.device} != cluster.device {self.clusters.device}"
            raise ValueError(msg)

        assignment = th.matmul(x, self.clusters)  # (BN x D) x (D x (K+G)) -> BN x (K+G)
        assignment = self.batch_norm(assignment)

        assignment = F.softmax(assignment, dim=1)  # BN x (K+G) -> BN x (K+G)
        # remove ghost assigments
        assignment = assignment[:, :self.cluster_size]
        assignment = assignment.view(-1, max_sample, self.cluster_size)  # -> B x N x K
        a_sum = th.sum(assignment, dim=1, keepdim=True)  # B x 1 x K
        a = a_sum * self.clusters2

        assignment = assignment.transpose(1, 2)  # B x N x K -> B x K x N

        x = x.view(-1, max_sample, self.feature_size)  # BN x D -> B x N x D
        vlad = th.matmul(assignment, x)  # (B x K x N) x (B x N x D) -> B x K x D
        vlad = vlad.transpose(1, 2)  # -> B x D x K
        vlad = vlad - a

        # L2 intra norm
        vlad = F.normalize(vlad)

        # flattening + L2 norm
        vlad = vlad.reshape(-1, self.cluster_size * self.feature_size)  # -> B x DK
        vlad = F.normalize(vlad)
        return vlad  # B x DK


class ModelNew(nn.Module):
    """
    Optimized ModelNew:
     - Uses FP16 matmul/bmm for heavy GEMMs to leverage Tensor Cores.
     - Caches FP16 copies of cluster weights (and fused BN weights when foldable) and clusters2.
     - Minimizes intermediate allocations and unnecessary casts.
     - Uses PyTorch's softmax on small width (<=64) rows (fast on Ampere), otherwise uses stable FP32 softmax then casts to FP16.
    """
    def __init__(self, cluster_size, feature_size, ghost_clusters):
        super(ModelNew, self).__init__()

        self.feature_size = feature_size
        self.cluster_size = cluster_size
        self.ghost_clusters = ghost_clusters

        init_sc = (1 / math.sqrt(feature_size))
        clusters = cluster_size + ghost_clusters

        # The `clusters` weights are the `(w,b)` in the paper (D x (K+G))
        self.clusters = nn.Parameter(init_sc * th.randn(feature_size, clusters))
        self.batch_norm = nn.BatchNorm1d(clusters)
        # The `clusters2` weights are the visual words `c_k` in the paper (1 x D x K)
        self.clusters2 = nn.Parameter(init_sc * th.randn(1, feature_size, cluster_size))
        self.out_dim = self.cluster_size * feature_size

        # Caches to avoid repeated conversions & allocations
        self._clusters_half = None
        self._clusters_half_device = None
        self._clusters_param_ptr = None

        self._fused_clusters_half = None
        self._fused_bias_half = None
        self._fused_for_device = None
        self._batchnorm_state_ptr = None

        self._clusters2_half = None
        self._clusters2_ptr = None

    def _refresh_cluster_half(self, device):
        param_ptr = self.clusters.data_ptr()
        if (self._clusters_half is None or
                self._clusters_half_device != device or
                self._clusters_param_ptr != param_ptr):
            self._clusters_half = self.clusters.detach().to(device=device, non_blocking=True).half().contiguous()
            self._clusters_half_device = device
            self._clusters_param_ptr = param_ptr
            # invalidate fused caches
            self._fused_clusters_half = None
            self._fused_bias_half = None
            self._fused_for_device = None
            self._batchnorm_state_ptr = None

    def _refresh_clusters2_half(self, device):
        clusters2_ptr = self.clusters2.data_ptr()
        if (self._clusters2_half is None or
                self._clusters2_ptr != clusters2_ptr or
                self._clusters_half_device != device):
            # clusters2 shape: (1, D, K)
            self._clusters2_half = self.clusters2.detach().to(device=device, non_blocking=True).half().contiguous()
            self._clusters2_ptr = clusters2_ptr

    def _maybe_build_fused_half(self, device):
        """
        Build fused half clusters and bias when BN folding is available (eval mode + affine + running stats).
        Fused in fp32 for correctness then cast to half for use.
        """
        param_ptr = self.clusters.data_ptr()
        # Only proceed if BN has running stats & affine and we're in eval mode
        if (not self.training) and getattr(self.batch_norm, "track_running_stats", False) and getattr(self.batch_norm, "affine", False):
            bn_state_ptr = (self.batch_norm.running_mean.data_ptr() if self.batch_norm.running_mean is not None else 0,
                            self.batch_norm.running_var.data_ptr() if self.batch_norm.running_var is not None else 0,
                            self.batch_norm.weight.data_ptr() if self.batch_norm.weight is not None else 0,
                            self.batch_norm.bias.data_ptr() if self.batch_norm.bias is not None else 0)
            needs_refresh = (self._fused_clusters_half is None or
                             self._fused_for_device != device or
                             self._batchnorm_state_ptr != bn_state_ptr or
                             self._clusters_param_ptr != param_ptr)
            if needs_refresh:
                gamma = self.batch_norm.weight
                beta = self.batch_norm.bias
                running_mean = self.batch_norm.running_mean
                running_var = self.batch_norm.running_var
                eps = self.batch_norm.eps
                # compute fused in fp32 for numerical correctness then cast to half
                scale = (gamma / torch.sqrt(running_var + eps)).to(device)
                bias_fp32 = (-running_mean * scale + beta).to(device).to(torch.float32)
                clusters_fused_fp32 = (self.clusters.detach() * scale.unsqueeze(0)).to(device)
                self._fused_clusters_half = clusters_fused_fp32.half().contiguous()
                self._fused_bias_half = bias_fp32.half().contiguous()
                self._fused_for_device = device
                self._batchnorm_state_ptr = bn_state_ptr
        else:
            # Clear fused if not applicable
            self._fused_clusters_half = None
            self._fused_bias_half = None
            self._fused_for_device = None
            self._batchnorm_state_ptr = None

    def forward(self, x, mask=None):
        """
        Optimized forward pass:
          - Cast input to half once and reuse views.
          - Compute logits via fp16 matmul (using cached half clusters).
          - If BN folding available, add fused bias in half; otherwise apply BN in fp32 after upcasting.
          - Use fast softmax heuristics: for width <=64 use PyTorch softmax; otherwise compute softmax in fp32 then cast to half.
          - Keep only real clusters, compute a and vlad in mixed precision minimizing casts/allocations.
        """
        B = x.size(0)
        N = x.size(1)
        D = self.feature_size
        K = self.cluster_size
        G = self.ghost_clusters
        K_total = K + G
        max_sample = N

        # Ensure contiguous input for efficient views/casts
        x_batched = x.contiguous()
        device = x_batched.device

        if x_batched.device != self.clusters.device and self.clusters.device.type != 'cpu':
            # clusters will be moved (cached) to match input device below
            pass

        # Refresh cached half parameters
        self._refresh_cluster_half(device)
        self._maybe_build_fused_half(device)
        self._refresh_clusters2_half(device)

        # Cast input to half once
        x_batched_half = x_batched.half()
        x_flat_half = x_batched_half.view(-1, D)  # (B*N, D)

        # choose clusters source (fused if available)
        clusters_half_to_use = self._fused_clusters_half if (self._fused_clusters_half is not None) else self._clusters_half
        bias_half = (self._fused_bias_half if (self._fused_bias_half is not None) else None)

        # 1) FP16 matmul for logits
        # (B*N, D) @ (D, K_total) -> (B*N, K_total) in half
        # using @ (matmul) which benefits from Tensor Cores when inputs are half and CUDA AMP enabled / GPU supports it.
        logits_fp16 = torch.matmul(x_flat_half, clusters_half_to_use)  # (BN, K_total) half

        # 2) Apply BN / bias and compute softmax probabilities
        # Heuristic: for K_total <= 64 PyTorch's softmax is very efficient on Ampere.
        if bias_half is not None:
            # folded case: just add bias in half
            logits_fp16 = logits_fp16 + bias_half.unsqueeze(0)
            # softmax: use PyTorch directly for small widths (typical here), otherwise stable fp32 softmax then cast
            if K_total <= 64:
                assignment_full_half = F.softmax(logits_fp16, dim=1).half()
            else:
                # convert to fp32 for stable softmax then cast back
                assignment_full = F.softmax(logits_fp16.float(), dim=1)
                assignment_full_half = assignment_full.half()
        else:
            # Need to apply BatchNorm in fp32 for correctness: convert then apply BN
            logits_fp32 = logits_fp16.float()
            logits_bn = self.batch_norm(logits_fp32)  # (BN, K_total) fp32
            # Softmax in fp32 for numerical stability, then cast to half
            if K_total <= 64:
                # PyTorch softmax on fp32 is still fast; cast to half afterwards
                assignment_full_half = F.softmax(logits_bn, dim=1).half()
            else:
                assignment_full = F.softmax(logits_bn, dim=1)
                assignment_full_half = assignment_full.half()

        # 3) Keep only real clusters and reshape to (B, N, K)
        # assignment_full_half: (BN, K_total) half
        assignment_k = assignment_full_half[:, :K].view(B, max_sample, K)  # B x N x K (half)

        # 4) Compute a_sum and a = a_sum * clusters2
        # sum in half to avoid upcasting when possible
        # assignment_k is half, so sum will be half as well
        a_sum_half = assignment_k.sum(dim=1, keepdim=True)  # B x 1 x K (half)
        # clusters2_half is (1, D, K); broadcast multiplication -> (B, D, K) half
        a_half = a_sum_half * self._clusters2_half  # keep in half until final conversion

        # 5) Compute vlad via batched bmm in half precision
        # assignment_t: B x K x N, x_batched_half: B x N x D -> result B x K x D (half)
        assignment_t = assignment_k.permute(0, 2, 1).contiguous()
        vlad_bk_d_half = torch.bmm(assignment_t, x_batched_half)  # B x K x D (half)

        # Keep computations in fp16 as long as possible:
        # transpose to B x D x K in half, subtract a_half (also half), then upcast before normalization.
        vlad = vlad_bk_d_half.transpose(1, 2).contiguous()  # B x D x K (half)
        vlad = vlad - a_half  # B x D x K (half)

        # Upcast to fp32 for numerically stable normalization
        vlad = vlad.float()

        # Intra-normalize across D
        vlad = F.normalize(vlad, p=2, dim=1)

        # Flatten and final L2 normalization
        vlad = vlad.reshape(-1, K * D)
        vlad = F.normalize(vlad, p=2, dim=1)
        return vlad


# Example input sizes preserved for compatibility with the harness
batch_size = 2048
num_features = 100
num_clusters = 32
feature_size = 512
ghost_clusters = 16

def get_inputs():
    # Prefer CUDA for best performance (Tensor Cores & mixed precision)
    return [torch.rand(batch_size, num_features, feature_size).cuda()]

def get_init_inputs():
    return [num_clusters, feature_size, ghost_clusters]