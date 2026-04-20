import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Complex multimodal fusion model that:
    - Applies circular 2D padding to an image-like tensor.
    - Expands the spatial tensor into a small volumetric tensor and applies 3D adaptive average pooling.
    - Applies 1D adaptive max pooling to a sequence-like tensor.
    - Projects both pooled features into a common embedding space and fuses them for a final prediction.

    The model is intentionally constructed to demonstrate use of:
      - nn.CircularPad2d
      - nn.AdaptiveAvgPool3d
      - nn.AdaptiveMaxPool1d
    along with standard linear projections and activations.
    """
    def __init__(self,
                 img_channels: int,
                 seq_channels: int,
                 pooled_seq_len: int = 32,
                 pooled_depth: int = 2,
                 pooled_h: int = 16,
                 pooled_w: int = 16,
                 hidden_dim: int = 256,
                 out_dim: int = 10):
        super(Model, self).__init__()

        # Padding layer (left, right, top, bottom)
        self.pad = nn.CircularPad2d((1, 2, 1, 2))

        # After padding we will create a small volumetric depth by repeating the spatial plane.
        # AdaptiveAvgPool3d will reduce/reshape the (D, H, W) to (pooled_depth, pooled_h, pooled_w)
        self.avgpool3d = nn.AdaptiveAvgPool3d((pooled_depth, pooled_h, pooled_w))

        # Adaptive max pooling for 1D sequence branch
        self.maxpool1d = nn.AdaptiveMaxPool1d(pooled_seq_len)

        # Projection sizes for fusion
        self.img_feat_dim = img_channels * pooled_depth * pooled_h * pooled_w
        self.seq_feat_dim = seq_channels * pooled_seq_len

        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

        # Linear projections for each modality
        self.img_proj = nn.Linear(self.img_feat_dim, hidden_dim, bias=True)
        self.seq_proj = nn.Linear(self.seq_feat_dim, hidden_dim, bias=True)

        # Fusion and output heads
        self.fusion_fc = nn.Linear(hidden_dim * 2, hidden_dim, bias=True)
        self.out_fc = nn.Linear(hidden_dim, out_dim, bias=True)

        # Non-linearities
        self.act = nn.ReLU()
        self.sig = nn.Sigmoid()

    def forward(self, img: torch.Tensor, seq: torch.Tensor) -> torch.Tensor:
        """
        Forward pass combining an image-like tensor and a sequence-like tensor.

        Args:
            img: Tensor of shape (B, C_img, H, W)
            seq: Tensor of shape (B, C_seq, L)

        Returns:
            out: Tensor of shape (B, out_dim)
        """
        B = img.size(0)

        # 1) Circular pad the image spatially
        padded = self.pad(img)  # -> (B, C_img, H + pad_top + pad_bottom, W + pad_left + pad_right)

        # 2) Make a small volumetric tensor by adding a depth dimension and repeating
        #    depth_repeat >= pooled_depth to allow AdaptiveAvgPool3d to downsample
        depth_repeat = max(4, self.avgpool3d.output_size[0] if hasattr(self.avgpool3d, 'output_size') else 4)
        volumetric = padded.unsqueeze(2).repeat(1, 1, depth_repeat, 1, 1)  # (B, C_img, D_in, H_in, W_in)

        # 3) Adaptive average pool in 3D to obtain fixed (D,H,W)
        pooled3d = self.avgpool3d(volumetric)  # (B, C_img, pooled_depth, pooled_h, pooled_w)

        # 4) Flatten image volumetric features per batch
        img_feat = pooled3d.view(B, -1)  # (B, img_feat_dim)
        img_emb = self.act(self.img_proj(img_feat))  # (B, hidden_dim)

        # 5) Adaptive max pool the sequence to fixed length and flatten
        pooled_seq = self.maxpool1d(seq)  # (B, C_seq, pooled_seq_len)
        seq_feat = pooled_seq.view(B, -1)  # (B, seq_feat_dim)
        seq_emb = self.act(self.seq_proj(seq_feat))  # (B, hidden_dim)

        # 6) Fuse embeddings and produce output
        fused = torch.cat([img_emb, seq_emb], dim=1)  # (B, hidden_dim*2)
        fused_hidden = self.act(self.fusion_fc(fused))  # (B, hidden_dim)

        # 7) Final projection with sigmoid gating to keep outputs bounded
        out = self.out_fc(fused_hidden)  # (B, out_dim)
        out = self.sig(out)

        return out


# Configuration / default sizes
BATCH = 8
IMG_C = 3
H = 64
W = 64

SEQ_C = 4
SEQ_L = 100

POOLED_SEQ_LEN = 32
POOLED_DEPTH = 2
POOLED_H = 16
POOLED_W = 16

HIDDEN_DIM = 256
OUT_DIM = 10

def get_inputs():
    """
    Returns a list with two inputs:
      - img: (BATCH, IMG_C, H, W)
      - seq: (BATCH, SEQ_C, SEQ_L)
    """
    img = torch.randn(BATCH, IMG_C, H, W)
    seq = torch.randn(BATCH, SEQ_C, SEQ_L)
    return [img, seq]

def get_init_inputs():
    """
    Returns the initialization parameters for the Model constructor.
    """
    return [IMG_C, SEQ_C, POOLED_SEQ_LEN, POOLED_DEPTH, POOLED_H, POOLED_W, HIDDEN_DIM, OUT_DIM]