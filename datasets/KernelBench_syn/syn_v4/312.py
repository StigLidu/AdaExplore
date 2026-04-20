import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Complex model combining 3D instance normalization (lazy), 2D softmax, 1x1 convolutions,
    and a TransformerDecoderLayer to fuse volumetric and image features.

    Forward pipeline:
    1. Apply LazyInstanceNorm3d to a volumetric input (B, C_vol, D, H, W).
    2. Collapse depth dimension (mean over D) to get (B, C_vol, H, W).
    3. Project volumetric features to d_model channels via 1x1 conv -> create 'memory' for decoder.
    4. Project image input (B, C_img, H, W) to d_model channels via 1x1 conv -> create 'tgt' for decoder.
    5. Flatten spatial dims and run a TransformerDecoderLayer: out_seq = decoder_layer(tgt, memory).
    6. Reshape output to (B, d_model, H, W), apply Softmax2d across channels at each spatial location.
    7. Elementwise-weight the projected image features by the softmax maps.
    8. Global average pool and a final linear layer to produce class logits (B, num_classes).
    """
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float,
        c_vol: int,
        c_img: int,
        num_classes: int
    ):
        super(Model, self).__init__()
        # Lazy InstanceNorm3d: num_features will be inferred on first forward using the volumetric input
        self.inst_norm3d = nn.LazyInstanceNorm3d()
        # 1x1 convs to project channel dimensions to the Transformer model dimension
        self.conv_vol = nn.Conv2d(in_channels=c_vol, out_channels=d_model, kernel_size=1)
        self.conv_img = nn.Conv2d(in_channels=c_img, out_channels=d_model, kernel_size=1)
        # Transformer decoder layer to fuse "tgt" (from image) with "memory" (from volume)
        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='relu'
        )
        # Softmax2d will apply softmax over channel dimension for each spatial location
        self.softmax2d = nn.Softmax2d()
        # Final classification head
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, vol: torch.Tensor, img: torch.Tensor) -> torch.Tensor:
        """
        Args:
            vol: Volumetric input tensor of shape (B, C_vol, D, H, W).
            img: Image input tensor of shape (B, C_img, H, W).

        Returns:
            logits: Tensor of shape (B, num_classes).
        """
        B = vol.size(0)
        # 1) Normalize volumetric features (lazy instance norm will infer channels)
        vol_norm = self.inst_norm3d(vol)  # (B, C_vol, D, H, W)

        # 2) Collapse depth dimension by averaging -> (B, C_vol, H, W)
        vol2d = vol_norm.mean(dim=2)

        # 3) Project volumetric features to d_model channels for 'memory'
        #    Input channels must match c_vol supplied at init
        memory_feats = self.conv_vol(vol2d)  # (B, d_model, H, W)

        # 4) Project image features to d_model channels for 'tgt'
        img_feats = self.conv_img(img)  # (B, d_model, H, W)

        # 5) Flatten spatial dims and permute to (S, B, d_model) for Transformer
        #    where S = H * W
        S = memory_feats.size(2) * memory_feats.size(3)
        memory = memory_feats.flatten(2).permute(2, 0, 1).contiguous()  # (S, B, d_model)
        tgt = img_feats.flatten(2).permute(2, 0, 1).contiguous()        # (S, B, d_model)

        # 6) Transformer decoder layer: fuse tgt with memory
        decoded = self.decoder_layer(tgt=tgt, memory=memory)  # (S, B, d_model)

        # 7) Reshape decoded back to (B, d_model, H, W)
        decoded_img = decoded.permute(1, 2, 0).contiguous().view(B, -1, memory_feats.size(2), memory_feats.size(3))

        # 8) Apply Softmax2d over channels at each spatial location to get attention-like weights
        weights = self.softmax2d(decoded_img)  # (B, d_model, H, W)

        # 9) Weight the image features and pool
        weighted = weights * img_feats  # (B, d_model, H, W)
        pooled = weighted.mean(dim=(2, 3))  # (B, d_model)

        # 10) Final classification logits
        logits = self.fc(pooled)  # (B, num_classes)
        return logits

# Configuration variables
batch_size = 8
C_vol = 16       # channels of volumetric input
D = 8            # depth of volumetric input
H = 32           # spatial height
W = 32           # spatial width
C_img = 3        # channels of image input (e.g., RGB)

d_model = 64
nhead = 8
dim_feedforward = 256
dropout = 0.1
num_classes = 10

def get_inputs():
    """
    Returns:
        List containing:
            - vol: Tensor of shape (batch_size, C_vol, D, H, W)
            - img: Tensor of shape (batch_size, C_img, H, W)
    """
    vol = torch.randn(batch_size, C_vol, D, H, W)
    img = torch.randn(batch_size, C_img, H, W)
    return [vol, img]

def get_init_inputs():
    """
    Returns the initialization parameters for the Model constructor in order:
    [d_model, nhead, dim_feedforward, dropout, C_vol, C_img, num_classes]
    """
    return [d_model, nhead, dim_feedforward, dropout, C_vol, C_img, num_classes]