import torch.nn as nn
import timm

__all__ = [
    "DinoLinear",
    "unfreeze_last_blocks",
]

class DinoLinear(nn.Module):
    """ViT‑S/14 (DINOv2) backbone + single linear classification head"""

    def __init__(self, num_classes: int):
        super().__init__()
        # backbone has no classification head (num_classes=0)
        self.backbone = timm.create_model(
            "vit_small_patch14_dinov2.lvd142m", pretrained=True, num_classes=0, img_size=392,
        )
        self.feat_dim = self.backbone.num_features
        self.head = nn.Linear(self.feat_dim, num_classes)

    def forward(self, x, return_feat: bool = False):
        feats = self.backbone(x)
        logits = self.head(feats)
        if return_feat:
            return feats, logits
        return logits


def unfreeze_last_blocks(model: "DinoLinear", n_blocks: int = 2):
    """Enable gradient updates for the last *n_blocks* transformer blocks"""
    blocks = model.backbone.blocks[-n_blocks:]
    for block in blocks:
        for p in block.parameters():
            p.requires_grad = True