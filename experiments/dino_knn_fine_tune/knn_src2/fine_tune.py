import os
import json
from pathlib import Path
from typing import List, Tuple, Dict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from tqdm import tqdm

from PIL import Image

from .knn_utils import get_transform, load_dataset

class PillDataset(Dataset):
    """Simple ImageFolder‑like dataset using the existing load_dataset helper."""

    def __init__(self, root_dir: str, label2idx: Dict[str, int], transform):
        self.paths, raw_labels = load_dataset(root_dir)
        self.labels = [label2idx[l] for l in raw_labels]
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        x = self.transform(img)
        y = self.labels[idx]
        return x, y

class FineTuneModel(nn.Module):
    """ViT‑S/16 (DINO) backbone with a linear classifier.

    * Only the **last 4 Transformer blocks** + classifier are unfrozen, giving
      a light yet effective domain adaptation (≈3–4 M trainable params).
    * Embedding dimension is taken from the backbone to avoid hard‑coding.
    """

    def __init__(self, num_classes: int, unfreeze_blocks: int = 4):
        super().__init__()

        # Load ViT‑S/16 backbone (DINO SSL weights)
        self.backbone: nn.Module = torch.hub.load(
            "facebookresearch/dino:main", "dino_vits16", pretrained=True
        )
        embed_dim: int = getattr(self.backbone, "embed_dim", 384)

        # Classification head
        self.classifier = nn.Linear(embed_dim, num_classes)

        # Freeze entire backbone then unfreeze last N blocks
        for p in self.backbone.parameters():
            p.requires_grad_(False)

        # Transformer blocks are stored in self.backbone.blocks (nn.ModuleList)
        if unfreeze_blocks > 0:
            for blk in self.backbone.blocks[-unfreeze_blocks:]:
                for p in blk.parameters():
                    p.requires_grad_(True)
        # Always unfreeze final norm layer (helps fine‑tune CLS token)
        for p in self.backbone.norm.parameters():
            p.requires_grad_(True)

    def forward(self, x):
        # backbone returns a pooled 384‑d feature
        feat = self.backbone(x)
        feat = nn.functional.normalize(feat, dim=1)  # cosine‑friendly
        return self.classifier(feat), feat

@torch.inference_mode()
def eval_accuracy(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    correct = 0
    total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits, _ = model(x)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / total


def build_sampler(labels: List[int]):
    """Balance class sampling via inverse frequency weights."""
    from collections import Counter

    cnt = Counter(labels)
    class_weights = {c: 1.0 / f for c, f in cnt.items()}
    weights = [class_weights[l] for l in labels]
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

def train(train_dir: str, val_dir: str, epochs: int = 3, batch_size: int = 64, lr: float = 5e-5):
    """Light fine‑tuning with frozen backbone except last 4 blocks."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------------- label mapping ----------------
    _, train_labels_raw = load_dataset(train_dir)
    classes = sorted(set(train_labels_raw))
    label2idx = {c: i for i, c in enumerate(classes)}

    # ---------------- datasets / loaders ----------------
    transform = get_transform(224)
    train_ds = PillDataset(train_dir, label2idx, transform)
    val_ds = PillDataset(val_dir, label2idx, transform)

    sampler = build_sampler(train_ds.labels)
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=16, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # ---------------- model / criterion / optim ----------------
    model = FineTuneModel(num_classes=len(classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc = 0.0
    ckpt_dir = Path("embed_checkpoints"); ckpt_dir.mkdir(exist_ok=True)

    for ep in range(1, epochs + 1):
        # ---------- train ----------
        model.train()
        running_loss = 0.0
        bar = tqdm(train_loader, leave=False)
        for x, y in bar:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad()
            logits, _ = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * x.size(0)
        scheduler.step()
        train_loss = running_loss / len(train_loader.dataset)

        # ---------- validate ----------
        model.eval(); correct = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                logits, _ = model(x)
                preds = torch.argmax(logits, dim=1)
                correct += (preds == y).sum().item()
        val_acc = correct / len(val_loader.dataset)

        print(f"Epoch {ep}/{epochs}  train_loss={train_loss:.4f}  val_acc={val_acc:.4f}")

        # save best ckpt
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            ckpt_path = ckpt_dir / f"dino_vits16_light_ep{ep}_acc{val_acc:.3f}.pth"
            torch.save(model.state_dict(), ckpt_path)
            print(f"[✓] New best saved → {ckpt_path}")

    print(f"Best val acc: {best_val_acc:.4f}")

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--train", default="old_dataset/train")
    p.add_argument("--val", default="old_dataset/val")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--bs", type=int, default=64)
    p.add_argument("--lr", type=float, default=5e-4)
    args = p.parse_args()

    train(args.train, args.val, epochs=args.epochs, batch_size=args.bs, lr=args.lr)