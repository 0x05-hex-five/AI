import os
import pickle
import argparse
from pathlib import Path
from typing import List

import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from knn_utils import get_transform, load_dataset
from fine_tune import FineTuneModel


class EmbeddingDataset(Dataset):
    """
    Dataset wrapping a list of image paths and integer labels.
    Returns (tensor, label) pairs.
    """
    def __init__(self, paths: List[str], labels: List[int], transform):
        self.paths = paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        x = self.transform(img)
        y = self.labels[idx]
        return x, y


def extract_embeddings(
    dirs: List[str],
    ckpt: str,
    batch_size: int,
    out_path: str
):
    # collect all image paths and labels
    paths_all, labels_all = [], []
    for d in dirs:
        p, l = load_dataset(d)
        paths_all.extend(p)
        labels_all.extend(l)

    print(f"[i] Total images: {len(paths_all):,}")

    # map class names to indices
    classes = sorted(set(labels_all))
    label2idx = {c: i for i, c in enumerate(classes)}
    idx_labels = [label2idx[l] for l in labels_all]

    # prepare DataLoader
    transform = get_transform(224)
    ds = EmbeddingDataset(paths_all, labels_all, transform)
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True
    )

    # load fine-tuned model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FineTuneModel(num_classes=len(classes)).to(device)
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state, strict=False)
    model.eval()

    # extract and normalize embeddings
    feats_list, ys = [], []
    with torch.no_grad():
        for xb, yb in tqdm(dl, desc="Extracting embeddings"):
            xb = xb.to(device, non_blocking=True)
            # get raw features from backbone
            feat = model.backbone(xb)
            # L2-normalize
            feat = feat / feat.norm(dim=1, keepdim=True)
            feats_list.append(feat.cpu())
            ys.extend(yb)

    feats = torch.cat(feats_list, dim=0).numpy().astype("float32")

    # save to pickle
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump({"feats": feats, "labels": labels_all}, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[i] Saved {feats.shape[0]} embeddings to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dirs",
        nargs="+",
        default=["old_dataset/train", "old_dataset/val"],
        help="List of image-root directories"
    )
    parser.add_argument(
        "--ckpt",
        default="embed_checkpoints/dino_vits16_light_ep3_acc1.000.pth",
        help="Fine-tuned checkpoint path (.pth)"
    )
    parser.add_argument(
        "--bs",
        type=int,
        default=256,
        help="Batch size for embedding extraction"
    )
    parser.add_argument(
        "--out",
        default="embeddings2/embeddings_fine_tuned.pkl",
        help="Output pickle path"
    )
    args = parser.parse_args()

    extract_embeddings(
        dirs=args.dirs,
        ckpt=args.ckpt,
        batch_size=args.bs,
        out_path=args.out
    )
