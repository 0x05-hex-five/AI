import argparse
import pickle
from pathlib import Path
from typing import List

import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .knn_utils import get_transform, load_dataset

"""
# This script extracts raw DINO embeddings for images in specified directories.
# It loads a pretrained DINO model, processes images, and saves their embeddings and labels to a pickle file.
"""

class EmbeddingDataset(Dataset):
    def __init__(self, paths: List[str], labels: List[int], transform):
        self.paths = paths
        self.labels = labels
        self.transform = transform
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('RGB')
        return self.transform(img), self.labels[idx]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract DINO Embeddings')
    parser.add_argument('--dirs', nargs='+', default=["old_dataset/train", "old_dataset/val"], help='Dataset directories')
    parser.add_argument('--bs',   type=int, default=256, help='Batch size')
    parser.add_argument('--size', type=int, default=224, help='Image size')
    parser.add_argument('--out',  default="embeddings/embeddings_dino_raw.pkl", help='Output pickle path')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load DINO backbone from Torch Hub
    model = torch.hub.load('facebookresearch/dino:main', 'dino_vits16').to(device)
    model.eval()
    def embed(x):
        with torch.no_grad():
            out = model(x)  # get features
            feat = out if out.dim() == 2 else out[:,0]  # CLS if needed
            return feat / feat.norm(dim=1, keepdim=True)

    # Collect image paths and corresponding class names
    paths, labels = [], []
    for d in args.dirs:
        pths, lbs = load_dataset(d)
        paths += pths
        labels += lbs  # store actual class names

    transform = get_transform(args.size)

    # Extract embeddings one by one
    feats, names = [], []
    for img_path, label in tqdm(zip(paths, labels), total=len(paths), desc='Extract'):
        img = Image.open(img_path).convert('RGB')
        x = transform(img).unsqueeze(0).to(device)
        emb = embed(x).cpu().numpy()[0]
        feats.append(emb)
        names.append(label)

    feats = np.stack(feats).astype('float32')

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out,'wb') as f:
        pickle.dump({'feats':feats,'labels':names}, f)
    print(f'Saved embeddings {feats.shape} and labels to {args.out}')