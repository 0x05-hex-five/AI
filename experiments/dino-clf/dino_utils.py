from pathlib import Path
from typing import Tuple
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision.datasets import ImageFolder
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Global constants
IMAGE_SIZE = 392
BATCH_SIZE = 64

# Transforms
def make_train_transform() -> A.Compose:
    """Strong augmentation tailored to pill photos"""
    return A.Compose([
        A.LongestMaxSize(max_size=IMAGE_SIZE), # resize to max size while keeping aspect ratio
        A.PadIfNeeded(IMAGE_SIZE, IMAGE_SIZE, border_mode=0, value=(255, 255, 255)), # pad to square
        A.RandomBrightnessContrast(0.4, 0.4, p=.6), # random brightness and contrast
        A.HueSaturationValue(20, 15, 15, p=.4), # random hue, saturation and value
        A.ImageCompression(quality_lower=30, quality_upper=100, p=.4), # random image compression
        A.GaussianBlur(p=.2), # random gaussian blur
        A.Rotate(limit=180, border_mode=0, value=(255, 255, 255), p=.5), # random rotation
        A.CoarseDropout(max_holes=4, max_height=32, max_width=32, fill_value=255, p=.3), # randomly change pixels into white
        ToTensorV2(), # convert to tensor
    ])


def make_val_transform() -> A.Compose:
    return A.Compose([
        A.LongestMaxSize(max_size=IMAGE_SIZE),
        A.PadIfNeeded(IMAGE_SIZE, IMAGE_SIZE, border_mode=0, value=(255, 255, 255)),
        ToTensorV2(),
    ])

# Dataset class
class AlbumentationsImageFolder(Dataset):
    """torchvision.datasets.ImageFolder + Albumentations transform"""

    def __init__(self, root: Path, transform: A.Compose):
        self.ds = ImageFolder(root)
        self.transform = transform

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index: int):
        path, label = self.ds.samples[index]
        image = self.ds.loader(path)
        image = np.array(image)
        image = self.transform(image=image)["image"]
        if image.dtype == torch.uint8:
            image = image.float() / 255.0
        return image, label

    @property
    def targets(self):
        """Expose class labels to the sampler"""
        return self.ds.targets
    
# Sampler
def get_balanced_sampler(labels) -> WeightedRandomSampler:
    """Inverse‑frequency weighted random sampler"""
    counts = torch.bincount(torch.tensor(labels))
    class_weights = 1.0 / counts.float()
    sample_weights = class_weights[labels]
    return WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    
# Loader factory
def get_dataloaders(data_root: Path, batch_size: int = BATCH_SIZE, num_workers: int = 8):
    train_tf = make_train_transform()
    val_tf   = make_val_transform()

    train_ds = AlbumentationsImageFolder(data_root / "train", train_tf)
    val_ds   = AlbumentationsImageFolder(data_root / "val",   val_tf)
    test_ds  = AlbumentationsImageFolder(data_root / "test",  val_tf)

    train_sampler = get_balanced_sampler(train_ds.targets)

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=train_sampler,
                              num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers//2, pin_memory=True)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers//2, pin_memory=True)

    return train_loader, val_loader, test_loader