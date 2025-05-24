import os, cv2, numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, Subset
from torchvision.datasets import ImageFolder
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import StratifiedKFold
from albumentations import Normalize

mean = (0.485, 0.456, 0.406)
std  = (0.229, 0.224, 0.225)

def find_under_represented_classes(root, threshold=None):
    ds = ImageFolder(root)
    counts = np.bincount(ds.targets)
    if threshold is None:
        threshold = counts.mean()
    under = {i for i, c in enumerate(counts) if c < threshold}
    return under
    
def get_transforms(img_size=300):
    base_tf = A.Compose([
        A.Resize(img_size, img_size),
        # A.Normalize(),
        Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])
    heavy_tf = A.Compose([
        A.Resize(img_size, img_size),
        A.ShiftScaleRotate(
            shift_limit=0.0625, scale_limit=0.1, rotate_limit=15,
            border_mode=cv2.BORDER_CONSTANT, p=0.5
        ),
        A.RandomBrightnessContrast(brightness_limit=0.1,
                                   contrast_limit=0.1, p=1.0),
        # A.Normalize(),
        Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])
    return base_tf, heavy_tf

class CustomAugDataset(Dataset):
    def __init__(self, root, base_tf, heavy_tf, under_cls):
        self.dataset = ImageFolder(root)
        self.base_tf = base_tf
        self.heavy_tf = heavy_tf
        self.under_cls = under_cls

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, target = self.dataset[idx]
        imp_np = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        tf = self.heavy_tf if target in self.under_cls else self.base_tf
        out = tf(image=imp_np)["image"]
        return out, target

# Weights for the WeightedRandomSampler
def make_sampler(ds):
    if isinstance(ds, Subset): # If using a subset, get the base dataset and indices
        base_ds = ds.dataset
        idxs = ds.indices
    else:
        base_ds = ds # CustomAugDataset
        idxs = list(np.arange(len(ds)))

    labels = base_ds.dataset.targets # ImageFolder labels
    counts = np.bincount(labels) # Count the number of images in each class
    class_weights = 1.0 / counts.astype(float)
    sample_weights = [class_weights[labels[i]] for i in idxs] # Weights for each sample
    return WeightedRandomSampler(sample_weights,
                                 num_samples=len(sample_weights),
                                 replacement=True ) # Allow duplicate samples

def alb_to_tensor(img, tf):
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) # Convert to BGR for albumentations
    img = tf(image=img)["image"] # Apply the transformation
    return img

def make_dataloaders(cfg):
    root = Path(cfg["data"]["root"])
    img_size = cfg["data"].get("img_size", 300)
    base_tf, heavy_tf = get_transforms(img_size)

    under = find_under_represented_classes(root/"train",
                                           threshold=cfg["data"].get("threshold"))

    train_ds = CustomAugDataset(root/"train", base_tf, heavy_tf, under)
    val_ds   = ImageFolder(root/"val", transform=lambda x: alb_to_tensor(x, base_tf))
    test_ds  = ImageFolder(root/"test",  transform=lambda x: alb_to_tensor(x, base_tf))

    sampler = make_sampler(train_ds)

    train_loader = DataLoader(train_ds,
                              batch_size=cfg["train"]["batch_size"],
                              sampler=sampler, # Use the sampler for class imbalance
                              num_workers=cfg["train"]["num_workers"],
                              pin_memory=True, # Pin memory for faster data transfer to GPU
                              drop_last=True
                              )
    val_loader = DataLoader(val_ds,
                            batch_size=cfg["train"]["batch_size"],
                            shuffle=False,
                            num_workers=cfg["train"]["num_workers"]//4,
                            pin_memory=True
                            )
    test_loader = DataLoader(test_ds,
                             batch_size=cfg["train"]["batch_size"],
                             shuffle=False,
                             num_workers=cfg["train"]["num_workers"]//4,
                             pin_memory=True
                             )
    return train_loader, val_loader, test_loader

# Function for cross-validation
def make_cv_dataloaders(cfg, fold: int, n_splits: int = 5):
    root = Path(cfg["data"]["root"])
    img_size = cfg["data"].get("img_size", 300)
    base_tf, heavy_tf = get_transforms(img_size)

    # The whole train ImageFolder
    full = CustomAugDataset(root/"train", base_tf, heavy_tf, find_under_represented_classes(root/"train"))

    # The base ImageFolder for validation
    val_base = ImageFolder(root/"train", transform=lambda img: alb_to_tensor(img, base_tf))

    # Stratified K-Folds cross-validation
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    splits = list(skf.split(np.zeros(len(full)), full.dataset.targets))
    train_idx, val_idx = splits[fold]

    train_ds = Subset(full, train_idx)
    val_ds   = Subset(val_base, val_idx) # Use the original dataset for validation

    sampler = make_sampler(train_ds)

    train_loader = DataLoader(train_ds,
                              batch_size=cfg["train"]["batch_size"],
                              sampler=sampler, # Use the sampler for class imbalance
                              num_workers=cfg["train"]["num_workers"],
                              pin_memory=True, # Pin memory for faster data transfer to GPU
                              drop_last=True
                              )
    val_loader = DataLoader(val_ds,
                            batch_size=cfg["train"]["batch_size"],
                            shuffle=False,
                            num_workers=cfg["train"]["num_workers"]//4,
                            pin_memory=True
                            )
    
    return train_loader, val_loader

if __name__ == "__main__":
    from argparse import ArgumentParser
    import yaml

    p = ArgumentParser()
    p.add_argument("--config", default="default.yaml")
    args = p.parse_args()

    cfg = yaml.safe_load(open(args.config, "r"))
    # tl, vl, tsl = make_dataloaders(cfg)
    # print("datasets:", len(tl.dataset), len(vl.dataset), len(tsl.dataset))
    # print("batches :", len(tl), len(vl), len(tsl))

    tl, vl = make_cv_dataloaders(cfg, fold=0, n_splits=5)
    print("datasets:", len(tl.dataset), len(vl.dataset))