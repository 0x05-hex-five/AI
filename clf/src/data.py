import os, cv2, numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision.datasets import ImageFolder
import albumentations as A
from albumentations.pytorch import ToTensorV2

root = Path("./dataset/split")

tmp_ds = ImageFolder(root / "train")
counts = np.bincount(tmp_ds.targets) # Count the number of images in each class

threshold = counts.mean() # Set threshold to the mean of images in each class
print("threshold: ", threshold)
under_cls = {i for i, c in enumerate(counts) if c < threshold} # Classes with less than threshold images
# print("under_cls", [tmp_ds.classes[i] for i in under_cls])
print("num under-classes:", len(under_cls))

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
    
base_tf = A.Compose([
    A.Resize(300, 300), # Resize to 300x300
    A.Normalize(), # Normalize the image
    ToTensorV2(), # Convert to tensor
])
heavy_tf = A.Compose([
    A.Resize(300, 300), # Resize to 300x300
    A.ShiftScaleRotate(shift_limit=0.0625,   # location
                       scale_limit=0.1,      # scale
                       rotate_limit=15,      # rotation
                       border_mode=cv2.BORDER_CONSTANT,
                       p=0.5), # Randomly shift, scale, and rotate the image
    A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=1.0), # Brightness and contrast
    A.Normalize(), # Normalize the image
    ToTensorV2(), # Convert to tensor
])

train_ds = CustomAugDataset(root / "train", base_tf, heavy_tf, under_cls) # solve class imbalance with augmentation
val_ds = ImageFolder(root / "val", transform=base_tf)
test_ds = ImageFolder(root / "test", transform=base_tf)

# Weights for the WeightedRandomSampler
counts = np.bincount(train_ds.dataset.targets) # Count the number of images in each class
class_weights = 1.0 / counts.astype(float)
sample_weights = class_weights[train_ds.dataset.targets] # Weights for each sample
sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(train_ds.dataset),
    replacement=True, # Allow duplicate samples
)

train_loader = DataLoader(
    train_ds,
    batch_size=64,
    sampler=sampler, # Use the sampler for class imbalance
    num_workers=0,
    pin_memory=True,
    shuffle=False,
    drop_last=True,
)

val_loader = DataLoader(
    val_ds,
    batch_size=64,
    num_workers=0,
    pin_memory=True,
    shuffle=False,
)

test_loader = DataLoader(
    test_ds,
    batch_size=64,
    num_workers=0,
    pin_memory=True,
    shuffle=False,
)

print(len(train_loader.dataset), len(val_loader.dataset), len(test_loader.dataset))
print(len(train_loader), len(val_loader), len(test_loader))