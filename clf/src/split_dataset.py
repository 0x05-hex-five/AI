from pathlib import Path
import random, shutil, os

random.seed(42)  # For reproducibility

root = Path("./dataset/crop")
out = Path("./dataset/split")
out.mkdir(parents=True, exist_ok=True)

for cls_dir in root.iterdir():
    print(cls_dir.name)
    imgs = list(cls_dir.glob("*.jpg")) # All images in the class directory
    random.shuffle(imgs)
    
    n = len(imgs)
    train = imgs[:int(n * 0.8)]
    val = imgs[int(n * 0.8):int(n * 0.9)]
    test = imgs[int(n * 0.9):]

    # Create directories for train, val, test
    (out / "train" / cls_dir.name).mkdir(parents=True, exist_ok=True)
    (out / "val" / cls_dir.name).mkdir(parents=True, exist_ok=True)
    (out / "test" / cls_dir.name).mkdir(parents=True, exist_ok=True)

    for img in imgs:
        if img in train:
            shutil.copy(img, out / "train" / cls_dir.name)
        elif img in val:
            shutil.copy(img, out / "val" / cls_dir.name)
        elif img in test:
            shutil.copy(img, out / "test" / cls_dir.name)