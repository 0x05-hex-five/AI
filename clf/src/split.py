from pathlib import Path
import random, shutil, os
import pandas as pd

"""
This script splits only the specific classes in excel file of the dataset into train, validation, and test sets.
"""

random.seed(42)  # For reproducibility

root = Path("./dataset/crop")
out = Path("./dataset/split_15")
out.mkdir(parents=True, exist_ok=True)

# Read the CSV file
mapping_df = pd.read_excel("new class_15.xlsx", dtype=str)
code2group = dict(zip(mapping_df["C-Code"], mapping_df["통합클래스명"]))

# iterate through each class directory
for cls_dir in sorted(root.iterdir()):
    if not cls_dir.is_dir():
        continue
    code = cls_dir.name
    group = code2group.get(code)
    if group is None:
        # skip if group is not found
        continue

    # get all images in the class directory
    imgs = list(cls_dir.glob("*.jpg"))
    random.shuffle(imgs)

    n = len(imgs)
    n_train = int(n * 0.7)
    n_val   = int(n * 0.85)

    train_imgs = imgs[:n_train]
    val_imgs   = imgs[n_train:n_val]
    test_imgs  = imgs[n_val:]

    # create target directories
    for split, subset in (("train", train_imgs),
                          ("val",   val_imgs),
                          ("test",  test_imgs)):
        target_dir = out/split/group
        target_dir.mkdir(parents=True, exist_ok=True)

        # copy images to target directory
        for img in subset:
            shutil.copy(img, target_dir/img.name)

    print(f"[{code}] → {group}:  train={len(train_imgs)}, val={len(val_imgs)}, test={len(test_imgs)}")