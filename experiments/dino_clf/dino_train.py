import random, time, math, os, json, numpy as np
from pathlib import Path

import torch, torch.nn as nn
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm

from dino_utils import IMAGE_SIZE, BATCH_SIZE, get_dataloaders
from dino_model     import DinoLinear, unfreeze_last_blocks

NUM_CLASSES   = 15
RUN_DIR       = Path("runs/foundation")
RUN_DIR.mkdir(parents=True, exist_ok=True)
EPOCHS_STAGE1 = 6
EPOCHS_STAGE2 = 4
LR_HEAD       = 3e-3
LR_BACKBONE   = 3e-4
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"

def epoch_loop(model, loader, criterion, optimizer=None, desc="train"):
    train = optimizer is not None
    if train:
        model.train()
    else:
        model.eval()
    preds, targets, losses = [], [], []
    bar = tqdm(loader, desc=desc, leave=False)
    for imgs, labels in bar:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        logits = model(imgs)
        loss   = criterion(logits, labels)
        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        losses.append(loss.item())
        preds.extend(logits.argmax(1).cpu().tolist())
        targets.extend(labels.cpu().tolist())
    acc  = accuracy_score(targets, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(targets, preds, average="macro", zero_division=0)
    return float(np.mean(losses)), acc, prec, rec, f1

if __name__ == "__main__":
    torch.manual_seed(42) # initialize random seed for weights
    random.seed(42) # initialize random seed for sampler
    np.random.seed(42) # initialize random seed for data augmentation

    data_root = Path("dataset/split_15")
    tl, vl, test_loader = get_dataloaders(data_root, BATCH_SIZE)

    model = DinoLinear(NUM_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05) # regularization

    # ---------------------------- STAGE 1 : linear head ------------------------
    for p in model.backbone.parameters():
        p.requires_grad = False
    optimizer = AdamW(model.head.parameters(), lr=LR_HEAD)
    best_val_acc = 0.0

    for ep in range(1, EPOCHS_STAGE1+1):
        tr_loss, tr_acc, *_ = epoch_loop(model, tl, criterion, optimizer, f"train‑h {ep}/{EPOCHS_STAGE1}")
        val_loss, val_acc, v_prec, v_rec, v_f1 = epoch_loop(model, vl, criterion, None, f"val‑h   {ep}")
        print(f"[Stage1] ep{ep}  train_acc={tr_acc:.3f}  val_acc={val_acc:.3f}")
        if val_acc > best_val_acc + 1e-3:
            best_val_acc = val_acc
            torch.save(model.state_dict(), RUN_DIR/"best_stage1.pth")

    # ---------------------------- STAGE 2 : unfreeze last blocks --------------
    unfreeze_last_blocks(model, n_blocks=2)
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR_BACKBONE)

    for ep in range(1, EPOCHS_STAGE2+1):
        tr_loss, tr_acc, *_ = epoch_loop(model, tl, criterion, optimizer, f"train‑ft {ep}/{EPOCHS_STAGE2}")
        val_loss, val_acc, v_prec, v_rec, v_f1 = epoch_loop(model, vl, criterion, None, f"val‑ft   {ep}")
        print(f"[Stage2] ep{ep}  train_acc={tr_acc:.3f}  val_acc={val_acc:.3f}")
        torch.save(model.state_dict(), RUN_DIR/"best_stage2.pth")
        if val_acc > best_val_acc + 1e-3:
            best_val_acc = val_acc
            torch.save(model.state_dict(), RUN_DIR/"best_stage2.pth")

    # ---------------------------- TEST ---------------------------------------
    model.load_state_dict(torch.load(RUN_DIR/"best_stage2.pth"))
    test_loss, test_acc, t_prec, t_rec, t_f1 = epoch_loop(model, test_loader, criterion, None, "test   ")
    print(f"TEST   acc={test_acc:.3f}  prec={t_prec:.3f}  rec={t_rec:.3f}  f1={t_f1:.3f}")

    # ---------------------------- EXPORT TS ----------------------------------
    example = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE).to(DEVICE)
    ts = torch.jit.trace(model, example)
    ts.save(RUN_DIR/"model_ts.pt")
    print("TorchScript export done →", RUN_DIR/"model_ts.pt")