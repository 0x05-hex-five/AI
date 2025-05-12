import argparse, os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast
from tqdm.auto import tqdm

from src.data import make_dataloaders, make_cv_dataloaders
from src.model import build_model

# A training function for one epoch
def train_one_epoch(model, dataloader, criterion, optimizer, device, scaler, epoch, max_epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    # Use tqdm to show progress bar
    progress_bar = tqdm(
                        dataloader, 
                        desc=f"Epoch {epoch}/{max_epoch}",
                        leave=True
                        )

    # Iterate over batches in the dataloader
    for images, labels in progress_bar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        with autocast(enabled=scaler.is_enabled()):
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update() # reduce the scale for the next iteration

        preds = outputs.argmax(1)
        running_loss += loss.item() * images.size(0)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        progress_bar.set_postfix(
            loss=f"{running_loss / total:.4f}",
            acc=f"{correct / total:.4f}" 
        )

    return running_loss / total, correct / total

@torch.no_grad()
def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds, all_labels = [], []

    progress_bar = tqdm(
                    dataloader, 
                    desc="Validation",
                    leave=True
                    )

    for images, labels in progress_bar:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)
        preds = outputs.argmax(1)

        running_loss += loss.item() * images.size(0)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        # Update predictions and labels
        # Convert to numpy arrays and extend the lists
        all_preds.extend(preds.cpu().numpy().tolist())
        all_labels.extend(labels.cpu().numpy().tolist())

    avg_loss = running_loss / total
    acc = correct / total

    from sklearn.metrics import precision_score, recall_score, f1_score
    precision = precision_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')
    f1 = f1_score(all_labels, all_preds, average='macro')

    return avg_loss, acc, precision, recall, f1

@torch.no_grad()
def testing(model, dataloader, criterion, device, class_names=None):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds, all_labels = [], []

    progress_bar = tqdm(
                    dataloader, 
                    desc="Testing",
                    leave=True
                    )

    for images, labels in progress_bar:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)
        preds = outputs.argmax(1)

        running_loss += loss.item() * images.size(0)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        # Update predictions and labels
        # Convert to numpy arrays and extend the lists
        all_preds.extend(preds.cpu().numpy().tolist())
        all_labels.extend(labels.cpu().numpy().tolist())

    avg_loss = running_loss / total
    acc = correct / total

    from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
    precision = precision_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')
    f1 = f1_score(all_labels, all_preds, average='macro')

    report_df = None
    if class_names:
        report_dict = classification_report(
            all_labels,
            all_preds,
            target_names=class_names,
            output_dict=True,   # to get a dict
            digits=4,
            zero_division=0
        )
        # Convert the report_dict to a DataFrame
        report_df = pd.DataFrame(report_dict).T
        out_dir = Path("reports")
        out_dir.mkdir(parents=True, exist_ok=True)
        report_df.to_csv(out_dir / "test_classification_report.csv")

    return avg_loss, acc, precision, recall, f1

def log_gpu_usage(device):
    allocated = torch.cuda.memory_allocated(device)   # currently allocated memory
    reserved  = torch.cuda.memory_reserved(device)    # reserved memory
    total     = torch.cuda.get_device_properties(device).total_memory
    used_pct  = allocated / total * 100 # percentage of used memory
    print(f"[GPU] Allocated: {allocated/1024**3:.2f} GB  "
          f"Reserved: {reserved/1024**3:.2f} GB  "
          f"Utilization: {used_pct:.1f}%")

def main(args):
    cfg = yaml.safe_load(open(args.config, "r"))

    # bring in the config file
    ckpt_path = cfg["train"]["checkpoint_path"]
    ckpt_dir  = os.path.dirname(ckpt_path)
    os.makedirs(ckpt_dir, exist_ok=True)

    train_loader, val_loader, test_loader = make_dataloaders(cfg)

    model, device = build_model(
        model_name=cfg["model"]["name"],
        num_classes=cfg["model"]["num_classes"],
        pretrained=cfg["model"]['pretrained'],
        freeze_backbone=cfg["model"]['freeze_backbone'],
        multi_gpu=cfg["model"]['multi_gpu']
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=cfg["train"]["lr"])
    scaler = GradScaler(enabled=cfg["train"]["amp"]) # Automatic Mixed Precision

    patience = cfg["train"]["patience"]  # Early stopping patience
    wait = 0
    best_val_acc = 0.0
    for epoch in range(1, cfg['train']['epochs'] + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler, epoch, cfg['train']['epochs'])
        val_loss, val_acc, val_prec, val_rec, val_f1 = validate(model, val_loader, criterion, device)
        log_gpu_usage(device)
        tqdm.write(f"Epoch {epoch}/{cfg['train']['epochs']}  "
                   f"Train Loss: {train_loss:.4f}  Train Acc: {train_acc:.4f}  "
                   f"Val Loss: {val_loss:.4f}  Val Acc: {val_acc:.4f}")

        # Save the model if validation accuracy improves
        if val_acc > best_val_acc + 1e-4:
            best_val_acc = val_acc
            wait = 0
            torch.save(model.state_dict(), cfg["train"]["checkpoint_path"])
            print(f"Model saved at epoch {epoch} with accuracy {best_val_acc:.4f}")
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stopping at epoch {epoch} with patience {patience}")
                break

    # Load the best model for testing
    test_loss, test_acc, test_prec, test_rec, test_f1 = testing(model, test_loader, criterion, device, class_names=test_loader.dataset.classes)
    print(f"Test Loss: {test_loss:.4f}, Acc: {test_acc:.4f}, Prec: {test_prec:.4f}, \n"
          f"Rec: {test_rec:.4f}, F1: {test_f1:.4f}")

# cross-validation main function 
def cv_main(args):
    cfg = yaml.safe_load(open(args.config))

    ckpt_path = cfg["train"]["cv_checkpoint_path"]
    ckpt_dir  = os.path.dirname(ckpt_path)
    os.makedirs(ckpt_dir, exist_ok=True)
    
    # iterate over 5 folds
    for fold in range(5):
        print(f"Fold {fold + 1}/5")
        train_loader, val_loader = make_cv_dataloaders(cfg, fold=fold, n_splits=5)

        model, device = build_model(
            model_name=cfg["model"]["name"],
            num_classes=cfg["model"]["num_classes"],
            pretrained=cfg["model"]['pretrained'],
            freeze_backbone=cfg["model"]['freeze_backbone'],
            multi_gpu=cfg["model"]['multi_gpu']
        )

        criterion = nn.CrossEntropyLoss()
        optimizer = AdamW(model.parameters(), lr=cfg["train"]["lr"])
        scaler = GradScaler(enabled=cfg["train"]["amp"])

        patience = cfg["train"]["patience"]  # Early stopping patience
        wait = 0
        best_scores = []
        best_val_acc = 0.0

        # Training loop for each fold
        for epoch in range(1, cfg['train']['epochs'] + 1):

            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler, epoch, cfg['train']['epochs'])
            val_loss, val_acc, val_prec, val_rec, val_f1 = validate(model, val_loader, criterion, device)
            log_gpu_usage(device)
            tqdm.write(f"Epoch {epoch}/{cfg['train']['epochs']}  "
                    f"Train Loss: {train_loss:.4f}  Train Acc: {train_acc:.4f}  "
                    f"Val Loss: {val_loss:.4f}  Val Acc: {val_acc:.4f}   "
                    f"Val Prec: {val_prec:.4f}  Val Rec: {val_rec:.4f}  Val F1: {val_f1:.4f}")

            if val_acc > best_val_acc + 1e-4:
                best_val_acc = val_acc
                wait = 0
                torch.save(model.state_dict(), cfg["train"]["cv_checkpoint_path"])
                print(f"Model saved at fold {fold} with accuracy {best_val_acc:.4f}")
            else:
                wait += 1
                if wait >= patience:
                    print(f"Early stopping at fold {fold} with patience {patience}")
                    break
        best_scores.append(best_val_acc)

    print(f"\n===== Cross-Val Results =====")
    print(f"Mean val_acc: {np.mean(best_scores):.4f} ± {np.std(best_scores):.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="default.yaml")

    args = parser.parse_args()
    main(args) # main function for training
    # cv_main(args) # cross-validation function for training