import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from src.model import build_model
from src.data import make_dataloaders
import yaml

def main():
    # load config
    cfg = yaml.safe_load(open("default.yaml"))

    # define model, device
    model, device = build_model(
        model_name=cfg["model"]["name"],
        num_classes=cfg["model"]["num_classes"],
        pretrained=False,
        freeze_backbone=False,
        multi_gpu=cfg["model"]['multi_gpu'],
    )
    # directory for saving the model
    ckpt = cfg["train"]["checkpoint_path"]
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    # prepare dataloaders for test set
    _, _, test_loader = make_dataloaders(cfg)

    # collect all predictions and labels
    all_labels, all_preds = [], []
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            logits = model(imgs)
            preds = logits.argmax(1).cpu().numpy()
            all_preds .extend(preds.tolist())
            all_labels.extend(labels.numpy().tolist())

    # calculate confusion matrix
    cm = confusion_matrix(all_labels, all_preds, labels=range(cfg["model"]["num_classes"]))
    disp = ConfusionMatrixDisplay(cm, display_labels=test_loader.dataset.classes)

    fig, ax = plt.subplots(figsize=(12,12))
    disp.plot(ax=ax, cmap="Blues", xticks_rotation="vertical")
    plt.title("Confusion Matrix on Test Set")
    plt.tight_layout()
    plt.show()
    fig.savefig("cm.png")

if __name__=="__main__":
    main()
