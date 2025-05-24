# import os
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import json
import pickle
import numpy as np
import faiss
from PIL import Image
from knn_utils import get_transform
from fine_tune import FineTuneModel

# define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load DINO ViT model without projection head
model = FineTuneModel(num_classes=96)           
state = torch.load('embed_checkpoints/dino_vits16_light.pth', map_location=device)
model.load_state_dict(state, strict=False)          
backbone = model.backbone.to(device)
backbone.eval()

# load k-NN index & labels
index = faiss.read_index('embed_checkpoints/knn_cosine.index')
with open('embed_checkpoints/labels.pkl', 'rb') as f:
    class_labels = pickle.load(f)

# define transform
transform = get_transform(224)

# predict function
def predict(img: Image.Image, k: int = 5, ood_threshold: float = 0.6):
    x   = transform(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        feat = backbone(x).squeeze()
        feat = feat / feat.norm()                # L2-normalize
        vector = feat.cpu().numpy().astype('float32').reshape(1,-1)
    D, I = index.search(vector, k)       # D: distance, I: index
    sims = D[0].tolist() # top N similarity
    idxs  = I[0].tolist()

    class_best = {}
    for idx, sim in zip(idxs, sims):
        cls = class_labels[idx]
        # 아직 없거나 더 높은 sim이면 갱신
        if cls not in class_best or sim > class_best[cls]:
            class_best[cls] = sim

    # sort by sim and reverse it
    sorted_classes = sorted(
        class_best.items(),        # [(class, sim), ...]
        key=lambda x: x[1],        # sort by sim
        reverse=True
    )
    top5_distinct = sorted_classes[:5]  # [(class1,sim1),...]

    # OOD
    top1_cls, top1_sim = top5_distinct[0]
    label = top1_cls if top1_sim >= ood_threshold else "알 수 없는 알약"

    top5 = [
        {"class": cls_name, "similarity": float(sim)}
        for cls_name, sim in top5_distinct
    ]
    sim_mean = np.mean(sims)
    print(f"[DEBUG] sim_mean={sim_mean:.4f}, top5={top5}, threshold={ood_threshold:.4f}")
    print("index type:", type(index), "  metric:", index.metric_type)
    print("index.ntotal:", index.ntotal, "  len(labels):", len(class_labels))


    print("approx sims[:5]:", sims[:5])

    with open("embed_checkpoints/embeddings_fine_tuned.pkl","rb") as f:
        data = pickle.load(f)
    feats = data["feats"]   # (N, D)
    true_sims = (feats[idxs[:5]] @ vector.T).flatten()
    print("true sims[:5]:  ", true_sims.tolist())

    return {"label": label, "top5": top5, "top1_sim": top1_sim}