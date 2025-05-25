import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import pickle
import faiss
import numpy as np
import torch, cv2
import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image
from collections import OrderedDict

from knn_src.knn_utils import get_transform

# configuration: use environment variables or defaults
INDEX_PATH  = os.getenv("INDEX_PATH",  "embeddings/knn_flatip.index") # path to FAISS index file
LABELS_PATH = os.getenv("LABELS_PATH", "embeddings/labels.pkl") # path to labels pickle
IMG_SIZE    = int(os.getenv("IMG_SIZE", 224)) # input image size

# device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load FAISS index and labels once in the beginning
index = faiss.read_index(INDEX_PATH)
with open(LABELS_PATH, "rb") as f:
    class_labels = pickle.load(f)

# load DINO ViT-S/16 backbone with pretrained weights
backbone = torch.hub.load(
    'facebookresearch/dino:main', 'dino_vits16', pretrained=True
).to(device)
backbone.eval()

# prepare image transform
transform = get_transform(IMG_SIZE)

# Initialize FastAPI application
app = FastAPI(title="Pill Embedding & Classification")

@app.post("/ai/predict")
async def predict(file: UploadFile = File(...), k: int = 100, ood_threshold: float = 0.6):
    # read uploaded file and decode to BGR image using OpenCV
    data = await file.read()
    bgr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise HTTPException(status_code=400, detail="Invalid image file")

    # BGR -> RGB -> PIL Image
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb)

    # preprocess and extract embedding
    x = transform(img).unsqueeze(0).to(device) # (1,3,224,224)
    with torch.no_grad():
        feat = backbone(x)                      # (1, 384)
        feat = feat / feat.norm(dim=1, keepdim=True)               # L2 normalization
        vector = feat.cpu().numpy().astype('float32') # (1, 384)

    # perform FAISS search (inner product on normalized vectors)
    D, I = index.search(vector, k) # D: similarity, I: indices
    sims = D[0] # top-k similarity
    idxs = I[0] # top-k index

    # collect the highest similarity per class
    unique = OrderedDict()
    top_list = []  # list of (idx, similarity)
    for sim, idx in sorted(zip(sims, idxs), key=lambda x: -x[0]):
        class_name = class_labels[int(idx)]

        # update if class not seen or found with higher similarity
        if class_name not in unique or sim > unique[class_name]:
            unique[class_name] = float(sim)

    # sort top-5        
    top_list = sorted(unique.items(), key=lambda x: x[1], reverse=True)[:5] # [(class_name, sim), ...]

    # out-of-distribution
    top_label, top_score = top_list[0]
    detected = (top_score >= ood_threshold) # if top-1 similarity is above threshold, returns true

    # primary prediction
    primary = {
        "class_id": top_label,
        "similarity": round(top_score, 3)
    }

    # top_predictions list
    top_predictions = []
    for class_name, score in top_list:
        top_predictions.append({
            "class_id": class_name,
            "similarity": round(score, 3)
        })

    response = {
        "success": True,
        "detected": detected,
        "data": {
            "primary_prediction": primary,
            "top_predictions": top_predictions
        }
    }
    print(top_predictions)
    return response

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)