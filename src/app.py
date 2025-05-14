import os, sys

from fastapi import FastAPI, UploadFile, File, HTTPException
from ultralytics import YOLO
import torch, cv2, numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms
from collections import OrderedDict

script_dir = os.path.dirname(__file__)  
lib_src = os.path.abspath(os.path.join(script_dir, "..", "clf", "src"))
if lib_src not in sys.path:
    sys.path.insert(0, lib_src)

from data import get_transforms, alb_to_tensor
from model import build_model

app = FastAPI(title="Pill Detection & Classification")

# Load models
yolo_model = YOLO("weights/best_yolo_TS_70.pt")
eff_model, _ = build_model(
    model_name="tf_efficientnetv2_s",
    num_classes=96,
    pretrained=False,
    freeze_backbone=False,
    multi_gpu=False
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load weights
orig_state = torch.load("weights/best_eff.pth", map_location=device)
# Remove "module." prefix from keys
new_state = OrderedDict({
    (k[7:] if k.startswith("module.") else k): v
    for k, v in orig_state.items()
})
# Load state dict into model
eff_model.load_state_dict(new_state)
eff_model.eval().to(device)

base_tf, _ = get_transforms(img_size=300)
mapping_df = pd.read_excel("class_mapping.xlsx", index_col=0)
mapping_df = mapping_df.sort_values("C-Code")
class_names = mapping_df["제품명"].tolist()

@app.post("/ai/predict")
async def predict(file: UploadFile = File(...)):
    data = await file.read()
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image data")
    
    # YOLOv8 Inference
    results = yolo_model(img, imgsz=640)[0]
    boxes = results.boxes.xyxy.cpu().numpy()
    if len(boxes) == 0:
        return {"error": "No pills detected"}
    
    # Get the largest box
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    x1, y1, x2, y2 = boxes[np.argmax(areas)].astype(int)

    # Crop the image
    cropped_img = img[y1:y2, x1:x2]

    # Convert to PIL Image
    pil_img = Image.fromarray(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB))

    # Transform the image
    img_tensor = alb_to_tensor(pil_img, base_tf)
    img_tensor = img_tensor.unsqueeze(0).to(device) # Add batch dimension with unsqueeze(0)

    # Inference
    with torch.no_grad():
        res = eff_model(img_tensor) # [1, 96]
        preds = torch.softmax(res, dim=1).cpu().numpy()[0] # [96,]

    # Get top 1 class
    top_idx = preds.argmax()

    # JSON response
    response = {
        "box": [int(x1), int(y1), int(x2), int(y2)],
        "top_class": class_names[top_idx],
        "top_prob": float(preds[top_idx]),
        "all_classes": [
            {"class": class_names[i], "prob": float(preds[i])}
            for i in range(len(class_names))
        ]
    }

    return response