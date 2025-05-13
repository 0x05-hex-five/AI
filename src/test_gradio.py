# === monkey-patch starts ===
# debug error: APIInfoParseError: Cannot parse schema True
import gradio_client.utils as _gc

# 원본 저장
_orig_get_type = _gc.get_type
_orig_json2py = _gc._json_schema_to_python_type

# bool 스키마 무시
def _safe_get_type(schema):
    if isinstance(schema, bool):
        return "Any"
    return _orig_get_type(schema)

def _safe_json_schema_to_python_type(schema, defs=None):
    if isinstance(schema, bool):
        return "Any"
    return _orig_json2py(schema, defs)

# 패치 적용
_gc.get_type = _safe_get_type
_gc._json_schema_to_python_type = _safe_json_schema_to_python_type
# === monkey-patch ends ===

import os, sys
import gradio as gr
from pathlib import Path
from collections import OrderedDict

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image

from ultralytics import YOLO

script_dir = os.path.dirname(__file__)  
lib_src = os.path.abspath(os.path.join(script_dir, "..", "clf", "src"))
if lib_src not in sys.path:
    sys.path.insert(0, lib_src)

from data import get_transforms, alb_to_tensor
from model import build_model

# settings
yolo_weights = "weights/best_yolo_TS_70.pt"
eff_weights = "weights/best_eff.pth"
img_size = 300
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load models
yolo_model = YOLO(yolo_weights)
eff_model, _ = build_model(
    model_name="tf_efficientnetv2_s",
    num_classes=96,
    pretrained=False,
    freeze_backbone=False,
    multi_gpu=False
)

orig_state = torch.load(eff_weights, map_location=device)
# remove prefix from state dict keys
# to avoid issues with DataParallel
new_state = OrderedDict()
for k, v in orig_state.items():
    if k.startswith("module."):
        new_state[k[7:]] = v
    else:
        new_state[k] = v

eff_model.load_state_dict(new_state)
eff_model.eval().to(device)

# load transforms
base_tf, _ = get_transforms(img_size=img_size)

# load class names
mapping_df = pd.read_excel("class_mapping.xlsx", index_col=0)
mapping_df = mapping_df.sort_values("C-Code")
class_names = mapping_df["제품명"].tolist()

def inference(img_pill: Image.Image):
    # YOLO detect
    img = np.array(img_pill)
    results = yolo_model(img, imgsz=640)[0]

    # get bounding boxes
    annotated = results.plot() # BGR image

    boxes = results.boxes.xyxy.cpu().numpy()
    if len(boxes) == 0:
        return annotated, "Failed to detect", None

    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) # (x2-x1)*(y2-y1)
    x1, y1, x2, y2 = boxes[areas.argmax()].astype(int) # get the largest area
    crop = img[y1:y2, x1:x2] # crop the original numpy image

    # Classify
    # restore crop into PIL Image
    tensor = alb_to_tensor(Image.fromarray(crop), base_tf).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = eff_model(tensor)                        # (1, num_classes)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]  # shape: (num_classes,)

    # top-1 class
    top_idx = int(probs.argmax())
    top_class = class_names[top_idx]
    top_prob  = float(probs[top_idx])

    # 모든 클래스 확률을 name→prob dict 로
    prob_dict = { name: float(probs[i]) 
                  for i, name in enumerate(class_names) }

    return annotated, crop, prob_dict

demo = gr.Interface(
    fn=inference,
    inputs=gr.Image(type="pil", label="Pill Image"),
    outputs=[
        gr.Image(label="YOLO Detection"),
        gr.Image(label="Crop"),
        gr.Label(num_top_classes=None, label="All Class Probabilities")
    ],
    title="Pill Detection + Classification",
    description="Upload a pill image to detect and classify it."
).launch(inbrowser=True, share=True, show_api=False)