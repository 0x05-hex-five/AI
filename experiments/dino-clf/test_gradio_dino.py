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

import cv2
import torch
import torch.nn.functional as F
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import gradio as gr
import numpy as np
from ultralytics import YOLO

IMAGE_SIZE = 392              # keep consistent with training
MODEL_PATH = "weights/model_ts_stage1.pt"
yolo_weights = "weights/best_yolo_TS_70.pt"
CLASSES = ["글리아타민", "글리아티린", "로수젯", "리바로", '리피토', '릭시아나', '비리어드', '아리셉트', '아리셉트에비스', '아모잘탄', '아모잘탄큐', '아모잘탄플러스', '아토젯', '트윈스타', '플라빅스']

# preprocess
val_tf = A.Compose([
    A.LongestMaxSize(max_size=IMAGE_SIZE),
    A.PadIfNeeded(IMAGE_SIZE, IMAGE_SIZE, border_mode=0, value=(255,255,255)),
    ToTensorV2(),
])

def preprocess(pil_img: Image.Image):
    arr = np.array(pil_img.convert("RGB"))
    tensor = val_tf(image=arr)["image"]  # uint8→float32 0‑1
    if tensor.dtype == torch.uint8:
        tensor = tensor.float() / 255.0 # convert to float32
    return tensor.unsqueeze(0)  # [1,3,H,W]

# load model
yolo_model = YOLO(yolo_weights)
print("Loading TorchScript model…")
model_ts = torch.jit.load(str(MODEL_PATH), map_location="cpu").eval()

# inference
def predict(image: Image.Image):
    # YOLO detect
    img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    results = yolo_model(img_bgr, imgsz=640, conf=0.08)[0]

    # get bounding boxes
    annotated = results.plot() # BGR image
    boxes = results.boxes.xyxy.cpu().numpy()

    if len(boxes) == 0:
        return annotated, "Failed to detect", None
    
    # get the largest area
    # areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) # (x2-x1)*(y2-y1)
    # x1, y1, x2, y2 = boxes[areas.argmax()].astype(int) # get the largest area

    # get highest confidence box
    confs = results.boxes.conf.cpu().numpy()
    x1, y1, x2, y2 = boxes[confs.argmax()].astype(int) # get the highest confidence box

    crop_bgr = img_bgr[y1:y2, x1:x2] # crop the original numpy image
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB) # convert to RGB
    crop = Image.fromarray(crop_rgb) # convert to PIL image

    x = preprocess(crop)
    with torch.no_grad():
        logits = model_ts(x)
        probs  = F.softmax(logits, dim=1)[0]
    # labels = [CLASSES[i] for i in probs.indices.tolist()]
    # scores = [float(p) for p in probs.values.tolist()]

    prob_dict = {c: float(p) for c, p in zip(CLASSES, probs.tolist())}

    return annotated[...,::-1], crop, prob_dict

# Gradio UI
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload pill image"),
    outputs=[
        gr.Image(label="YOLO Detection"),
        gr.Image(label="Crop"),
        gr.Label(num_top_classes=None, label="All Class Probabilities")
    ],
    title="Pill Classification (DINOv2 demo)",
    description="Upload a pill image to detect and classify it.",
    examples=[
        # optional local image paths for quick demo
    ]
)

if __name__ == "__main__":
    demo.launch()