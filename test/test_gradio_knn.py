# === monkey-patch starts ===
# debug error: APIInfoParseError: Cannot parse schema True
import gradio_client.utils as _gc

# save the orginal
_orig_get_type = _gc.get_type
_orig_json2py = _gc._json_schema_to_python_type

# ignore bool schema
def _safe_get_type(schema):
    if isinstance(schema, bool):
        return "Any"
    return _orig_get_type(schema)

def _safe_json_schema_to_python_type(schema, defs=None):
    if isinstance(schema, bool):
        return "Any"
    return _orig_json2py(schema, defs)

# apply patch
_gc.get_type = _safe_get_type
_gc._json_schema_to_python_type = _safe_json_schema_to_python_type
# === monkey-patch ends ===

# debug
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import gradio as gr
import json
import numpy as np
from PIL import Image
from infer import predict

def gradio_predict(img: Image.Image):

    try:
        res = predict(img, k=300, ood_threshold=0.6)
        rows = [
            [c["class"], f"{c['similarity']:.3f}"]
            for c in res["top5"]
        ]
        return [np.array(img)], rows
    except Exception as e:
        return [], [[f"Error: {e}"]]


iface = gr.Interface(
    fn=gradio_predict,
    inputs=gr.Image(type="pil", label="Upload a pill image"),
    outputs=[
        gr.Gallery(label="Input Image (No Crop)", columns=1),
        gr.Dataframe(headers=["Class", "Similarity"], label="Top-5 Distinct Classes")
    ],
    title="Pill Classification without YOLO",
    description="Classifiy the uploaded image through DINO + KNN"
)

if __name__ == "__main__":
    iface.launch(inbrowser=True)