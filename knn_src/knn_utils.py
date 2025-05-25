import os
from PIL import Image
from torchvision import transforms

def load_dataset(root_dir: str):
    """
    root_dir/
      └── C-000001/
            ├── img1.jpg
            └── img2.jpg
    Returns:
      - image_paths: ["/.../C-000001/img1.jpg", ...]
      - labels:      ["C-000001", ...]
    """
    images, labels = [], []
    for cls in sorted(os.listdir(root_dir)):
        cls_dir = os.path.join(root_dir, cls)
        if not os.path.isdir(cls_dir):
            continue
        for fname in sorted(os.listdir(cls_dir)):
            if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                images.append(os.path.join(cls_dir, fname))
                labels.append(cls)
    return images, labels

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]

def resize_and_pad(img: Image.Image, target_size: int = 224) -> Image.Image:
    """
    Resize an image to fit within target_size while maintaing aspect ratio,
    then pad the remaining area with black pixels to create a square image.
    """
    w, h = img.size
    # compute scale factor
    scale = target_size / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    # resize using Lanczos filter for high-quality sampling
    img_resized = img.resize((new_w, new_h), Image.LANCZOS)
    # create a black background and paste resized image centered
    background = Image.new("RGB", (target_size, target_size), (0, 0, 0))
    x_offset = (target_size - new_w) // 2
    y_offset = (target_size - new_h) // 2
    background.paste(img_resized, (x_offset, y_offset))
    return background

def get_transform(image_size: int = 224):
    """
    Return a torchvision transform pipeline that:
        1. Resizes and pads the image to (image_size, image_size)
        2. Converts to a tensor
        3. Normalizes using ImageNet mean and std
    """
    return transforms.Compose([
        transforms.Lambda(lambda img: resize_and_pad(img, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ])