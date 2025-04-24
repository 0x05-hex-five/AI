import os
import shutil

image_dir = r"C:\Users\daniel\Capstone\AI\yolo\datasets\images\train"
label_dir = r"C:\Users\daniel\Capstone\AI\yolo\datasets\labels\train"

image_files = list(os.listdir(image_dir))
label_files = list(os.listdir(label_dir))

# remove files that don't have corresponding labels
for image_name in image_files:
    label_name = image_name.replace('.png', '.txt')
    label_path = os.path.join(label_dir, label_name)
    image_path = os.path.join(image_dir, image_name)
    
    # 상응하는 이미지 파일 없으면 삭제
    if not os.path.exists(label_path):
        print(f"⚠️ 라벨 파일 없음: {label_path}")
        os.remove(image_path) # Optionally remove the image if the image is not found

image_files = list(os.listdir(image_dir))
label_files = list(os.listdir(label_dir))

print(len(image_files), len(label_files))