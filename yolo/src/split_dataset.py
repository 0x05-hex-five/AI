import os
import shutil
import random

'''
훈련용 데이터들로부터 검증용 데이터셋을 생성하는 스크립트입니다.
훈련용 데이터셋에서 일정 비율의 이미지를 무작위로 선택하여 검증용 데이터셋으로 이동합니다.
'''

image_dir = r"C:\Users\daniel\Capstone\AI\yolo\datasets\images\train"
label_dir = r"C:\Users\daniel\Capstone\AI\yolo\datasets\labels\train"
val_image_dir = r"C:\Users\daniel\Capstone\AI\yolo\datasets\images\val"
val_label_dir = r"C:\Users\daniel\Capstone\AI\yolo\datasets\labels\val"

val_ratio = 0.2

random.seed(42)  # 랜덤 시드 설정

image_files = list(os.listdir(image_dir))
label_files = list(os.listdir(label_dir))

val_samples = random.sample(image_files, int(len(image_files) * val_ratio)) # 검증용 데이터셋으로 이동할 이미지 파일 목록
print(f"검증용 데이터셋으로 이동할 이미지 수: {len(val_samples)}")

os.makedirs(val_image_dir, exist_ok=True)
os.makedirs(val_label_dir, exist_ok=True)

moved_count = 0

for image_name in val_samples:
    label_name = image_name.replace('.png', '.txt')
    label_path = os.path.join(label_dir, label_name)
    image_path = os.path.join(image_dir, image_name)
    
    # 라벨 파일이 존재하면 함께 이동
    if os.path.exists(label_path):
        shutil.move(image_path, os.path.join(val_image_dir, image_name))
        shutil.move(label_path, os.path.join(val_label_dir, label_name))
    else:
        print(f"⚠️ 라벨 파일 없음: {label_name}")

    moved_count += 1

print(f"{moved_count} image-label pairs (or images) moved to val.")