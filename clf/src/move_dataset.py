import os
import shutil

# YOLO 이미지 추론 결과 폴더더
crop_dir = r"C:\Users\daniel\Capstone\AI\yolo\runs\detect\predict\crops\pill"

# 옮길 대상 폴더
output_dir = r"C:\Users\daniel\Capstone\AI\clf\dataset\crop"

for file_name in os.listdir(crop_dir):
    # 파일 경로
    file_path = os.path.join(crop_dir, file_name)

    # 파일 이름에서 라벨 추출 (예: pill_0.jpg -> 0)
    label = file_name.split("_")[0]
    
    # 라벨 폴더 경로
    label_dir = os.path.join(output_dir, label)

    # 라벨 폴더가 없으면 생성
    if not os.path.exists(label_dir):
        os.makedirs(label_dir)

    # 파일 이동
    shutil.move(file_path, os.path.join(label_dir, file_name))