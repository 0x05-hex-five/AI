import os
import json
import glob

groups = ['1.Training', '2.Validation']
med_types = ['경구약제조합 5000종', '단일경구약제 5000종']

root = r"E:\dataset\166.약품식별 인공지능 개발을 위한 경구약제 이미지 데이터\01.데이터"
output_dir = r"C:\Users\daniel\Capstone\AI\yolo\datasets\labels"

# COCO 형식의 JSON 파일을 YOLO 형식의 텍스트 파일로 변환하는 함수
def convert_to_yolo(json_path, save_dir):
    # JSON 파일 읽기
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 이미지 정보에서 폭, 높이 가져오기
    image_info = data['images'][0]
    img_width = image_info['width']
    img_height = image_info['height']
    img_name = image_info['file_name']

    annotations = data['annotations']

    yolo_labels = []

    # 각 바운딩 박스를 YOLO 형식으로 변환
    for ann in annotations:
        x, y, w, h = ann['bbox']
        x_center = (x + w / 2) / img_width
        y_center = (y + h / 2) / img_height
        width = w / img_width
        height = h / img_height
        yolo_labels.append(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

    # 이미지 이름을 기반으로 .txt 라벨 파일 생성
    txt_name = img_name.replace('.png', '.txt')
    txt_path = os.path.join(save_dir, txt_name)

    with open(txt_path, 'w') as f:
        f.write('\n'.join(yolo_labels))

    # os.remove(json_path)

    return txt_name

# 카메라 각도별로 특정 JSON 파일들만 필터링하는 함수
def get_filtered_jsons(json_root, camera_la=90, camera_lo=000):
    pattern = f'*_{camera_la}_{camera_lo:03d}_*.json'
    return glob.glob(os.path.join(json_root, pattern))

camera_las = [60, 70, 75, 90] # 카메라 위도
camera_los = range(0, 380, 20) # 카메라 경도

os.makedirs(output_dir, exist_ok=True)

# 모든 데이터 조합에 대해 순회하며 YOLO 라벨로 변환
for group in groups:
    for med_type in med_types:
        base_path = os.path.join(root, group, '라벨링데이터', med_type)
        
        if not os.path.isdir(base_path):
            continue
        
        for folder in os.listdir(base_path):
            json_root = os.path.join(base_path, folder)

            # 지정한 위도-경도 조합으로만 필터링
            for la in camera_las:
                for lo in camera_los:
                    json_paths = get_filtered_jsons(json_root, camera_la=la, camera_lo=lo)
                    
                    if json_paths:
                        for json_path in json_paths:
                            group_type = 'train' if 'Training' in json_path else 'val'
                            output_path = os.path.join(output_dir, group_type)
                            os.makedirs(output_path, exist_ok=True)
                            txt_name = convert_to_yolo(json_path, output_path)
                            print(f"Converted {json_path} -> {txt_name}")