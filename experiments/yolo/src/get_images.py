import os
import shutil
import glob

groups = ['1.Training', '2.Validation']
med_types = ['경구약제조합 5000종', '단일경구약제 5000종']

root = r"E:\dataset\166.약품식별 인공지능 개발을 위한 경구약제 이미지 데이터\01.데이터"
output_dir = r"C:\Users\daniel\Capstone\AI\yolo\datasets\images"

# 카메라 각도별로 특정 이미지들만 필터링하는 함수
def get_filtered_images(image_root, camera_la=90, camera_lo=000):
    pattern = f'*_{camera_la}_{camera_lo:03d}_*.png'
    return glob.glob(os.path.join(image_root, pattern))

camera_las = range(50, 95, 5) # 카메라 위도
camera_los = range(0, 380, 20) # 카메라 경도

os.makedirs(output_dir, exist_ok=True)

for group in groups:
    for med_type in med_types:
        base_path = os.path.join(root, group, '원천데이터', med_type)

        if not os.path.isdir(base_path):
            continue

        for folder in os.listdir(base_path):
            image_data_root = os.path.join(base_path, folder)
            
            if not os.path.isdir(image_data_root):
                continue

            for file in os.listdir(image_data_root):
                image_path = os.path.join(image_data_root, file)
                # 지정한 위도-경도 조합으로만 필터링
                for la in camera_las:
                    for lo in camera_los:
                        image_datas = get_filtered_images(image_path, camera_la=la, camera_lo=lo)

                        if image_datas:
                            for image_data in image_datas:
                                group_type = 'train' if 'Training' in image_data else 'val'
                                output_path = os.path.join(output_dir, group_type)
                                os.makedirs(output_path, exist_ok=True)

                                # 지정 디렉토리에 이미지 복사
                                shutil.copy(image_data, os.path.join(output_path, os.path.basename(image_data)))
                                # os.remove(image_data)