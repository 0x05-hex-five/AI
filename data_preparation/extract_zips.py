import os
import zipfile

"""
- AI HUB .tar 압축 파일을 지정된 경로에 해제
- zipfile 모듈을 사용하여 압축 해제
"""

groups = ['1.Training', '2.Validation']
types = ['원천데이터', '라벨링데이터']
med_types = ['경구약제조합 5000종', '단일경구약제 5000종']
root=r"E:\dataset\166.약품식별 인공지능 개발을 위한 경구약제 이미지 데이터\01.데이터"

for group in groups: # 1.Training, 2.Validation
    for dtype in types: # 원천데이터, 라벨링데이터
        for med_type in med_types: # 경구약제조합_5000종, 단일경구약제_5000종
            search_path = os.path.join(root, group, dtype, med_type)

            for path, dir, files in os.walk(search_path): # Traverse the directory tree
                for file in files:
                    if file.endswith('.zip'):
                        zip_path = os.path.join(path, file)

                        print(f"Extracting {zip_path} to {path} \n")
                        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                            zip_ref.extractall(path) # Extract the zip file to the same directory

                        os.remove(zip_path) # Optional delete
                        print(f"Deleted {zip_path} \n") # 
print("Extraction completed.")