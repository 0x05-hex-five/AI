import zipfile
import os
import pandas as pd
import openpyxl

def match_ccode_with_zip(zip_root, excel_path, output_path):
    df = pd.read_excel(excel_path)[['C-Code']]
    df['데이터 개수'] = 0
    df['데이터 위치'] = ''

    for file in os.listdir(zip_root):
        if file.endswith('.zip'):
            zip_path = os.path.join(zip_root, file)
            print(f"Processing: {file}")
            try:
                with zipfile.ZipFile(zip_path, 'r') as z:
                    all_files = z.namelist()
                    zip_folders = {f.split('/')[0] for f in all_files if '/' in f}

                    for i, row in df.iterrows():
                        ccode = row['C-Code']
                        for zip_folder in zip_folders:
                            if ccode in zip_folder:
                                file_count = sum(
                                    1 for f in all_files 
                                    if f.endswith('.json') and f.startswith(zip_folder + '/')
                                )
                                df.at[i, '데이터 개수'] = file_count
                                df.at[i, '데이터 위치'] = os.path.splitext(file)[0]
            except zipfile.BadZipFile:
                print(f"Bad zip file: {file}")

    df = df.sort_values(by='데이터 위치')
    df.to_excel(output_path, index=False)
    print(f"저장 완료: {output_path}")

if __name__ == "__main__":
    zip_root = r'C:\Users\daniel\Capstone\AI\166.약품식별 인공지능 개발을 위한 경구약제 이미지 데이터\01.데이터\1.Training\라벨링데이터\단일경구약제 5000종'
    excel_path = r'C:\Users\daniel\Capstone\AI\new 학습데이터 리스트.xlsx'
    output_path = r'C:\Users\daniel\Capstone\AI\new 학습데이터 리스트_결과.xlsx'

    match_ccode_with_zip(zip_root, excel_path, output_path)