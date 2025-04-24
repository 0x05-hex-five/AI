import zipfile
import os
import pandas as pd

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

    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"저장 완료: {output_path}")

if __name__ == '__main__':
    match_ccode_with_zip(
        zip_root=r"C:\Users\daniel\Capstone\AI\166.약품식별 인공지능 개발을 위한 경구약제 이미지 데이터\01.데이터\1.Training\라벨링데이터\단일경구약제 5000종",          # zip 파일들을 모아둔 폴더 경로로
        excel_path=r"C:\Users\daniel\Capstone\AI\clf\학습데이터 리스트.xlsx",        # ccode가 있는 엑셀 파일 경로로
        output_path=r"C:\Users\daniel\Capstone\AI\clf\results\output.csv"
    )