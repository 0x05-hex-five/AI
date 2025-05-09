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