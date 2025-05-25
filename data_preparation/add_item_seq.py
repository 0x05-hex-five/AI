import zipfile
import os
import pandas as pd
import json
import openpyxl

"""
This script adds an 'item_seq' column to an Excel file by extracting the sequence information
from JSON files inside ZIP archives. It matches each 'C-Code' from the Excel file with the
corresponding folder in the ZIP files, retrieves the 'item_seq' from the first JSON file found,
and saves the updated data to a new Excel file.
- zip_root: Directory containing ZIP files with labeled data.
- excel_path: Path to the input Excel file containing 'C-Code' values.
- output_path: Path to save the output Excel file with the added 'item_seq' column.
Usage:
    Run this script directly to process the data and generate the output Excel file.
"""

def add_item_seq(zip_root, excel_path, output_path):
    # add item_seq column matching with C-Code
    df = pd.read_excel(excel_path)[['C-Code']].copy()
    df['item_seq'] = ''

    for file in os.listdir(zip_root):
        if not file.endswith('.zip'):
            continue

        zip_path = os.path.join(zip_root, file)
        print(f"Processing: {file}")
        try:
            with zipfile.ZipFile(zip_path, 'r') as z:
                all_files = z.namelist()
                zip_folders = {f.split('/')[0] for f in all_files if '/' in f}

                for i, row in df.iterrows():
                    ccode = str(row['C-Code'])
                    for zip_folder in zip_folders:
                        if ccode in zip_folder:
                            # extract item_seq from the first json
                            json_files = [
                                f for f in all_files
                                if f.endswith('.json') and f.startswith(zip_folder + '/')
                            ]
                            if json_files:
                                first_json = json_files[0]
                                with z.open(first_json) as jf:
                                    data = json.load(jf)
                                    df.at[i, 'item_seq'] = data['images'][0].get('item_seq', '')
                            # break after matching
                            break

        except zipfile.BadZipFile:
            print(f"Bad zip file: {file}")

    df.to_excel(output_path, index=False)
    print(f"저장 완료: {output_path}")

if __name__ == "__main__":
    # define paths
    zip_root = r"C:\Users\daniel\OneDrive - Temple University (1)\Desktop\166.약품식별 인공지능 개발을 위한 경구약제 이미지 데이터\01.데이터\1.Training\라벨링데이터\단일경구약제 5000종"
    excel_path = r"C:\Users\daniel\Capstone\AI\학습데이터.xlsx"
    output_path = r"C:\Users\daniel\Capstone\AI\학습데이터_item_seq.xlsx"

    add_item_seq(zip_root, excel_path, output_path)
