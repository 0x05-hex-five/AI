import os
import shutil

# YOLO inferenced images
crop_dir = r"C:\Users\daniel\Capstone\AI\yolo\runs\detect\predict\crops\pill"

# output path
output_dir = r"C:\Users\daniel\Capstone\AI\dataset\crop"

for file_name in os.listdir(crop_dir):
    file_path = os.path.join(crop_dir, file_name)

    # extract label from file name (예: pill_0.jpg -> 0)
    label = file_name.split("_")[0]
    
    # label folder path
    label_dir = os.path.join(output_dir, label)

    if not os.path.exists(label_dir):
        os.makedirs(label_dir)

    # move file
    shutil.move(file_path, os.path.join(label_dir, file_name))