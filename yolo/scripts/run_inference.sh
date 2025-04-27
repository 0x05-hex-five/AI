#!/bin/bash

# YOLOv8n inference & cropping script
yolo task=detect mode=predict \
    model=weights/best_TS_70.pt \
    source='inference_imgs/**/*' \
    save_crop=True

# 결과 폴더 경로 자동 감지
LAST_RUN=$(ls -td runs/detect/predict*/ | head -1)

# 시각화된 이미지 및 라벨 파일 삭제
rm -f "$LAST_RUN"/*.jpg
rm -rf "$LAST_RUN/labels"