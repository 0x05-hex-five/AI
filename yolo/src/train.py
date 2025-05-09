from ultralytics import YOLO
model = YOLO('yolov8n.pt')  # load a pretrained model

results = model.train(
    data='/data1/team5_2/yolo/dataset.yaml',
    epochs=5,
    imgsz=640,
    batch=-1, # choose the size of batch automatically depending on GPU
    save=True,
    save_period=5,
    val=True,
    verbose=True,
    plots=True,
    project='runs_pill',
    name='yolov8n_pill_detector',
    device=0 # 0 for GPU, 'cpu' for CPU
)