from ultralytics import YOLO

model = YOLO("yolo11n.pt") 

results = model.train(
    data="/data01/syeul/Inair/all-1/data.yaml", # data.yaml의 절대 경로
    epochs=100,
    imgsz=640,
    batch=128,
    device=1,
    name="yolo11n_Inair"
)
