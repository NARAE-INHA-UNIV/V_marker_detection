from ultralytics import YOLO

model = YOLO("/Users/bosung/Desktop/V_Marker/V_marker_detection/runs/detect/yolo11s_Inair/weights/best.pt")  # 원래 모델 파일
model.export(format="onnx") 