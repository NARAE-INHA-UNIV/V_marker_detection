import cv2
import time
import os
import json
from ultralytics import YOLO



def measure_performance(model_name, model_path, video_path):

    model = YOLO(model_path, task="detect")
    
    video = cv2.VideoCapture(video_path)

    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(f'res_{model_name}.mp4', fourcc, 30, (width, height))

    total_inference_time = 0
    inference_time = 0
    encoding_time = 0
    decoding_time = 0
    frame_count = 0

    while video.isOpened():
        start_total = time.perf_counter() # 전체 시작

        start_dec = time.perf_counter() # 디코딩 시작
        ret, frame = video.read()
        if not ret:
            break
        end_dec = time.perf_counter() # 디코딩 끝

        start_inf = time.perf_counter() # 추론 시작
        results = model.predict(
        source=frame,   # 이미지 / 비디오 / 웹캠
        device=None,        # TensorRT는 GPU 필수
        conf=0.25,
        iou=0.7,
        stream=False
    ) 
        end_inf = time.perf_counter() # 추론 끝

        annotated_frame = results[0].plot() # 결과 그림 그리기

        start_enc = time.perf_counter() # 인코딩 시작
        out.write(annotated_frame)
        end_enc = time.perf_counter() # 인코딩 끝

        end_total = time.perf_counter() # 전체 끝

        total_inference_time += (end_total - start_total)
        inference_time += (end_inf - start_inf)
        encoding_time += (end_enc - start_enc)
        decoding_time += (end_dec - start_dec)
        frame_count += 1

    video.release()
    out.release()

    total_fps = frame_count / total_inference_time
    inf_fps = frame_count / inference_time
    enc_fps = frame_count / encoding_time
    dec_fps  = frame_count / decoding_time
    
    return total_fps, inf_fps, enc_fps, dec_fps

if __name__ == "__main__":

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    # 상대 경로 설정
    video_path = os.path.join(BASE_DIR, "test.mp4")
    models = {
        "yolo11_nano": os.path.join(BASE_DIR, "yolo11n_fp16.engine"),
        "yolo11_small": os.path.join(BASE_DIR, "yolo11s_fp16.engine")
    }

    results = {}

    for model, path in models.items():
        total_fps, inf_fps, enc_fps, dec_fps = measure_performance(model, path, video_path=video_path)
        results[model] = f"total_fps: {total_fps}, inf_fps: {inf_fps}, enc_fps: {enc_fps}, dec_fps: {dec_fps}"

    print(json.dumps(results, indent=4))
                          



