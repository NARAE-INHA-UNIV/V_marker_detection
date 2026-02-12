import os
import time
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ultralytics import YOLO
from roboflow import Roboflow
from jtop import jtop
import gc


ROBOFLOW_API_KEY = ".env API KEY 넣기" 
PROJECT_NAME = "all-54j1x" # Rohang25 프로젝트 ID
VERSION = 1

# 파일 경로 설정
BASE_DIR = "./V_marker_detection/runs/detect"
MODEL_PATHS = {
    '1. v11n (Vanilla)': os.path.join(BASE_DIR, 'yolo11n_lnair/weights/yolov11n.pt'),
    '2. v11n (FP16)':    os.path.join(BASE_DIR, 'yolo11n_lnair/weights/yolo11n_fp16.engine'),
    '3. v11n (INT8)':    os.path.join(BASE_DIR, 'yolo11n_lnair/weights/yolo11n_int8.engine'),
    
    '4. v11s (Vanilla)': os.path.join(BASE_DIR, 'yolo11s_lnair/weights/yolov11s.pt'),
    '5. v11s (FP16)':    os.path.join(BASE_DIR, 'yolo11s_lnair/weights/yolo11s_fp16.engine'),
    '6. v11s (INT8)':    os.path.join(BASE_DIR, 'yolo11s_lnair/weights/yolo11s_int8.engine')
}

# 결과 저장 경로
OUTPUT_DIR = "benchmark_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 데이터셋
def download_dataset():
    if os.path.exists(f"{PROJECT_NAME}-{VERSION}"):
        print("Dataset already exists.")
        return os.path.abspath(f"{PROJECT_NAME}-{VERSION}/data.yaml")
    
    try:
        rf = Roboflow(api_key=ROBOFLOW_API_KEY)
        project = rf.workspace("rohang25").project(PROJECT_NAME)
        dataset = project.version(VERSION).download("yolov11")
        return os.path.abspath(dataset.location + "/data.yaml")
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        exit()

# 측정 함수
def calculate_f1(precision, recall, eps=1e-7):
    # F1 Score 계산
    return 2 * (precision * recall) / (precision + recall + eps)

def measure_pipeline_speed(model, image_paths, jetson, num_samples=100):
    # Decoding -> Inference -> Encode/Overlay
  
    timings = {
        'decoding': [],
        'inference': [],
        'encoding': [], # 오버레이 포함
        'e2e': []
    }
    
    resources = {
        'cpu': [], 'gpu': [], 'ram': [], 'power': []
    }
    
    # Warm-up
    for _ in range(10):
        model(image_paths[0], verbose=False)
    
    print(f"Measuring speed & resources on {num_samples} frames")
    
    # 프레임(이미지) 하나씩 루프
    for i in range(num_samples):
        img_path = image_paths[i % len(image_paths)]
        
        # 1. 디코딩
        t0 = time.perf_counter()
        frame = cv2.imread(img_path)
        t1 = time.perf_counter()
        
        # 2. 추론
        results = model(frame, verbose=False)[0]
        t2 = time.perf_counter()
        
        # 3. 인코딩 / 오버레이 (추론 결과(box) 그리기)
        _ = results.plot() 
        
        # 오버레이 후 화면 출력 직전까지 측정
        t3 = time.perf_counter()
        
        # Latency 계산 (ms)
        dec_time = (t1 - t0) * 1000
        inf_time = (t2 - t1) * 1000
        enc_time = (t3 - t2) * 1000
        total_time = (t3 - t0) * 1000
        
        timings['decoding'].append(dec_time)
        timings['inference'].append(inf_time)
        timings['encoding'].append(enc_time)
        timings['e2e'].append(total_time)
        
        # 자원 소모량 (Snapshot)
        if hasattr(jetson.cpu, 'values'):
            cpu_val = np.mean(list(jetson.cpu.values())) if isinstance(jetson.cpu, dict) else 0
        elif 'CPU1' in jetson.stats:
            cpu_val = jetson.stats['CPU1']
        else:
            cpu_val = np.mean(list(jetson.cpu['util'].values()))

        if 'util' in jetson.memory['RAM']:
            ram_val = jetson.memory['RAM']['util']
        else:
            ram_val = (jetson.memory['RAM']['used'] / jetson.memory['RAM']['tot']) * 100

        resources['cpu'].append(cpu_val)
        resources['gpu'].append(jetson.gpu['util'] if jetson.gpu else 0)
        resources['ram'].append(ram_val)
        resources['power'].append(jetson.power['tot']['power'])

    return timings, resources

def plot_time_series(all_data, output_dir):
    
    # 각 모델별 리소스(Power, GPU, RAM) 변화량을 Line Chart로 저장
    # X축: 프레임(Sample) 순서 / Y축: 리소스 사용량
   
    metrics = [
        ('power', 'Power Consumption Over Time', 'Power (mW)'),
        ('gpu',   'GPU Utilization Over Time',   'GPU Util (%)'),
        ('ram',   'RAM Utilization Over Time',   'RAM Util (%)'),
        ('cpu',   'CPU Utilization Over Time',   'CPU Util (%)')
    ]
    
    for key, title, ylabel in metrics:
        plt.figure(figsize=(14, 7))
        
        for model_name, res_data in all_data.items():
            if key not in res_data: continue # 없으면 스킵
            # 데이터 로드

            values = res_data[key]
            # 그래프 그리기 (투명도 살짝 줘서 겹쳐도 보이게)
            plt.plot(values, label=model_name, alpha=0.8, linewidth=1.5)
            
        plt.title(title)
        plt.xlabel('Frame Index (Time)')
        plt.ylabel(ylabel)
        plt.legend(loc='upper right', fontsize='small')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        
        # 파일 저장
        filename = f"timeline_{key}.png"
        plt.savefig(os.path.join(output_dir, filename), dpi=300)
        plt.close() # 메모리 해제


# 메인 실행 루프 
def main():
    data_yaml_path = download_dataset()
    
    # 테스트 이미지 불러오기
    valid_img_dir = os.path.dirname(data_yaml_path) + "/valid/images"
    valid_images = [os.path.join(valid_img_dir, f) for f in os.listdir(valid_img_dir) if f.endswith(('.jpg', '.png'))]
    
    if not valid_images:
        print("Error: No Datasets")
        return

    final_results = []
    all_resource_data = {}

    with jtop() as jetson:
        if not jetson.ok():
            return

        for model_name, model_path in MODEL_PATHS.items():
            print(f"Testing Model: {model_name}")
            print(f"Path: {model_path}")
            print(f"=============================================\n")
            
            if not os.path.exists(model_path):
                print(f"File not found: {model_path}.")
                continue

            try:
                # 모델 불러오기
                model = YOLO(model_path, task='detect')
                
                metrics = model.val(data=data_yaml_path, verbose=False, plots=False)
                precision = metrics.results_dict['metrics/precision(B)']
                recall = metrics.results_dict['metrics/recall(B)']
                map50 = metrics.results_dict['metrics/mAP50(B)']
                f1_score = calculate_f1(precision, recall)
                
                timings, resources = measure_pipeline_speed(model, valid_images, jetson)

                # 평균적으로 한 프레임을 처리하는데 얼마나 걸리는지
                avg_dec = np.mean(timings['decoding'])
                avg_inf = np.mean(timings['inference'])
                avg_enc = np.mean(timings['encoding'])
                avg_e2e = np.mean(timings['e2e'])

                # 평균적으로 FPS가 얼마나 나오는지
                dec_fps = 1000.0 / avg_dec if avg_dec > 0 else 0
                inf_fps = 1000.0 / avg_inf if avg_inf > 0 else 0
                enc_fps = 1000.0 / avg_enc if avg_enc > 0 else 0
                e2e_fps = 1000.0 / avg_e2e if avg_e2e > 0 else 0
                
                avg_gpu = np.mean(resources['gpu'])
                avg_ram = np.mean(resources['ram'])
                avg_cpu = np.mean(resources['cpu'])
                avg_power = np.mean(resources['power'])
                
                # Store Data
                final_results.append({
                    'Model': model_name,
                    'mAP50': map50,
                    'Precision': precision,
                    'Recall': recall,
                    'F1-Score': f1_score,
                    # Latency
                    'Decode(ms)': avg_dec,
                    'Inference(ms)': avg_inf,
                    'Encode(ms)': avg_enc,
                    'E2E(ms)': avg_e2e,
                    # FPS
                    'Decode_FPS': dec_fps,
                    'Inference_FPS': inf_fps,
                    'Encode_FPS': enc_fps,
                    'E2E_FPS': e2e_fps,
                    # Resources
                    'CPU_Util(%)': avg_cpu,
                    'GPU_Util(%)': avg_gpu,
                    'RAM(%)' : avg_ram,
                    'Power(mW)': avg_power
                })
                
                all_resource_data[model_name] = resources
                
                # Memory Cleanup
                del model
                import gc
                gc.collect()
                time.sleep(2) # Cooldown
                
            except Exception as e:
                print(f"Error testing {model_name}: {e}")

    # 결과 저장 및 시각화
    df = pd.DataFrame(final_results)
    csv_path = os.path.join(OUTPUT_DIR, "yolo11_jetson_benchmark.csv")
    df.to_csv(csv_path, index=False)
    print(df)

    # Power (Timeline)
    plot_time_series(all_resource_data, OUTPUT_DIR)
    
    # Latency (Bar Chart)
    plt.figure(figsize=(12, 6))
    df_melt_lat = df.melt(id_vars='Model', value_vars=['Decode(ms)', 'Inference(ms)', 'Encode(ms)'], var_name='Stage', value_name='Latency(ms)')
    sns.barplot(x='Model', y='Latency(ms)', hue='Stage', data=df_melt_lat)
    plt.title('Latency Breakdown per Stage')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "latency_breakdown.png"))

    # FPS (Bar Chart)
    plt.figure(figsize=(14, 7))
    fps_cols = ['Decode_FPS', 'Inference_FPS', 'Encode_FPS', 'E2E_FPS']
    df_melt_fps = df.melt(id_vars='Model', value_vars=fps_cols, var_name='Stage', value_name='FPS')
    sns.barplot(x='Model', y='FPS', hue='Stage', data=df_melt_fps, palette='magma')
    plt.title('FPS Performance by Pipeline Stage')
    plt.ylabel('Frames Per Second (Log Scale if variance is high)')
    plt.xticks(rotation=45)

    for container in plt.gca().containers:
        plt.gca().bar_label(container, fmt='%.1f', fontsize=8)
        
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fps_breakdown.png"))

if __name__ == "__main__":
    main()



"""
  [YOLOv11 성능 및 자원 측정]

    측정 파이프라인: 디코딩 -> 추론 -> 인코딩/오버레이 (실제 비디오 처리 파이프라인 적용)
    
    1. Speed
        - Mean latency / MEAN FPS
        - Inference latency
        - End-to-End latency
        - Encoding + Overlay latency 
        - Decoding latency

    2. Performance
        - Precision
        - Recall
        - F1-score
        - mAP50

    3. Resource
        - CPU
        - GPU
        - RAM
        - Power(전력소모)

    [데이터셋] - API 
    https://universe.roboflow.com/rohang25/all-54j1x

    
    [모델 변인]
    1. YOLOv11 nano (Vanilla - FP32)
    2. YOLOv11 nano FP16 양자화 버전
    3. YOLOv11 nano int8 양자화 버전

    4. YOLOv11 small (Vanilla - FP32)
    5. YOLOv11 small FP16 양자화 버전
    6. YOLOv11 small int8 양자화 버전

"""

