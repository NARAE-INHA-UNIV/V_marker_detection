import os
import time
import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from ultralytics import YOLO
from jtop import jtop
import gc
import torch
import yaml
from dotenv import load_dotenv

print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
load_dotenv()

BASE_DIR = "runs/detect"
MODEL_PATHS = {
    '1. v11n (Vanilla)': os.path.join(BASE_DIR, 'yolo11n_Inair/weights/yolov11n.pt'),
    '2. v11n (FP16)':    os.path.join(BASE_DIR, 'yolo11n_Inair/weights/yolo11n_fp16.engine'),
    '3. v11n (INT8)':    os.path.join(BASE_DIR, 'yolo11n_Inair/weights/yolo11n_int8.engine'),
    '4. v11s (Vanilla)': os.path.join(BASE_DIR, 'yolo11s_Inair/weights/yolov11s.pt'), 
    '5. v11s (FP16)':    os.path.join(BASE_DIR, 'yolo11s_Inair/weights/yolo11s_fp16.engine'),
    '6. v11s (INT8)':    os.path.join(BASE_DIR, 'yolo11s_Inair/weights/yolo11s_int8.engine')
}

OUTPUT_DIR = "benchmark_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def calculate_f1(precision, recall, eps=1e-7):
    return 2 * (precision * recall) / (precision + recall + eps)

def measure_pipeline_speed(model, image_paths, jetson, num_samples=100):
    timings = {'decoding': [], 'inference': [], 'encoding': [], 'e2e': []}
    resources = {'cpu': [], 'gpu': [], 'ram': [], 'power': []}
    
    # 웜업 (Warm-up)
    for _ in range(10):
        model(image_paths[0], verbose=False)
    
    print(f"  -> 자원 소모량 및 추론 속도 측정 중 ({num_samples} frames)...")
    
    for i in range(num_samples):
        img_path = image_paths[i % len(image_paths)]
        
        # [속도 측정 파이프라인]
        t0 = time.perf_counter()
        frame = cv2.imread(img_path)
        t1 = time.perf_counter()
        
        results = model(frame, verbose=False)[0]
        t2 = time.perf_counter()
        
        _ = results.plot() 
        t3 = time.perf_counter()
        
        timings['decoding'].append((t1 - t0) * 1000)
        timings['inference'].append((t2 - t1) * 1000)
        timings['encoding'].append((t3 - t2) * 1000)
        timings['e2e'].append((t3 - t0) * 1000)
        
        # [자원 측정 파이프라인 - 회피 없이 정밀 타격]
        
        # 1. CPU (모든 코어 긁어모아서 정확한 평균 도출)
        cpu_keys = [k for k in jetson.stats.keys() if k.startswith('CPU') and k != 'CPU']
        if cpu_keys:
            cpu_val = np.mean([jetson.stats[k] for k in cpu_keys])
        else:
            cpu_val = jetson.stats.get('CPU', 0.0)
            
        # 2. GPU
        gpu_val = jetson.stats.get('GPU', 0.0)
        
        # 3. RAM (정확한 퍼센테이지 추출)
        try:
            ram_val = (jetson.memory['RAM']['used'] / jetson.memory['RAM']['tot']) * 100.0
        except:
            ram_val = jetson.stats.get('RAM', 0.0)
            
        # 4. Power (Orin Nano의 모든 전력 센서값을 끌어모아 총 전력(mW) 계산)
        power_val = 0.0
        if hasattr(jetson, 'power') and isinstance(jetson.power, dict):
            if 'tot' in jetson.power and 'power' in jetson.power['tot']:
                power_val = jetson.power['tot']['power']
            else:
                # 메인 센서가 없다면 모든 센서 전력을 합산
                power_val = sum([v['power'] for k, v in jetson.power.items() if isinstance(v, dict) and 'power' in v])
        
        resources['cpu'].append(cpu_val)
        resources['gpu'].append(gpu_val)
        resources['ram'].append(ram_val)
        resources['power'].append(power_val)

    return timings, resources

def plot_time_series(all_data, output_dir):
    metrics = [
        ('power', 'Power Consumption Over Time', 'Power (mW)'),
        ('gpu',   'GPU Utilization Over Time',   'GPU Util (%)'),
        ('ram',   'RAM Utilization Over Time',   'RAM Util (%)'),
        ('cpu',   'CPU Utilization Over Time',   'CPU Util (%)')
    ]
    
    for key, title, ylabel in metrics:
        plt.figure(figsize=(14, 7))
        for model_name, res_data in all_data.items():
            if key not in res_data: continue
            plt.plot(res_data[key], label=model_name, alpha=0.8, linewidth=1.5)
        plt.title(title)
        plt.xlabel('Frame Index (Time)')
        plt.ylabel(ylabel)
        plt.legend(loc='upper right', fontsize='small')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"timeline_{key}.png"), dpi=300)
        plt.close()

def main():
    data_yaml_path = "./all-1/data.yaml"
    valid_img_dir = os.path.dirname(data_yaml_path) + "/valid/images"
    valid_images = [os.path.join(valid_img_dir, f) for f in os.listdir(valid_img_dir) if f.endswith(('.jpg', '.png'))]

    if not valid_images:
        print("Error: No Datasets found.")
        return

    final_results = []
    all_resource_data = {}

    with jtop() as jetson:
        if not jetson.ok():
            print("Error: jtop is not running!")
            return

        for model_name, model_path in MODEL_PATHS.items():
            print(f"\n=============================================")
            print(f"Testing Model: {model_name}")
            print(f"Path: {model_path}")
            
            if not os.path.exists(model_path):
                print(f"[!] File not found: {model_path} -> 건너뜁니다.")
                continue

            from ultralytics.nn.autobackend import AutoBackend
            if not hasattr(AutoBackend, '_is_patched'):
                _original_init = AutoBackend.__init__
                def _patched_init(self, *args, **kwargs):
                    _original_init(self, *args, **kwargs)
                    
                    # 1. 이름표(names) 주입
                    if getattr(self, 'names', None) is None:
                        import yaml
                        with open(data_yaml_path, 'r', encoding='utf-8') as f:
                            d = yaml.safe_load(f)
                        self.names = {i: n for i, n in enumerate(d.get('names', []))}
                    
                    # 2. 메타데이터(metadata) 주입 (에러 발생 지점 해결)
                    if not hasattr(self, 'metadata') or self.metadata is None:
                        self.metadata = {} # 빈 딕셔너리라도 넣어줘야 .get()이 작동합니다.
                
                AutoBackend.__init__ = _patched_init
                AutoBackend._is_patched = True

            # 모델 불러오기
            model = YOLO(model_path, task='detect')

            # 성능(Performance) 측정 (예외 처리 없이 무조건 들이받습니다)
            print(f"  -> 성능(Precision, Recall, mAP50) 정밀 측정 중...")
            metrics = model.val(data=data_yaml_path, imgsz=640, device='cuda', plots=False, verbose=False)
            
            precision = metrics.results_dict['metrics/precision(B)']
            recall = metrics.results_dict['metrics/recall(B)']
            map50 = metrics.results_dict['metrics/mAP50(B)']
            f1_score = calculate_f1(precision, recall)
            
            # 속도 및 자원(Speed & Resource) 측정
            timings, resources = measure_pipeline_speed(model, valid_images, jetson)

            avg_dec = np.mean(timings['decoding'])
            avg_inf = np.mean(timings['inference'])
            avg_enc = np.mean(timings['encoding'])
            avg_e2e = np.mean(timings['e2e'])

            dec_fps = 1000.0 / avg_dec if avg_dec > 0 else 0
            inf_fps = 1000.0 / avg_inf if avg_inf > 0 else 0
            enc_fps = 1000.0 / avg_enc if avg_enc > 0 else 0
            e2e_fps = 1000.0 / avg_e2e if avg_e2e > 0 else 0
            
            avg_gpu = np.mean(resources['gpu'])
            avg_ram = np.mean(resources['ram'])
            avg_cpu = np.mean(resources['cpu'])
            avg_power = np.mean(resources['power'])
            
            # 측정된 모든 데이터 저장
            final_results.append({
                'Model': model_name,
                'Precision': precision,
                'Recall': recall,
                'mAP50': map50,
                'F1-Score': f1_score,
                'Decode(ms)': avg_dec,
                'Inference(ms)': avg_inf,
                'Encode(ms)': avg_enc,
                'E2E(ms)': avg_e2e,
                'Decode_FPS': dec_fps,
                'Inference_FPS': inf_fps,
                'Encode_FPS': enc_fps,
                'E2E_FPS': e2e_fps,
                'CPU_Util(%)': avg_cpu,
                'GPU_Util(%)': avg_gpu,
                'RAM(%)' : avg_ram,
                'Power(mW)': avg_power
            })
            
            all_resource_data[model_name] = resources
            
            del model
            gc.collect()
            time.sleep(2) # 다음 모델 측정을 위한 쿨다운

    # 결과 CSV 저장 및 화면 출력
    df = pd.DataFrame(final_results)
    csv_path = os.path.join(OUTPUT_DIR, "yolo11_jetson_benchmark.csv")
    df.to_csv(csv_path, index=False)
    
    print("\n\n✅ [최종 벤치마크 결과]")
    # 터미널에서 표가 잘리지 않게 설정
    pd.set_option('display.max_columns', None) 
    pd.set_option('display.width', 1000)
    print(df)

    # 전력(Power), CPU를 포함한 모든 그래프 파일 생성
    print("\n  -> 그래프 파일 생성 중...")
    plot_time_series(all_resource_data, OUTPUT_DIR)
    
    plt.figure(figsize=(12, 6))
    df_melt_lat = df.melt(id_vars='Model', value_vars=['Decode(ms)', 'Inference(ms)', 'Encode(ms)'], var_name='Stage', value_name='Latency(ms)')
    sns.barplot(x='Model', y='Latency(ms)', hue='Stage', data=df_melt_lat)
    plt.title('Latency Breakdown per Stage')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "latency_breakdown.png"))

    plt.figure(figsize=(14, 7))
    fps_cols = ['Decode_FPS', 'Inference_FPS', 'Encode_FPS', 'E2E_FPS']
    df_melt_fps = df.melt(id_vars='Model', value_vars=fps_cols, var_name='Stage', value_name='FPS')
    sns.barplot(x='Model', y='FPS', hue='Stage', data=df_melt_fps, palette='magma')
    plt.title('FPS Performance by Pipeline Stage')
    plt.ylabel('Frames Per Second')
    plt.xticks(rotation=45)

    for container in plt.gca().containers:
        plt.gca().bar_label(container, fmt='%.1f', fontsize=8)
        
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fps_breakdown.png"))
    print("✅ 모든 항목 측정 완료! benchmark_results 폴더를 확인해 주세요.")

if __name__ == "__main__":
    main()