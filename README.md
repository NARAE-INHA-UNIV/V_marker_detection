# Benchmark Results

## Speed

![alt text](benchmark/benchmark_results/fps_breakdown.png)
![alt text](benchmark/benchmark_results/latency_breakdown.png)

| Model             | Decode(ms) | Inference(ms) | Encode(ms) | E2E(ms) | Decode_FPS | Inference_FPS | Encode_FPS | E2E_FPS |
| ----------------- | ---------- | ------------- | ---------- | ------- | ---------- | ------------- | ---------- | ------- |
| 1. v11n (Vanilla) | 3.9914     | 34.3536       | 1.1003     | 39.4453 | 250.5399   | 29.1090       | 908.8268   | 25.3516 |
| 2. v11n (FP16)    | 4.7117     | 20.3921       | 4.0537     | 29.1574 | 212.2382   | 49.0386       | 246.6911   | 34.2966 |
| 3. v11n (INT8)    | 5.1701     | 16.5508       | 3.5892     | 25.3101 | 193.4191   | 60.4202       | 278.6152   | 39.5100 |
| 4. v11s (Vanilla) | 3.9211     | 33.0949       | 0.9453     | 37.9613 | 255.0326   | 30.2161       | 1057.8522  | 26.3426 |
| 5. v11s (FP16)    | 6.1534     | 28.1184       | 5.1544     | 39.4262 | 162.5109   | 35.5639       | 194.0093   | 25.3638 |
| 6. v11s (INT8)    | 6.6711     | 21.6819       | 4.6145     | 32.9675 | 149.8995   | 46.1215       | 216.7084   | 30.3329 |

## Performance

| Model             | Precision | Recall | mAP50  | F1-Score |
| ----------------- | --------- | ------ | ------ | -------- |
| 1. v11n (Vanilla) | 0.9975    | 0.9940 | 0.9947 | 0.9957   |
| 2. v11n (FP16)    | 0.9933    | 0.9917 | 0.9933 | 0.9925   |
| 3. v11n (INT8)    | 0.0000    | 0.0000 | 0.0000 | 0.0000   |
| 4. v11s (Vanilla) | 0.9977    | 0.9944 | 0.9946 | 0.9961   |
| 5. v11s (FP16)    | 0.9936    | 0.9908 | 0.9943 | 0.9922   |
| 6. v11s (INT8)    | 0.3674    | 0.1490 | 0.2274 | 0.2121   |

## Resource

![alt text](benchmark/benchmark_results/timeline_cpu.png)
![alt text](benchmark/benchmark_results/timeline_gpu.png)
![alt text](benchmark/benchmark_results/timeline_power.png)
![alt text](benchmark/benchmark_results/timeline_ram.png)

| Model             | CPU_Util(%) | GPU_Util(%) | RAM(%)  | Power(mW)  |
| ----------------- | ----------- | ----------- | ------- | ---------- |
| 1. v11n (Vanilla) | 21.3917     | 61.4590     | 70.5192 | 9209.5000  |
| 2. v11n (FP16)    | 20.8450     | 38.1190     | 72.5439 | 6649.3500  |
| 3. v11n (INT8)    | 22.9300     | 57.2100     | 72.5668 | 6420.1700  |
| 4. v11s (Vanilla) | 22.8967     | 55.2270     | 75.8153 | 13663.7200 |
| 5. v11s (FP16)    | 21.8667     | 67.0320     | 76.2814 | 7359.3900  |
| 6. v11s (INT8)    | 20.4683     | 37.7650     | 76.3412 | 6549.6700  |
