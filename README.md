# V_marker_detection
YOLO_model for V_marker detection

## 환경 설정
해당 레포를 받기 위한 bash 명령어 입니다.
```bash
git clone https://github.com/NARAE-INHA-UNIV/V_marker_detection.git
cd V_marker_detection

conda create -n Inair python=3.10 -y
conda activate Inair

pip install ultralytics opencv-python
pip install roboflow
```

## Dataset 받아오기
```bash
python dataset.py
```

## 입력 영상을 넣고 추론하기 
```bash
영상읽기 + 추론 + 결과 그리기 + 저장 까지의 시간을 측정하는 FPS와
추론에 걸리는 FPS를 각각 출력하는 코드를 작성해 주세요

사용하는 영상은 30초 내외 정도로 직접 제작해 주시면 됩니다. 
```

## nano와 small모델 간 성능 비교
```bash
cd ./runs/detect
```
해당 디렉토리에 nano model과 small model의 weight파일이 존재합니다.
두 weight파일의 추론시간을 비교해 주세요 



### V_marker Google Drive
https://drive.google.com/drive/folders/16EDUsnlFm6T7sW9rF5j7IPAtaa4Y80qB?usp=share_link
