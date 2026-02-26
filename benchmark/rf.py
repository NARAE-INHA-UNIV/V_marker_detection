import os
from dotenv import load_dotenv
import roboflow
from roboflow import Roboflow


load_dotenv()
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")
PROJECT_NAME = "all-54j1x" # Rohang25 프로젝트 ID
VERSION = 1


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


if __name__ == "__main__":
    dataset_path = download_dataset()
    print(f"Dataset downloaded to: {dataset_path}")