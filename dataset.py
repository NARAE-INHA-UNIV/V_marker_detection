
from roboflow import Roboflow
rf = Roboflow(api_key="")
project = rf.workspace("rohang25").project("all-54j1x")
version = project.version(1)
dataset = version.download("yolov11")
