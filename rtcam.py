import cv2
import time
from ultralytics import YOLO

def main():
    w_name = "yolov11n.pt"
    w_loc = "./runs/detect/yolo11n_Inair/weights"
    WEIGHT_PATH = f"{w_loc}/{w_name}"
    cam_id =0
    CONF_THRES = 0.25
    IOU_THRES = 0.5

    model = YOLO(WEIGHT_PATH)
    cap = cv2.VideoCapture(cam_id)
    
    if not cap.isOpened():
        raise RuntimeError("카메라 오류임")


    cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,720)
    prev_time = time.time()
    
    while True:
        # 프레임 읽기
        ret, frame = cap.read()
        
        if not ret:
            print("프레임을 읽어올 수 없습니다.")
            break
        
        results = model.track(
            source = frame,
            persist = True,
            tracker = "bytetrack.yaml",
            conf = CONF_THRES,
            iou = IOU_THRES,
            verbose = False
        )

        annotated_frame = results[0].plot()

        now = time.time()
        fps = 1 / (now - prev_time)
        prev_time = now

        cv2.putText(
            annotated_frame,
            f"FPS: {fps :.1f}",
            (20,40),
            cv2.FONT_HERSHEY_SIMPLEX, #?
            1,
            (0,255,0),
            2
        )


        cv2.imshow("Jetson Orin Nano - USB Cam _ powerranger", annotated_frame)

        

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    

    # 자원 해제
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    i=0
    main()