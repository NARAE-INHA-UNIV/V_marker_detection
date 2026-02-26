import cv2
import numpy as np

def detect_circles(frame, blur_ksize=9, blur_sigma=2):
    """원형 로고를 검출하고 라벨링"""
    # 그레이스케일 변환
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 노이즈 제거를 위한 가우시안 블러 (커널 크기는 홀수여야 함)
    k = max(3, blur_ksize | 1)  # 홀수 보장
    gray_blurred = cv2.GaussianBlur(gray, (k, k), blur_sigma)
    
    # Hough Circle Transform으로 원 검출
    circles = cv2.HoughCircles(
        gray_blurred,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=50,  # 원 간 최소 거리
        param1=100,  # Canny 엣지 검출 상위 임계값
        param2=30,   # 원 검출 임계값 (낮을수록 더 많이 검출)
        minRadius=20,  # 최소 반지름
        maxRadius=200  # 최대 반지름
    )
    
    return circles

def draw_circles(frame, circles):
    """검출된 원을 그리고 라벨링"""
    if circles is not None:
        circles = np.uint16(np.around(circles))
        
        for idx, circle in enumerate(circles[0, :]):
            center_x, center_y, radius = circle
            
            # 원의 중심점 표시 (빨간색)
            cv2.circle(frame, (center_x, center_y), 2, (0, 0, 255), 3)
            
            # 원 그리기 (초록색)
            cv2.circle(frame, (center_x, center_y), radius, (0, 255, 0), 2)
            
            # 라벨 텍스트
            label = f"Circle {idx + 1}"
            info = f"R: {radius}px"
            
            # 라벨 배경 (가독성 향상)
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(
                frame,
                (center_x - 10, center_y - radius - 40),
                (center_x + text_size[0] + 10, center_y - radius - 5),
                (0, 255, 0),
                -1
            )
            
            # 라벨 텍스트 출력
            cv2.putText(
                frame,
                label,
                (center_x, center_y - radius - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2
            )
            
            # 반지름 정보 출력
            cv2.putText(
                frame,
                info,
                (center_x, center_y - radius - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1
            )
            
            # 중심 좌표 출력
            coord_text = f"({center_x}, {center_y})"
            cv2.putText(
                frame,
                coord_text,
                (center_x + 5, center_y + 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 255),
                1
            )
    
    return frame

def main():
    """메인 함수 - 웹캠에서 실시간 원 검출"""
    # 웹캠 열기 (0은 기본 카메라)
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("카메라를 열 수 없습니다.")
        return
    
    print("원형 로고 검출 시작...")
    print("조정 키:")
    print("  'q' - 종료")
    print("  's' - 스크린샷 저장")
    print("  '+' - 검출 민감도 증가")
    print("  '-' - 검출 민감도 감소")
    print("  'b' - 가우시안 블러 강도 증가")
    print("  'n' - 가우시안 블러 강도 감소")
    
    # 검출 파라미터
    sensitivity = 30
    blur_ksize = 9   # 가우시안 블러 커널 크기 (홀수)
    blur_sigma = 2  # 가우시안 블러 시그마
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("프레임을 읽을 수 없습니다.")
            break
        
        # 원 검출 (가우시안 블러 파라미터 적용)
        circles = detect_circles(frame, blur_ksize=blur_ksize, blur_sigma=blur_sigma)
        
        # 검출된 원 그리기
        output_frame = draw_circles(frame.copy(), circles)
        
        # 상태 정보 표시
        status_text = f"Detected: {len(circles[0]) if circles is not None else 0} circles | Sensitivity: {sensitivity} | Blur: {blur_ksize}x{blur_ksize} σ={blur_sigma}"
        cv2.putText(
            output_frame,
            status_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )
        
        # 결과 표시
        cv2.imshow('Circle Logo Detection', output_frame)
        
        # 키 입력 처리
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('s'):
            # 스크린샷 저장
            filename = f"output/circle_detected_{cv2.getTickCount()}.jpg"
            cv2.imwrite(filename, output_frame)
            print(f"스크린샷 저장됨: {filename}")
        elif key == ord('+') or key == ord('='):
            sensitivity = min(100, sensitivity + 5)
            print(f"민감도 증가: {sensitivity}")
        elif key == ord('-'):
            sensitivity = max(10, sensitivity - 5)
            print(f"민감도 감소: {sensitivity}")
        elif key == ord('b'):
            blur_ksize = min(31, blur_ksize + 2)
            blur_sigma = min(10, blur_sigma + 0.5)
            print(f"가우시안 블러 증가: kernel={blur_ksize}, sigma={blur_sigma:.1f}")
        elif key == ord('n'):
            blur_ksize = max(3, blur_ksize - 2)
            blur_sigma = max(0.5, blur_sigma - 0.5)
            print(f"가우시안 블러 감소: kernel={blur_ksize}, sigma={blur_sigma:.1f}")
    
    # 리소스 해제
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
