import os
import cv2
import numpy as np


def get_red_mask_hsv(hsv, s_min=100, s_max=255, v_min=50, v_max=255):
    """
    HSV 기반 적색 마스크 생성.
    Hue: OpenCV 기준 0~180이므로 적색은 170~180과 0~10 구간.
    """
    # 적색 구간 1: 170 ~ 180 (진한 빨강 쪽)
    lower_red1 = np.array([170, s_min, v_min])
    upper_red1 = np.array([180, s_max, v_max])
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)

    # 적색 구간 2: 0 ~ 10 (밝은 빨강 쪽)
    lower_red2 = np.array([0, s_min, v_min])
    upper_red2 = np.array([10, s_max, v_max])
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)

    # 두 구간 합침
    red_mask = cv2.bitwise_or(mask1, mask2)
    return red_mask


def detect_crossmarker_regions(
    frame,
    hue_low1=0,
    hue_high1=10,
    hue_low2=170,
    hue_high2=180,
    s_min=100,
    s_max=255,
    v_min=50,
    v_max=255,
    min_area=500,
    blur_ksize=5,
):
    """
    HSV Hue 기반으로 적색 영역을 검출하고, 유효한 영역만 반환.
    """
    if blur_ksize >= 3:
        k = blur_ksize if blur_ksize % 2 else blur_ksize + 1
        frame_blur = cv2.GaussianBlur(frame, (k, k), 0)
    else:
        frame_blur = frame

    hsv = cv2.cvtColor(frame_blur, cv2.COLOR_BGR2HSV)

    # Hue 구간: 0~10, 170~180 (적색)
    lower1 = np.array([hue_low1, s_min, v_min])
    upper1 = np.array([hue_high1, s_max, v_max])
    lower2 = np.array([hue_low2, s_min, v_min])
    upper2 = np.array([hue_high2, s_max, v_max])

    mask1 = cv2.inRange(hsv, lower1, upper1)
    mask2 = cv2.inRange(hsv, lower2, upper2)
    red_mask = cv2.bitwise_or(mask1, mask2)

    # 모폴로지로 노이즈 제거
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)

    # 컨투어 추출
    contours, _ = cv2.findContours(
        red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    regions = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        M = cv2.moments(cnt)
        if M["m00"] <= 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        x, y, w, h = cv2.boundingRect(cnt)
        regions.append({
            "contour": cnt,
            "center": (cx, cy),
            "area": area,
            "bbox": (x, y, w, h),
        })

    # 면적 기준 정렬 (큰 것부터)
    regions.sort(key=lambda r: r["area"], reverse=True)
    return red_mask, regions


def draw_crossmarkers(frame, regions, draw_contour=True, draw_label=True):
    """검출된 적색(크로스 마커) 영역 그리기 및 라벨링."""
    out = frame.copy()
    for idx, r in enumerate(regions):
        cx, cy = r["center"]
        x, y, w, h = r["bbox"]
        area = r["area"]
        cnt = r["contour"]

        if draw_contour:
            cv2.drawContours(out, [cnt], -1, (0, 255, 0), 2)
            cv2.circle(out, (cx, cy), 4, (0, 0, 255), -1)

        if draw_label:
            label = f"CrossMarker {idx + 1}"
            info = f"A:{int(area)}"
            cv2.rectangle(out, (x, y - 22), (x + 120, y), (0, 255, 0), -1)
            cv2.putText(
                out, label, (x, y - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1
            )
            cv2.putText(
                out, info, (x, y - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1
            )
            cv2.putText(
                out, f"({cx},{cy})", (cx + 6, cy + 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1
            )
    return out


def main():
    """웹캠 실시간 HSV(Hue) 기반 적색 크로스 마커 검출."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("카메라를 열 수 없습니다.")
        return

    print("HSV(Hue) 기반 크로스 마커 검출 (적색: Hue 170~180, 0~10)")
    print("  'q' - 종료   's' - 스크린샷   '+'/'-' - 최소 면적 조절")

    min_area = 500
    s_min, s_max = 100, 255
    v_min, v_max = 50, 255
    hue_low1, hue_high1 = 0, 10
    hue_low2, hue_high2 = 170, 180
    show_mask = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        red_mask, regions = detect_crossmarker_regions(
            frame,
            hue_low1=hue_low1,
            hue_high1=hue_high1,
            hue_low2=hue_low2,
            hue_high2=hue_high2,
            s_min=s_min,
            s_max=s_max,
            v_min=v_min,
            v_max=v_max,
            min_area=min_area,
        )
        output = draw_crossmarkers(frame, regions)

        status = (
            f"Hue: 0~10, 170~180 | Red: {len(regions)} | "
            f"MinArea: {min_area} | [m]ask: {show_mask}"
        )
        cv2.putText(
            output, status, (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2
        )

        if show_mask:
            mask_bgr = cv2.cvtColor(red_mask, cv2.COLOR_GRAY2BGR)
            output = np.hstack([output, mask_bgr])

        cv2.imshow("CrossMarker (HSV Red)", output)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            os.makedirs("output", exist_ok=True)
            path = f"output/crossmarker_{cv2.getTickCount()}.jpg"
            cv2.imwrite(path, output)
            print(f"저장: {path}")
        elif key in (ord("+"), ord("=")):
            min_area = min(5000, min_area + 100)
        elif key == ord("-"):
            min_area = max(100, min_area - 100)
        elif key == ord("m"):
            show_mask = not show_mask

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
