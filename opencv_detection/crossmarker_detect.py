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
    min_rectangularity=0.4,
    min_aspect_ratio=0.45,
    min_solidity=0.75,
    quad_only=True,
    approx_epsilon_ratio=0.02,
):
    """
    HSV Hue 기반으로 적색 영역을 검출하고, cv2.minAreaRect로 회전 외접 사각형을 계산한 뒤
    직사각형도·종횡비·견고도로 필터링해 빨간 사각형 마커만 반환. 자글자글하거나 길쭉한 노이즈 제거.
    quad_only=True이면 approxPolyDP로 4꼭짓점(사각형) 형태만 통과시킴.
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

        # 회전 외접 사각형 계산 (minAreaRect)
        min_rect = cv2.minAreaRect(cnt)
        (rx, ry), (rw, rh), angle = min_rect
        rect_area = rw * rh
        if rect_area <= 0:
            continue
        rectangularity = area / rect_area  # 1에 가까울수록 직사각형에 가까움
        if rectangularity < min_rectangularity:
            continue

        # 노이즈 필터: 빨간 사각형에서 많이 벗어난 형태 제거
        # 종횡비 — 너무 길쭉한 것(선·자글자글한 잡음) 제외. 정사각형에 가까울수록 1
        side_min, side_max = min(rw, rh), max(rw, rh)
        aspect_ratio = side_min / side_max if side_max > 0 else 0
        if aspect_ratio < min_aspect_ratio:
            continue

        # 견고도(solidity) — 컨투어 면적/볼록껍질 면적. 자글자글한 형태는 값이 낮음
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity = (area / hull_area) if hull_area > 0 else 0
        if solidity < min_solidity:
            continue

        # 사각형(4꼭짓점) 형태만 허용 — 다각형 근사 후 꼭짓점 수로 노이즈 제거
        if quad_only:
            peri = cv2.arcLength(cnt, True)
            epsilon = approx_epsilon_ratio * peri
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            if len(approx) != 4:
                continue

        box_points = np.asarray(cv2.boxPoints(min_rect), dtype=np.int32)

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
            "min_rect": min_rect,
            "min_rect_box": box_points,
            "rect_area": rect_area,
            "rectangularity": rectangularity,
            "aspect_ratio": aspect_ratio,
            "solidity": solidity,
        })

    # 면적 기준 정렬 (큰 것부터)
    regions.sort(key=lambda r: r["area"], reverse=True)
    return red_mask, regions


def draw_crossmarkers(frame, regions, draw_contour=True, draw_label=True, draw_min_rect=True):
    """검출된 적색(크로스 마커) 영역 그리기 및 라벨링. 회전 외접 사각형(minAreaRect) 옵션."""
    out = frame.copy()
    for idx, r in enumerate(regions):
        cx, cy = r["center"]
        x, y, w, h = r["bbox"]
        area = r["area"]
        cnt = r["contour"]

        if draw_contour:
            cv2.drawContours(out, [cnt], -1, (0, 255, 0), 2)
            cv2.circle(out, (cx, cy), 4, (0, 0, 255), -1)

        if draw_min_rect and "min_rect_box" in r:
            box = r["min_rect_box"]
            if len(box) >= 4:
                cv2.drawContours(out, [box], 0, (255, 165, 0), 2)  # 주황색으로 회전 외접 사각형

        if draw_label:
            label = f"CrossMarker {idx + 1}"
            rect_str = f"R:{r.get('rectangularity', 0):.2f}" if "rectangularity" in r else ""
            info = f"A:{int(area)}" + (f" {rect_str}" if rect_str else "")
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
    print("  'q' - 종료   's' - 스크린샷   '+'/'-' - 최소 면적   'r'/'R' - 직사각형도   '4' - 사각형만 필터 토글")

    min_area = 500
    min_rectangularity = 0.4
    min_aspect_ratio = 0.45   # 정사각형에서 벗어난 길쭉한 것 노이즈 제거
    min_solidity = 0.75       # 자글자글한 형태 노이즈 제거
    quad_only = True          # True면 4꼭짓점(사각형) 형태만 표시
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
            min_rectangularity=min_rectangularity,
            min_aspect_ratio=min_aspect_ratio,
            min_solidity=min_solidity,
            quad_only=quad_only,
        )
        output = draw_crossmarkers(frame, regions)

        status = (
            f"Red: {len(regions)} | Area≥{min_area} Rect≥{min_rectangularity:.2f} "
            f"Aspect≥{min_aspect_ratio:.2f} Solid≥{min_solidity:.2f} | QuadOnly:{quad_only} [4] | [m]ask: {show_mask}"
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
        elif key == ord("r"):
            min_rectangularity = max(0.1, min_rectangularity - 0.05)
        elif key == ord("R"):
            min_rectangularity = min(0.95, min_rectangularity + 0.05)
        elif key == ord("m"):
            show_mask = not show_mask
        elif key == ord("4"):
            quad_only = not quad_only

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
