import cv2
import numpy as np

# ------------------------------
# 參數設定
# ------------------------------
cap = cv2.VideoCapture(0)

# 平滑參數，值越大越穩定
SMOOTHING = 0.6
last_green_points = []

# 自動亮度/對比調整函式
def auto_brightness(frame, alpha=1.2, beta=15):
    """
    alpha: 對比度係數 (>1 提高對比)
    beta: 亮度增量
    """
    return cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)

# 形態學去雜訊
def clean_mask(mask):
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)   # 去除小白點
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # 填補小黑洞
    return mask

# 座標平滑
def smooth_point(prev, current):
    if prev is None:
        return current
    x = int(prev[0] * SMOOTHING + current[0] * (1 - SMOOTHING))
    y = int(prev[1] * SMOOTHING + current[1] * (1 - SMOOTHING))
    return (x, y)

# ------------------------------
# 主迴圈
# ------------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 自動亮度調整
    frame = auto_brightness(frame)

    # 高斯模糊去雜訊
    frame = cv2.GaussianBlur(frame, (7,7), 0)

    # BGR → HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # ------------------------------
    # 紅色遮罩（基準點）
    # ------------------------------
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])

    mask_red = cv2.inRange(hsv, lower_red1, upper_red1) + cv2.inRange(hsv, lower_red2, upper_red2)
    mask_red = clean_mask(mask_red)

    # 找紅點
    contours_r, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    ref_point = None
    if contours_r:
        c = max(contours_r, key=cv2.contourArea)
        if cv2.contourArea(c) > 150:   # 過小忽略
            x_r, y_r, w_r, h_r = cv2.boundingRect(c)
            ref_point = (x_r + w_r//2, y_r + h_r//2)
            cv2.circle(frame, ref_point, 8, (0,0,255), -1)
            cv2.putText(frame, "(0,0)", (ref_point[0]+10, ref_point[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

    # ------------------------------
    # 綠色遮罩（多目標）
    # ------------------------------
    lower_green = np.array([35, 100, 50])
    upper_green = np.array([85, 255, 255])
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    mask_green = clean_mask(mask_green)

    contours_g, _ = cv2.findContours(mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    green_centers = []

    for c in contours_g:
        if cv2.contourArea(c) < 500:
            continue
        x_g, y_g, w_g, h_g = cv2.boundingRect(c)
        green_centers.append((x_g + w_g//2, y_g + h_g//2))

    # ------------------------------
    # 平滑綠點
    # ------------------------------
    smoothed_centers = []
    for i, pt in enumerate(green_centers):
        prev = last_green_points[i] if i < len(last_green_points) else None
        smoothed_centers.append(smooth_point(prev, pt))
    last_green_points = smoothed_centers

    # ------------------------------
    # 顯示綠點及相對紅點座標
    # ------------------------------
    if ref_point:
        for i, (gx, gy) in enumerate(smoothed_centers):
            dx = gx - ref_point[0]
            dy = gy - ref_point[1]

            # 畫綠點
            cv2.circle(frame, (gx, gy), 8, (0,255,0), -1)
            # 顯示相對座標
            cv2.putText(frame, f"G{i+1} ({dx},{dy})", (gx+10, gy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            # 畫線
            cv2.line(frame, ref_point, (gx, gy), (0,255,0), 2)

    # ------------------------------
    # 顯示畫面
    # ------------------------------
    cv2.imshow("Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ------------------------------
# 釋放資源
# ------------------------------
cap.release()
cv2.destroyAllWindows()
