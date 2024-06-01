import cv2
import numpy as np

# 加載圖像
img1 = cv2.imread("prove.jpg", -1)
img2 = img1.copy()

# 轉換為灰度圖
gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

# 使用雙邊濾波來保護邊緣同時去除噪音
blurred = cv2.bilateralFilter(gray, 9, 75, 75)

# 自適應閾值化
thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

# 形態學操作：膨脹和腐蝕
kernel = np.ones((3, 3), np.uint8)
thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

# 邊緣檢測
edges = cv2.Canny(thresh, 50, 150)

# 霍夫概率直線變換
lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=50, maxLineGap=10)

# 初始化一個列表來存儲檢測到的垂直線
vertical_lines = []

# 檢測並繪製垂直線
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        if abs(angle) < 10 or abs(angle - 180) < 10:
            vertical_lines.append(((x1, y1), (x2, y2)))
            cv2.line(img2, (x1, y1), (x2, y2), (255, 0, 0), 2)

# 霍夫圓變換檢測圓形
circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=1, minDist=50, param1=100, param2=30, minRadius=10, maxRadius=50)

# 初始化變量來存儲檢測到的球的中心點
ball_center = None

# 檢測並繪製圓形
if circles is not None:
    circles = np.round(circles[0, :]).astype("int")
    for (x, y, r) in circles:
        # 添加條件來過濾不合理的圓形檢測結果
        if 20 <= r <= 50:
            cv2.circle(img2, (x, y), r, (0, 255, 0), 4)
            cv2.rectangle(img2, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
            ball_center = (x, y)

# 計算球心到最近垂直線的距離
if ball_center and vertical_lines:
    x_ball, y_ball = ball_center
    min_distance = float('inf')

    for (pt1, pt2) in vertical_lines:
        x1, y1 = pt1
        x2, y2 = pt2
        # 計算點到線的距離
        distance = abs((y2 - y1) * x_ball - (x2 - x1) * y_ball + x2 * y1 - y2 * x1) / np.sqrt((y2 - y1)**2 + (x2 - x1)**2)
        if distance < min_distance:
            min_distance = distance

    # 輸出球心到最近垂直線的距離
    print(f"球心到最近垂直線的距離: {min_distance:.2f} 像素")

    # 將距離標準化
    image_width = img1.shape[1]
    normalized_distance = min_distance / image_width
    print(f"標準化距離: {normalized_distance:.4f}")

    # 判斷球是否出界
    threshold = 0.05
    if normalized_distance > threshold:
        print("球已出界")
    else:
        print("球未出界")

# 保存圖像
#cv2.imwrite("/mnt/data/Ori.jpg", img1)
#cv2.imwrite("/mnt/data/Canny.jpg", edges)
#cv2.imwrite("/mnt/data/dec.jpg", img2)

# 顯示結果圖像
cv2.imshow("Original", img1)
cv2.imshow("Canny", edges)
cv2.imshow("After", img2)
cv2.waitKey()
cv2.destroyAllWindows()
