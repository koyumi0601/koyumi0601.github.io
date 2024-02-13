import cv2
import numpy as np

# 이미지 파일 경로
# image_path = '/home/ko/Documents/GitHub/koyumi0601.github.io/_posts/ImageSignalProcessing/QT/project_build/convex_wide_images/Versana_convex.png'
image_path = '/home/ko/Documents/GitHub/koyumi0601.github.io/_posts/ImageSignalProcessing/QT/project_build/convex_wide_images/image_00001.png'
# 이미지 불러오기
image = cv2.imread(image_path)
image_height, image_width = image.shape[:2]
print("Hello")



# lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)

# result_image = image.copy()
# if lines is not None:
#     for line in lines:
#         rho, theta = line[0]
#         a = np.cos(theta)
#         b = np.sin(theta)
#         x0 = a * rho
#         y0 = b * rho
#         x1 = int(x0 + 1000 * (-b))
#         y1 = int(y0 + 1000 * (a))
#         x2 = int(x0 - 1000 * (-b))
#         y2 = int(y0 - 1000 * (a))
#         cv2.line(result_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
apex_x = 1071
apex_y = -156.566
distance_margin = 200
border_size_top = int(np.ceil(np.abs(apex_y))+distance_margin)  # 추가할 픽셀 수
expanded_image = np.full((image_height + border_size_top, image_width, 3), 16, dtype=np.uint8)
expanded_image[border_size_top:, :, :] = image

# 에지 검출 (Canny 에지 검출 사용)
gray = cv2.cvtColor(expanded_image, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, threshold1=50, threshold2=150) # 직각으로 움직이는 것 잘 함
# edges = cv2.Sobel(gray, cv2.CV_8U, 1, 1, ksize=3) # 왼쪽->오른쪽 검출 잘 함
# edges = cv2.Laplacian(gray, cv2.CV_8U) # 대각선은 잘 못하고 최외곽선은 잘 함
# edges = cv2.Scharr(gray, cv2.CV_8U, 1, 0) # 왼쪽 -> 오른쪽 검출 잘함. 노이즈 많음

# innerRadius:  281 outerRadius:  1013
innerRadius=  281 
outerRadius=  1013
initial_circles = np.array([
    [apex_x, distance_margin, innerRadius],
    [apex_x, distance_margin, outerRadius]
], dtype=np.float32)

circles = cv2.HoughCircles(
    edges, 
    cv2.HOUGH_GRADIENT, 
    dp=1, 
    minDist=50, 
    param1=50, 
    param2=30, 
    minRadius=200, 
    maxRadius=1200, 
    circles=initial_circles,  # 초기 추정값을 포함하는 배열을 전달
    maxCircles=5)  # 찾을 원의 최대 개수 설정
num_circles = circles.shape[1]  # circles 배열의 두 번째 차원 크기
print(num_circles)

if circles is not None:
    circles = np.uint16(np.around(circles))

    for circle in circles[0, :]:
        center_x, center_y, radius = circle

        # 특정 위치 근처의 원 필터링 (예: (100, 100) 위치 근처의 원)
        target_x, target_y = apex_x, distance_margin
        if abs(center_x - target_x) < 20 and abs(center_y - target_y) < 20:
            cv2.circle(expanded_image, (center_x, center_y), radius, (0, 255, 0), 2)

# 결과 이미지 보기
cv2.imshow('Largest Circle', expanded_image)
cv2.waitKey(0)
cv2.destroyAllWindows()