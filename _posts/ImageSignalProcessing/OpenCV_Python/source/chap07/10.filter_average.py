import numpy as np, cv2

def average_filter(image, ksize):
    rows, cols = image.shape[:2]
    dst = np.zeros((rows, cols), np.uint8)
    center = ksize // 2                                 # 마스크 절반 크기

    for i in range(rows):           # 입력 영상 순회
        for j in range(cols):
            y1, y2 = i - center, i + center + 1        # 마스크 높이 범위
            x1, x2 = j - center, j + center + 1        # 마스크 너비 범위
            if y1 < 0 or y2 > rows or x1 < 0 or x2 > cols :
                dst[i, j] = image[i, j]
            else:
                mask = image[y1:y2, x1:x2]                 # 범위 지정
                dst[i, j] = cv2.mean(mask)[0]
    return dst

image = cv2.imread("images/filter_avg.jpg", cv2.IMREAD_GRAYSCALE)
if image is None: raise Exception("영상파일 읽기 오류")

avg_img  = average_filter(image, 5)                    # 사용자 정의 평균값 필터 함수
blur_img = cv2.blur(image, (5, 5), borderType=cv2.BORDER_CONSTANT) # OpenCV의 블러링 함수
box_img  = cv2.boxFilter(image, ddepth=-1, ksize=(5, 5))   # OpenCV의 박스 필터 함수

cv2.imshow("image", image),
cv2.imshow("avg_img", avg_img)
cv2.imshow("blur_img", box_img)
cv2.imshow("box_img", box_img)
cv2.waitKey(0)