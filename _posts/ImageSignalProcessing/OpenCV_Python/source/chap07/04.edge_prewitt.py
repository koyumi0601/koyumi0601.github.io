import numpy as np, cv2
from  Common.filters import filter

def differential(image, data1, data2):
    mask1 = np.array(data1, np.float32).reshape(3, 3)
    mask2 = np.array(data2, np.float32).reshape(3, 3)

    dst1 = filter(image, mask1)                     # 사용자 정의 회선 함수
    dst2 = filter(image, mask2)
    dst = cv2.magnitude(dst1, dst2)                 # 회선 결과 두 행렬의 크기 계산

    dst = cv2.convertScaleAbs(dst)                      # 윈도우 표시 위해 OpenCV 함수로 형변환 및 saturation 수행
    dst1 = cv2.convertScaleAbs(dst1)
    dst2 = cv2.convertScaleAbs(dst2)
    return dst, dst1, dst2

image = cv2.imread("images/edge.jpg", cv2.IMREAD_GRAYSCALE)
if image is None: raise Exception("영상파일 읽기 오류")

data1 = [-1, 0, 1,                         # 프리윗 수직 마스크
         -1, 0, 1,
         -1, 0, 1]
data2 = [-1,-1,-1,                         # 프리윗 수평 마스크
          0, 0, 0,
          1, 1, 1]
dst, dst1, dst2 = differential(image, data1, data2)

cv2.imshow("image", image)
cv2.imshow("prewitt edge", dst)
cv2.imshow("dst1 - vertical mask", dst1)
cv2.imshow("dst2 - horizontal mask", dst2)
cv2.waitKey(0)