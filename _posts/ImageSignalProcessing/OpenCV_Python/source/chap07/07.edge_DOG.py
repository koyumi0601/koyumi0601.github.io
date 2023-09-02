import numpy as np, cv2

image = cv2.imread("images/dog.jpg", cv2.IMREAD_GRAYSCALE)
if image is None: raise Exception("영상파일 읽기 오류")

gaus = cv2.GaussianBlur(image, (9,9), 0, 0)            # 가우시안 마스크 적용
dst1 = cv2.Laplacian(gaus, cv2.CV_16S, 9)             # 라플라시안 수행

gaus1 = cv2.GaussianBlur(image, (3, 3), 0)          # 가우사안 블러링
gaus2 = cv2.GaussianBlur(image, (9, 9), 0)
dst2 = gaus1 - gaus2          # DoG 수행

cv2.imshow("image", image)
cv2.imshow("dst1 - LoG", dst1.astype("uint8"))
cv2.imshow("dst2 - DoG", dst2)
cv2.waitKey(0)
