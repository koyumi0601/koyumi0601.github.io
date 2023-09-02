import numpy as np, cv2

image1 = np.zeros((50, 512), np.uint8)         # 50 x 512 영상 생성
image2 = np.zeros((50, 512), np.uint8)

rows, cols = image1.shape[:2]

for i in range(rows):                           # 행렬 전체 조회
    for j in range(cols):
        image1.itemset((i, j), j//2)          # 화소값 점진적 증가
        image2.itemset((i, j), j // 20*10)    # 계단 현상 증가

cv2.imshow("image1", image1)
cv2.imshow("image2", image2)
cv2.waitKey(0)