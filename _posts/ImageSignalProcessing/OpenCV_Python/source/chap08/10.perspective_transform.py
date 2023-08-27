import numpy as np, cv2

image = cv2.imread('images/perspective.jpg', cv2.IMREAD_COLOR)
if image is None: raise Exception("영상 파일을 읽기 에러")

pts1 = np.float32([(80, 40),  (315, 133), (75, 300), (335, 300)] )
pts2 = np.float32([(50, 60),  (340, 60), (50, 320), (340, 320)])

perspect_mat = cv2.getPerspectiveTransform(pts1, pts2)
dst = cv2.warpPerspective(image, perspect_mat, image.shape[1::-1], cv2.INTER_CUBIC)
print("[perspect_mat] = \n%s\n" % perspect_mat )

## 변환 좌표 계산 - 행렬 내적 이용 방법
ones = np.ones((4,1), np.float64)
pts3 = np.append(pts1, ones, axis=1)              # 원본 좌표 -> 동차 좌표 저장
pts4 = cv2.gemm(pts3, perspect_mat.T, 1, None, 1)  # 행렬 곱으로 좌표 변환값 계산

## 변환 좌표 계산 - cv2.transform() 함수 이용방법
# pts3 = np.expand_dims(pts1, axis=0)             # 차원 증가
# pts4 = cv2.transform(pts3, perspect_mat)
# pts4 = np.squeeze(pts4, axis=0)                 # 차원 감소
# pts3 = np.squeeze(pts3, axis=0)                 # 출력 위해

print(" 원본 영상 좌표 \t 목적 영상 좌표 \t\t 동차 좌표 \t\t 변환 결과 좌표")
for i in range(len(pts4)):
    pts4[i] /= pts4[i][2]
    print("%i : %-14s %-14s %-18s%-18s" % (i, pts1[i], pts2[i], pts3[i], pts4[i]))
    cv2.circle(image, tuple(pts1[i].astype(int)), 4, (0, 255, 0), -1) # 원본 영상에 pts1 표시
    cv2.circle(dst  , tuple(pts2[i].astype(int)), 4, (0, 255, 0), -1) # 목적 영상에 pts2 표시

cv2.imshow("image", image)
cv2.imshow("dst_perspective", dst)
cv2.waitKey(0)