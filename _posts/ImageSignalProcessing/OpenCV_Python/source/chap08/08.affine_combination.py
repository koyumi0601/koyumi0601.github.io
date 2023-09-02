import numpy as np, math, cv2
from Common.interpolation import affine_transform

def getAffineMat(center, degree, fx = 1, fy = 1, translate = (0,0)):
    cen_trans = np.eye(3, dtype=np.float32)
    org_trans = np.eye(3, dtype=np.float32)
    scale_mat = np.eye(3, dtype=np.float32)         # 크기 변경 행렬
    trans_mat = np.eye(3, dtype=np.float32)         # 평행 이동 행렬
    rot_mat   = np.eye(3, dtype=np.float32)         # 회전 변환 행렬

    radian = (degree/180.0) * np.pi                 # 회전 각도 - 라디언  계산
    rot_mat[0] = [ np.cos(radian), np.sin(radian), 0]
    rot_mat[1] = [-np.sin(radian), np.cos(radian), 0]

    cen_trans[:2, 2] = center                       # 중심 좌표를 기준으로 회전
    org_trans[:2, 2] = np.multiply(center[0], -1)   # 원점으로 이동
    trans_mat[:2, 2] = translate                    # 평행 이동 행렬의 원소 지정
    scale_mat[0, 0], scale_mat[1, 1] = fx, fy       # 크기 변경 행렬의 원소 지정

    ret_mat = cen_trans.dot(rot_mat.dot(trans_mat.dot(scale_mat.dot(org_trans))))
    # ret_mat = cen_trans.dot(rot_mat.dot(scale_mat.dot(trans_mat.dot(org_trans))))
    return np.delete(ret_mat, 2, axis=0)            # 행 제거 ret_mat[0:2,:]

image = cv2.imread('images/affine2.jpg', cv2.IMREAD_GRAYSCALE)
if image is None: raise Exception("영상 파일 읽기 에러")

size = image.shape[::-1]
center = np.divmod(size, 2)[0]                  # 회전 중심 좌표 - 크기는 행,열의 역순
angle, tr = 45.0, (200, 0)                                      # 각도와 평행이동

aff_mat1 = getAffineMat(center, angle)                          # 중심 좌표 기준 회전
aff_mat2 = getAffineMat((0,0), 0, 2.0, 1.5)                     # 크기 변경 - 확대
aff_mat3 = getAffineMat(center, angle, 0.7, 0.7)                # 회전 및 축소
aff_mat4 = getAffineMat(center, angle, 0.7, 0.7, tr)            # 복합 변환

dst1 = cv2.warpAffine(image, aff_mat1, size)             # OpenCV 함수
dst2 = cv2.warpAffine(image, aff_mat2, size)
dst3 = affine_transform(image, aff_mat3)                        # 사용자 정의 함수
dst4 = affine_transform(image, aff_mat4)

cv2.imshow("image", image)
cv2.imshow("dst1_only_rotate", dst1)
cv2.imshow("dst2_only_scaling", dst2)
cv2.imshow("dst3_rotate_scaling", dst3)
cv2.imshow("dst4_rotate_scaling_translate", dst4)
cv2.waitKey(0)