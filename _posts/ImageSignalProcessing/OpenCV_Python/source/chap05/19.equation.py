import numpy as np, cv2

data = [ 3, 0, 6, -3, 4, 2, -5,-1, 9]				    # 1차원 리스트 생성
m1 = np.array(data, np.float32).reshape(3,3)
m2 = np.array([36, 10, 28], np.float32)

ret, inv = cv2.invert(m1, cv2.DECOMP_LU)                # 역행렬 계산
if ret:
    dst1 = inv.dot(m2)                                  # numpy 제공 행렬곱 함수
    dst2 = cv2.gemm(inv, m2, 1, None, 1)                # OpenC 제공 행렬곱 함수
    ret, dst3 = cv2.solve(m1, m2, cv2.DECOMP_LU)        # 연립방정식 풀이

    print("[inv] = \n%s\n" % inv)
    print("[dst1] =", dst1.flatten())                   # 다행 1열 행렬을 한행에 표시
    print("[dst2] =", dst2.flatten())                   # 행렬을 벡터로 변환
    print("[dst3] =", dst3.flatten())                   # 행렬을 벡터로 변환
else:
    print("역행렬이 존재하지 않습니다.")


