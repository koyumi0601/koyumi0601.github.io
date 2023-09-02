import numpy as np, cv2
# numpy array 생성 예시
v1 = np.array([1, 2, 3], np.float32)          # 1차원 리스트로 생성- 행벡터
v2 = np.array([[1], [2], [3]], np.float32)      # 2차원 리스트로(3행, 1열) - 행벡터
v3 = np.array([[1, 2, 3]], np.float32)        	# 2차원 리스트로(1행, 3열) - 일반 행렬

# OpenCV 산술 연산 함수는 numpy array만 가능함
v1_exp = cv2.exp(v1)                             # 벡터에 대한
v2_exp = cv2.exp(v2)                             # 행렬에 대한 지수 계산
v3_exp = cv2.exp(v3)                             # 행렬에 대한 지수 계산
log = cv2.log(v1)                             # 로그 계산
sqrt= cv2.sqrt(v2)                            # 제곱근 계산
pow = cv2.pow(v3, 3)                          # 3의 거듭제곱 계산

# 결과 출력
print("[v1] 형태: %s 원소: %s" % ( v1.shape, v1))
print("[v2] 형태: %s 원소:\n%s" % ( v2.shape, v2))
print("[v3] 형태: %s 원소: %s" % ( v3.shape, v3))
print()

# 행렬 정보 출력 - OpenCV 결과는 행렬로 반환됨 - 행벡터는 열벡터로 반환됨
print("[v1_exp] 자료형: %s 형태: %s" % ( type(v1_exp), v1_exp.shape))  # 행벡터 인수의 결과
print("[v2_exp] 자료형: %s 형태: %s" % ( type(v2_exp), v2_exp.shape))  # 행벡터 인수의 결과
print("[v3_exp] 자료형: %s 형태: %s" % ( type(v3_exp), v3_exp.shape))  # 행벡터 인수의 결과
print()

# 열벡터를 1 행에 출력하는 예시 - 행벡터로 변환
print("[log] =", log.T)
print("[sqrt] =", np.ravel(sqrt))         
print("[pow] =", pow.flatten())
