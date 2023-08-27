import numpy as np, cv2

m1 = np.full((3, 6), 10, np.uint8)			# 단일 채널 생성 및 초기화
m2 = np.full((3, 6), 50, np.uint8)
m_mask = np.zeros(m1.shape, np.uint8)		# 마스크 생성
m_mask[ :, 3: ] = 1							# 관심 영역을 지정한 후, 1을 할당

m_add1 = cv2.add(m1, m2)                    # 행렬 덧셈
m_add2 = cv2.add(m1, m2, mask=m_mask)       # 관심 영역만 덧셈 수행

# 행렬 나눗셈 수행
m_div1 = cv2.divide(m1, m2)
m1 = m1.astype(np.float32)                  # 형변환 - 소수 부분 보존
m2 = np.float32(m2)
m_div2 = cv2.divide(m1, m2)

titles = ['m1', 'm2', 'm_mask','m_add1','m_add2','m_div1', 'm_div2']
for title in titles:
    print("[%s] = \n%s \n" % (title, eval(title)))