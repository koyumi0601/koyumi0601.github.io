import numpy as np, cv2

x = np.array([1, 2, 3, 5, 10], np.float32)            # 리스트를 numpy array로 변환
y = np.array([2, 5, 7, 2, 9]).astype("float32")

mag = cv2.magnitude(x, y)                   # 크기 계산
ang = cv2.phase(x, y)                       # 각도(방향) 계산
p_mag, p_ang  = cv2.cartToPolar(x, y)  # 극좌표로 변환
x2, y2 = cv2.polarToCart(p_mag, p_ang)  # 직교좌표로 변한

print("[x] 형태: %s 원소: %s" % ( x.shape, x))
print("[mag] 형태: %s 원소: %s" % ( mag.shape, mag))

print(">>>열벡터를 1행에 출력하는 방법")
print("[m_mag] = %s" % mag.T)
print("[p_mag] = %s" % np.ravel(p_mag))
print("[p_ang] = %s" % np.ravel(p_ang))
print("[x_mat2] = %s" % x2.flatten())
print("[y_mat2] = %s" % y2.flatten())