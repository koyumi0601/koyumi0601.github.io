import numpy as np, cv2, math
import scipy.fftpack as sf

def cos(n,k,N):
    return math.cos((n + 1/2) * math.pi * k / N)

def C(k, N):
    return math.sqrt(1/N) if k==0 else math.sqrt(2/N)

def dct(g):
    N = len(g)
    f = [C(k, N) * sum(g[n] * cos(n, k, N ) for n in range(N)) for k in range(N)]
    return np.array(f, np.float32)

def idct(f):
    N = len(f)
    g = [sum(C(k, N) * f[k] * cos(n, k, N) for k in range(N)) for n in range(N)]
    return np.array(g)

def dct2(image):
    tmp = [dct(row) for row in image]
    dst = [dct(row) for row in np.transpose(tmp)]
    return np.transpose(dst)                   # 전치 환원 후 반환

def idct2(image):
    tmp = [idct(row) for row in image]
    dst = [idct(row) for row in np.transpose(tmp)]
    return np.transpose(dst)                   # 전치 환원 후 반환

def scipy_dct2(a):
    tmp = sf.dct(a, axis=0, norm='ortho' )
    return sf.dct(tmp, axis=1, norm='ortho' )

def scipy_idct2(a):
    tmp = sf.idct(a, axis=0, norm='ortho')
    return sf.idct(tmp, axis=1 , norm='ortho')

block = np.zeros((8,8), np.uint8)
cv2.randn(block, 128, 50)

dct1 = dct2(block)
dct2 = scipy_dct2(block)
dct3 = sf.dctn(block, shape=block.shape, norm='ortho')			# 2차원 dct 수행 다른 방식
dct4 = cv2.dct(block.astype("float32"))

idct1 = idct2(dct1)
idct2 = scipy_idct2(dct2)
idct3 = sf.idctn(dct3, shape=dct2.shape, norm='ortho')			# 2차원 dct 수행 다른 방식
idct4 = cv2.dct(dct4, flags=cv2.DCT_INVERSE)

print('block=\n', block)
print('dct1(저자구현 함수)=\n', dct1)
print('dct2(scipy 모듈 함수1)=\n', dct2)
print('dct3(scipy 모듈 함수2)=\n', dct3)
print('dct4(OpenCV 함수)=\n', dct4)
print()
print('idct1(저자구현 함수)=\n', cv2.convertScaleAbs(idct1))
print('idct2(scipy 모듈 함수1)=\n', cv2.convertScaleAbs(idct2))
print('idct3(scipy 모듈 함수2)=\n', cv2.convertScaleAbs(idct3))
print('idct4(OpenCV 함수)=\n', cv2.convertScaleAbs(idct4))