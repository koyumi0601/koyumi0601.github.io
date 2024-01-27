import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np

# CUDA 코드에서 사용할 데이터 생성
n = 100
a = np.random.randint(0, 10, n).astype(np.int32)
b = np.random.randint(0, 10, n).astype(np.int32)
c = np.zeros_like(a)

# CUDA 커널 실행
from pycuda.compiler import SourceModule
mod = SourceModule(open("cuda_code.cu").read())
add_vectors = mod.get_function("addVectors")
add_vectors(cuda.In(a), cuda.In(b), cuda.Out(c), np.int32(n), block=(128, 1, 1), grid=(n//128+1, 1))

# 결과 출력
print("a:", a)
print("b:", b)
print("c:", c)