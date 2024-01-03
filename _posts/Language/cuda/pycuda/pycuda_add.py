# pip install pycuda
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy

# CUDA 커널 코드 작성
cuda_code = """
__global__ void add_kernel(float *a, float *b, float *c, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}
"""

# 커널 코드 컴파일
mod = SourceModule(cuda_code)

# 커널 함수 가져오기
add_kernel = mod.get_function("add_kernel")

# 입력 데이터 생성
a = numpy.array([1.0, 2.0, 3.0], dtype=numpy.float32)
b = numpy.array([4.0, 5.0, 6.0], dtype=numpy.float32)
c = numpy.zeros_like(a)

# GPU에서 커널 실행
block_size = 256
grid_size = (a.size + block_size - 1) // block_size
add_kernel(cuda.In(a), cuda.In(b), cuda.Out(c), numpy.int32(a.size), block=(block_size, 1, 1), grid=(grid_size, 1))

# 결과 출력
print("Result:", c)