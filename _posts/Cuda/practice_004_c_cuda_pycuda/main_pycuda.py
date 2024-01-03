import os
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
from pycuda import gpuarray


# 주어진 커널 코드
kernel_code = """
__global__ void mul_const_kernel(float *pddst, float *pdsrc, float dconst, int *pnsz) {
    int nx = pnsz[0];
    int ny = pnsz[1];
    int nz = pnsz[2];
    int id = 0;

    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;
    int idz = blockDim.z * blockIdx.z + threadIdx.z;

    if (idz >= nz || idy >= ny || idx >= nx) return;

    id = ny * nx * idz + nx * idy + idx;
    pddst[id] = pdsrc[id] * dconst;

    
    return ;
}
"""

# 커널 코드를 컴파일
module = SourceModule(kernel_code)
mul_const_kernel = module.get_function("mul_const_kernel")

# 데이터 및 메모리 할당
# pnsz = np.asarray([100, 50, 30], dtype=np.int32)
# pnsz = np.int32([100, 50, 30])
# pnsz = np.array([100, 50, 30], dtype=np.int32)
# pnsz = np.asarray([100, 50, 30], dtype=np.int32)
pnsz = np.array([100, 50, 30], dtype=np.int32)
# mul = np.random.randn()
# mul = np.random.randn().astype(np.float32)
mul = float(np.random.randn())
# src = np.random.randn(pnsz[0], pnsz[1], pnsz[2]).astype(dtype=np.float32)
src = np.random.randn(pnsz[0], pnsz[1], pnsz[2]).astype(dtype=np.float32)
# CPU 메모리 할당 및 데이터 복사
src_cpu = src.copy()
dst_cpu = np.zeros_like(src_cpu)

# GPU 메모리 할당 및 데이터 복사
src_gpu = cuda.mem_alloc(src_cpu.nbytes)
dst_gpu = cuda.mem_alloc(dst_cpu.nbytes)
cuda.memcpy_htod(src_gpu, src_cpu)


# 커널 설정
cuda.init()
device = cuda.Device(0) # first GPU
warp_size = device.warp_size
sm_count = device.get_attribute(cuda.device_attribute.MULTIPROCESSOR_COUNT)
print("Warp 크기:", warp_size)
print("SM 수:", sm_count)
# 스레드수 = 병렬연산 수
block_size = (8, 8, 8) 
# 2의 제곱수. 128, 256, 512, 1024...
# 블록당 포함되는 스레드 수
# 1차원 데이터 (x, 1, 1) 2차원 데이터 (x, y, 1), 3차원 데이터 (x, y, z)
grid_size = (pnsz[0] // block_size[0] + 1, pnsz[1] // block_size[1] + 1, pnsz[2] // block_size[2] + 1)
# 그리드당 포함되는 블록 수
print(block_size, grid_size)

# CUDA 커널 실행

# mul_const_kernel(dst_gpu, src_gpu, mul, pnsz, block=block_size, grid=grid_size) # block당 스레드 수, 그리드(GPU)당 스레드 수
# mul_const_kernel(dst_gpu, src_gpu, np.float32(mul), cuda.InOut(pnsz), block=block_size, grid=grid_size)
# mul_const_kernel(dst_gpu, src_gpu, np.float32(mul), pnsz.astype(np.int32), block=block_size, grid=grid_size)
# mul_const_kernel(dst_gpu, src_gpu, np.float32(mul), pnsz, block=block_size, grid=grid_size)
# mul_const_kernel(dst_gpu, src_gpu, np.float32(mul), cuda.InOut(pnsz), block=block_size, grid=grid_size)
# mul_const_kernel(dst_gpu, src_gpu, np.float32(mul), pnsz, block=block_size, grid=grid_size)
# mul_const_kernel(dst_gpu, src_gpu, np.float32(mul), pnsz.astype(np.int32), block=block_size, grid=grid_size)
mul_const_kernel(dst_gpu, src_gpu, np.float32(mul), pnsz.astype(np.int32), block=block_size, grid=grid_size)
cuda.memcpy_dtoh(dst_cpu, dst_gpu)

# 결과 확인
print("GPU Result:")
print(dst_cpu)

# https://cuda.readthedocs.io/ko/latest/PyCUDA_int/
# https://www.cudahandbook.com/sample-chapters/
# https://www.youtube.com/watch?v=X9mflbX1NL8 Boston University - PyOpenCL, PyCUDA GPU programming