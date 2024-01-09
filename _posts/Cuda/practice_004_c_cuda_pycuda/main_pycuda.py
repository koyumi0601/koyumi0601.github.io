import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy
import copy
import time

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


__global__ void add_const_kernel(float *pddst, float *pdsrc, float dconst, int *pnsz) {
    int nx = pnsz[0];
    int ny = pnsz[1];
    int nz = pnsz[2];
    int id = 0;

    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;
    int idz = blockDim.z * blockIdx.z + threadIdx.z;

    if (idz >= nz || idy >= ny || idx >= nx) return;

    id = ny * nx * idz + nx * idy + idx;
    pddst[id] = pdsrc[id] + dconst;
    
    return ;

}


__global__ void mul_mat_kernel(float *pddst, float *pdsrc, float *pdsrc2, int *pnsz) {
    int nx = pnsz[0];
    int ny = pnsz[1];
    int nz = pnsz[2];
    int id = 0;

    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;
    int idz = blockDim.z * blockIdx.z + threadIdx.z;

    if (idz >= nz || idy >= ny || idx >= nx) return;

    id = ny * nx * idz + nx * idy + idx;
    pddst[id] = pdsrc[id] * pdsrc2[id];

    return ;

}


__global__ void add_mat_kernel(float *pddst, float *pdsrc, float *pdsrc2, int *pnsz) {
    int nx = pnsz[0];
    int ny = pnsz[1];
    int nz = pnsz[2];
    int id = 0;

    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;
    int idz = blockDim.z * blockIdx.z + threadIdx.z;

    if (idz >= nz || idy >= ny || idx >= nx) return;

    id = ny * nx * idz + nx * idy + idx;
    pddst[id] = pdsrc[id] + pdsrc2[id];
    
    return ;

}
"""

# 커널 코드를 컴파일
module = SourceModule(kernel_code)
mul_const_kernel = module.get_function("mul_const_kernel")
add_const_kernel = module.get_function("add_const_kernel")
mul_mat_kernel = module.get_function("mul_mat_kernel")
add_mat_kernel = module.get_function("add_mat_kernel")

pnsz = numpy.int32(numpy.array([300, 1024, 760]))
mul = numpy.float32(numpy.random.randn())
# mul = numpy.float32(3.0)
add = numpy.float32(numpy.random.randn())
# add = numpy.float32(5.0)
# src1 = numpy.float32(numpy.ones((pnsz[0], pnsz[1], pnsz[2]))*4)
src1 = numpy.float32(numpy.random.randn(pnsz[0], pnsz[1], pnsz[2]))
# src2 = numpy.float32(numpy.ones((pnsz[0], pnsz[1], pnsz[2]))*2)
src2 = numpy.float32(numpy.random.randn(pnsz[0], pnsz[1], pnsz[2]))
dst = numpy.float32(numpy.zeros_like(src1))


## numpy in CPU
src_numpy = copy.deepcopy(src1)
pre_time_numpy = time.time()
dst_numpy = src_numpy
dst_numpy = dst_numpy * mul
dst_numpy = dst_numpy + add
dst_numpy = dst_numpy * dst_numpy
dst_numpy = dst_numpy + dst_numpy
after_time_numpy = time.time()
print(f'elapsed time numpy: {after_time_numpy - pre_time_numpy}')
print("Result: ", dst_numpy[:4,0,0])
## cuda 


block_size = (32, 8, 4) 
# GTX 960: warp size 32 * streaming multiprocessor 8 * scheduler 4 = 1024
# RTX 3060: warp size 32 * streaming multiprocessor 28 * scheduler 4 = 3584 (cuda core)
grid_size = (int(pnsz[0] // block_size[0] + 1), int(pnsz[1] // block_size[1] + 1), int(pnsz[2] // block_size[2] + 1))

pre_time_pycuda = time.time()

mul_const_kernel(cuda.Out(dst), cuda.In(src1), numpy.float32(mul), cuda.In(pnsz), block=block_size, grid=grid_size)
add_const_kernel(cuda.Out(dst), cuda.In(dst), numpy.float32(add), cuda.In(pnsz), block=block_size, grid=grid_size)
mul_mat_kernel(cuda.Out(dst), cuda.In(dst), cuda.In(dst), cuda.In(pnsz), block=block_size, grid=grid_size)
add_mat_kernel(cuda.Out(dst), cuda.In(dst), cuda.In(dst), cuda.In(pnsz), block=block_size, grid=grid_size)

after_time_pycuda = time.time()
print(f'elapsed time pycuda: {after_time_pycuda - pre_time_pycuda}')

print("Result: ", dst[:4,0,0])

# https://cuda.readthedocs.io/ko/latest/PyCUDA_int/
# https://www.cudahandbook.com/sample-chapters/
# https://www.youtube.com/watch?v=X9mflbX1NL8 Boston University - PyOpenCL, PyCUDA GPU programming