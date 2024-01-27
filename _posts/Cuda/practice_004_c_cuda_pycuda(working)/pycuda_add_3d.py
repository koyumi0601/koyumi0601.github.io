# pip install pycuda
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy


# CUDA kernel code
cuda_code1 = """
__global__ void add_kernel(float *a, float *b, float *c, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}
"""

cuda_code2 = """
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
"""

# kernel code compile
mod1 = SourceModule(cuda_code1)
mode2 = SourceModule(cuda_code2)
# warning C4819: multi-character. encoding utf-8 with BOM, with signature using visual studio save dropdown menu

# get kernel function
add_kernel = mod1.get_function("add_kernel")
add_const_kernel = mode2.get_function("add_const_kernel")

# input data
a = numpy.array([1.0, 2.0, 3.0], dtype=numpy.float32)
b = numpy.array([4.0, 5.0, 6.0], dtype=numpy.float32)
c = numpy.zeros_like(a)

pnsz = numpy.int32(numpy.array([100, 40, 30]))
mul = numpy.float32(numpy.random.randn())
add = numpy.float32(numpy.random.randn())
src = numpy.float32(numpy.random.randn(pnsz[0], pnsz[1], pnsz[2]))
dst = numpy.float32(numpy.zeros_like(src))

# GPU - execute kernel, 1D
block_size = 256
grid_size = (a.size + block_size - 1) // block_size
add_kernel(cuda.In(a), cuda.In(b), cuda.Out(c), numpy.int32(a.size), block=(block_size, 1, 1), grid=(grid_size, 1))
print("Result:", c)

# GPU - execute kernel, 3D
block_size = (32, 8, 4) # less than warp size 32 * streaming multiprocessor 8 * scheduler 4 for GTX960 
grid_size = (int(pnsz[0] // block_size[0] + 1), int(pnsz[1] // block_size[1] + 1), int(pnsz[2] // block_size[2] + 1))
add_const_kernel(cuda.Out(dst), cuda.In(src), numpy.float32(add), cuda.In(pnsz), block=block_size, grid=grid_size)

print("Result: ", dst[:,:,:])