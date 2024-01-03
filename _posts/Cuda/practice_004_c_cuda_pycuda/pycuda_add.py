# pip install pycuda
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
# import pycuda.compiler as compiler
import numpy


# import pycuda.driver as cuda
# import pycuda.compiler as compiler
# cuda.init()
# driver_version = cuda.get_driver_version()
# cuda_version = cuda.get_version()
# print("CUDA driver version:", driver_version)
# print("Current CUDA Toolkit Version:", cuda_version)
# Name: pycuda
# Version: 2023.1
# Summary: Python wrapper for Nvidia CUDA
# Home-page: http://mathema.tician.de/software/pycuda
# Author: Andreas Kloeckner
# Author-email: inform@tiker.net
# License: MIT
# Location: D:\Program Files\Python311\Lib\site-packages
# Requires: appdirs, mako, pytools
# Required-by:



# CUDA kernel code
cuda_code = """
__global__ void add_kernel(float *a, float *b, float *c, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}
"""

# kernel code compile
mod = SourceModule(cuda_code)
# warning C4819: multi-character. encoding utf-8 with BOM, with signature using visual studio save dropdown menu

# # get kernel function
# add_kernel = mod.get_function("add_kernel")

# # input data
# a = numpy.array([1.0, 2.0, 3.0], dtype=numpy.float32)
# b = numpy.array([4.0, 5.0, 6.0], dtype=numpy.float32)
# c = numpy.zeros_like(a)

# # GPU - execute kernel
# block_size = 256
# grid_size = (a.size + block_size - 1) // block_size
# add_kernel(cuda.In(a), cuda.In(b), cuda.Out(c), numpy.int32(a.size), block=(block_size, 1, 1), grid=(grid_size, 1))

# # print result
# print("Result:", c)