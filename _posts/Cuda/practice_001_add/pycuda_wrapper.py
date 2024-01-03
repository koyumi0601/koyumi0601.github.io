# add_wrapper.py
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
from pycuda.compiler import SourceModule
 
# CUDA kernel source code
source = """
__global__ void addKernel(float *a, float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}
"""

# Compile CUDA kernel
mod = SourceModule(source)

# Get the kernel function
add_kernel = mod.get_function("addKernel")

# Array size
# n = 256

# # Create input arrays and allocate device memory
# a = np.random.randn(n).astype(np.float32)
# b = np.random.randn(n).astype(np.float32)
# c = np.empty_like(a)

# # Copy input arrays to device
# a_gpu = cuda.mem_alloc(a.nbytes)
# b_gpu = cuda.mem_alloc(b.nbytes)
# c_gpu = cuda.mem_alloc(c.nbytes)
# cuda.memcpy_htod(a_gpu, a)
# cuda.memcpy_htod(b_gpu, b)

# # Define block and grid dimensions
# block_size = 256
# grid_size = (n + block_size - 1) // block_size

# # Launch the CUDA kernel
# add_kernel(a_gpu, b_gpu, c_gpu, np.int32(n), block=(block_size, 1, 1), grid=(grid_size, 1))

# # Copy the result back to the host
# cuda.memcpy_dtoh(c, c_gpu)

# # Print the result
# print("Result:")
# print(c)