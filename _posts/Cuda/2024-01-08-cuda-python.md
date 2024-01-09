---
layout: single
title: "python, c, cuda, pycuda and embedded python"
categories: cuda
tags: [language, programming, cpp, cuda, python, pycuda, embedded python]
toc: true
author_profile: false
sidebar:
    nav: "docs"
search: true

---

*python, cpp, cuda, embedded python conversion sample code*

- see practice_004_c_cuda_pycuda
- performance numpy (internally c vectorization) > cuda > cpp (for)
- need add embedded python
- need add npp

# python
```python
# add constant, multipy constant, add matrix, multipy matrix
# numpy
import os, copy
import numpy as np

pnsz = np.asarray([300, 1024, 760], dtype = np.int32)
mul = np.random.randn()
add = np.random.randn()
src = np.random.randn(pnsz[0], pnsz[1], pnsz[2]).astype(dtype=np.float32)

## numpy in CPU
src_numpy = copy.deepcopy(src)
dst_numpy = src_numpy
dst_numpy = dst_numpy * mul
dst_numpy = dst_numpy + add
dst_numpy = dst_numpy * dst_numpy
dst_numpy = dst_numpy + dst_numpy
```

# cpp
- math_clang.c

```c
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

/*
 * Multiplies each element of the source array by a constant value.
 * 
 * @param pddst     Destination array to store the result.
 * @param pdsrc     Source array to be multiplied.
 * @param dconst    Constant value to multiply each element.
 * @param pnsz      Integer array containing the dimensions [nx, ny, nz].
 * 
 * p represents pointer
 * d represents data
 */
void mul_const(float *pddst, float *pdsrc, float dconst, int *pnsz) {
    int nx = pnsz[0];
    int ny = pnsz[1];
    int nz = pnsz[2];
    int id = 0;

    for (int idz = 0; idz < nz; idz++) {
        for (int idy = 0; idy < ny; idy++) {
            for (int idx = 0; idx < nx; idx++) {
                id = ny * nx * idz + nx * idy + idx;
                pddst[id] = pdsrc[id] * dconst;
            }
        }
    }
    
    return ;
}

/*
 * Adds a constant value to each element of the source array.
 * 
 * @param pddst     Destination array to store the result.
 * @param pdsrc     Source array to which the constant is added.
 * @param dconst    Constant value to be added to each element.
 * @param pnsz      Integer array containing the dimensions [nx, ny, nz].
 */
void add_const(float *pddst, float *pdsrc, float dconst, int *pnsz) {
    int nx = pnsz[0];
    int ny = pnsz[1];
    int nz = pnsz[2];
    int id = 0;

    for (int idz = 0; idz < nz; idz++) {
        for (int idy = 0; idy < ny; idy++) {
            for (int idx = 0; idx < nx; idx++) {
                id = ny * nx * idz + nx * idy + idx;
                pddst[id] = pdsrc[id] + dconst;
            }
        }
    }
    
    return ;

}

/*
 * Multiplies two arrays element-wise and stores the result in the destination array.
 * 
 * @param pddst     Destination array to store the result.
 * @param pdsrc     Source array to be multiplied with another source array.
 * @param pdsrc2    Second source array for element-wise multiplication.
 * @param pnsz      Integer array containing the dimensions [nx, ny, nz].
 */
void mul_mat(float *pddst, float *pdsrc, float *pdsrc2, int *pnsz) {
    int nx = pnsz[0];
    int ny = pnsz[1];
    int nz = pnsz[2];
    int id = 0;

    for (int idz = 0; idz < nz; idz++) {
        for (int idy = 0; idy < ny; idy++) {
            for (int idx = 0; idx < nx; idx++) {
                id = ny * nx * idz + nx * idy + idx;
                pddst[id] = pdsrc[id] * pdsrc2[id];
            }
        }
    }
    
    return ;

}

/*
 * Adds two arrays element-wise and stores the result in the destination array.
 * 
 * @param pddst     Destination array to store the result.
 * @param pdsrc     Source array to which the second source array is added.
 * @param pdsrc2    Second source array for element-wise addition.
 * @param pnsz      Integer array containing the dimensions [nx, ny, nz].
 */
void add_mat(float *pddst, float *pdsrc, float *pdsrc2, int *pnsz) {
    int nx = pnsz[0];
    int ny = pnsz[1];
    int nz = pnsz[2];
    int id = 0;

    for (int idz = 0; idz < nz; idz++) {
        for (int idy = 0; idy < ny; idy++) {
            for (int idx = 0; idx < nx; idx++) {
                id = ny * nx * idz + nx * idy + idx;
                pddst[id] = pdsrc[id] + pdsrc2[id];
            }
        }
    }
    
    return ;

}


```
- compile math_clang.c -> math_clang.o -> math_clang.so
- gcc 설치 필요 (msys2 or other install managers)

```bash
gcc -c -fPIC math_clang.c -o math_clang.o
gcc -shared math_clang.o -o libmath_clang.so
```

- .py

```python
import os, copy
import ctypes
from ctypes import *
import sys
import time

if sys.platform.startswith('win'): 
    print('platform is window')
    platform = 'window'
elif sys.platform.startswith('linux'): 
    print('platform is linux')
    platform = 'linux'
else: 
    print('platform is others')
    platform = 'other platform'

pnsz = np.asarray([300, 1024, 760], dtype = np.int32)
mul = np.random.randn()
add = np.random.randn()
src = np.random.randn(pnsz[0], pnsz[1], pnsz[2]).astype(dtype=np.float32)

## Clang in CPU
if platform == 'window':
    clang_file = os.path.join(os.path.dirname(__file__), 'libmath_clang.so')
elif platform == 'linux':
    clang_file = os.path.join(os.getcwd(), 'libmath_clang.so')

_math_clang = ctypes.CDLL(clang_file)

__mul_const_clang = _math_clang.mul_const
__add_const_clang = _math_clang.add_const
__mul_mat_clang = _math_clang.mul_mat
__add_mat_clang = _math_clang.add_mat

## 초기화, 타입 변환에 주의한다.
# init
__mul_const_clang.argtypes = (POINTER(c_float), POINTER(c_float), c_float, POINTER(c_int)) # argument
__mul_const_clang.restypes = c_void_p # return variable
__add_const_clang.argtypes = (POINTER(c_float), POINTER(c_float), c_float, POINTER(c_int)) # argument
__add_const_clang.restypes = c_void_p # return variable
__mul_mat_clang.argtypes = (POINTER(c_float), POINTER(c_float), POINTER(c_float), POINTER(c_int)) # argument
__mul_mat_clang.restypes = c_void_p # return variable
__add_mat_clang.argtypes = (POINTER(c_float), POINTER(c_float), POINTER(c_float), POINTER(c_int)) # argument
__add_mat_clang.restypes = c_void_p # return variable

c_float_p = lambda x: x.ctypes.data_as(POINTER(c_float))
c_int_p = lambda x: x.ctypes.data_as(POINTER(c_int))

##
src_clang = copy.deepcopy(src)
dst_clang = src_clang
__mul_const_clang(c_float_p(dst_clang), c_float_p(dst_clang), mul, c_int_p(pnsz))
__add_const_clang(c_float_p(dst_clang), c_float_p(dst_clang), add, c_int_p(pnsz))
__mul_mat_clang(c_float_p(dst_clang), c_float_p(dst_clang), c_float_p(dst_clang), c_int_p(pnsz))
__add_mat_clang(c_float_p(dst_clang), c_float_p(dst_clang), c_float_p(dst_clang), c_int_p(pnsz))

```



# cuda (.cu)

- .cu

```c
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <cuda.h>
// #define DLLEXPORT extern "C" __declspec(dllexport) // window


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

// cpu interface
// DLLEXPORT void mul_const(float *pddst, float *pdsrc, float dconst, int *pnsz){ // window only expression
extern "C" void mul_const(float *pddst, float *pdsrc, float dconst, int *pnsz){ // available on linux, window
    float *gpddst = 0;
    float *gpdsrc = 0;
    int *gpnsz = 0;

    cudaMalloc((void **)&gpddst, pnsz[0]*pnsz[1]*pnsz[2]*sizeof(float));
    cudaMalloc((void **)&gpdsrc, pnsz[0]*pnsz[1]*pnsz[2]*sizeof(float));
    cudaMalloc((void **)&gpnsz, 3*sizeof(int));

    cudaMemset(gpddst, 0, pnsz[0]*pnsz[1]*pnsz[2]*sizeof(float));
    cudaMemset(gpdsrc, 0, pnsz[0]*pnsz[1]*pnsz[2]*sizeof(float));
    cudaMemset(gpnsz, 0, 3*sizeof(float));

    cudaMemcpy(gpdsrc, pdsrc, pnsz[0]*pnsz[1]*pnsz[2]*sizeof(float), cudaMemcpyHostToDevice); // destination, source, memory size, direction
    cudaMemcpy(gpnsz, pnsz, 3*sizeof(int), cudaMemcpyHostToDevice); // destination, source, memory size, direction

    int nthread = 8;
    dim3 nblock(nthread, nthread, nthread);
    dim3 ngrid ((pnsz[0] + nthread - 1) / nthread,
                (pnsz[1] + nthread - 1) / nthread,
                (pnsz[2] + nthread - 1) / nthread);

    mul_const_kernel<<<ngrid, nblock>>>(gpddst, gpdsrc, dconst, gpnsz);

    cudaMemcpy(pddst, gpddst, pnsz[0]*pnsz[1]*pnsz[2]*sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(gpddst);
    cudaFree(gpdsrc);
    cudaFree(gpnsz);

    gpddst = 0;
    gpdsrc = 0;
    gpnsz = 0;

    return ;
}

// DLLEXPORT void add_const(float *pddst, float *pdsrc, float dconst, int *pnsz){ // window only
extern "C" void add_const(float *pddst, float *pdsrc, float dconst, int *pnsz){ // available on window and linux 
    float *gpddst = 0;
    float *gpdsrc = 0;
    int *gpnsz = 0;

    cudaMalloc((void **)&gpddst, pnsz[0]*pnsz[1]*pnsz[2]*sizeof(float));
    cudaMalloc((void **)&gpdsrc, pnsz[0]*pnsz[1]*pnsz[2]*sizeof(float));
    cudaMalloc((void **)&gpnsz, 3*sizeof(int));

    cudaMemset(gpddst, 0, pnsz[0]*pnsz[1]*pnsz[2]*sizeof(float));
    cudaMemset(gpdsrc, 0, pnsz[0]*pnsz[1]*pnsz[2]*sizeof(float));
    cudaMemset(gpnsz, 0, 3*sizeof(float));

    cudaMemcpy(gpdsrc, pdsrc, pnsz[0]*pnsz[1]*pnsz[2]*sizeof(float), cudaMemcpyHostToDevice); // destination, source, memory size, direction
    cudaMemcpy(gpnsz, pnsz, 3*sizeof(int), cudaMemcpyHostToDevice); // destination, source, memory size, direction

    int nthread = 8;
    dim3 nblock(nthread, nthread, nthread);
    dim3 ngrid ((pnsz[0] + nthread - 1) / nthread,
                (pnsz[1] + nthread - 1) / nthread,
                (pnsz[2] + nthread - 1) / nthread);

    add_const_kernel<<<ngrid, nblock>>>(gpddst, gpdsrc, dconst, gpnsz);

    cudaMemcpy(pddst, gpddst, pnsz[0]*pnsz[1]*pnsz[2]*sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(gpddst);
    cudaFree(gpdsrc);
    cudaFree(gpnsz);

    gpddst = 0;
    gpdsrc = 0;
    gpnsz = 0;

    return ;
}

// DLLEXPORT void mul_mat(float *pddst, float *pdsrc1, float *pdsrc2, int *pnsz){ // window only
extern "C" void mul_mat(float *pddst, float *pdsrc1, float *pdsrc2, int *pnsz){ // available on window and linux
    float *gpddst = 0;
    float *gpdsrc1 = 0;
    float *gpdsrc2 = 0;
    int *gpnsz = 0;

    cudaMalloc((void **)&gpddst, pnsz[0]*pnsz[1]*pnsz[2]*sizeof(float));
    cudaMalloc((void **)&gpdsrc1, pnsz[0]*pnsz[1]*pnsz[2]*sizeof(float));
    cudaMalloc((void **)&gpdsrc2, pnsz[0]*pnsz[1]*pnsz[2]*sizeof(float));
    cudaMalloc((void **)&gpnsz, 3*sizeof(int));


    cudaMemset(gpddst, 0, pnsz[0]*pnsz[1]*pnsz[2]*sizeof(float));
    cudaMemset(gpdsrc1, 0, pnsz[0]*pnsz[1]*pnsz[2]*sizeof(float));
    cudaMemset(gpdsrc2, 0, pnsz[0]*pnsz[1]*pnsz[2]*sizeof(float));
    cudaMemset(gpnsz, 0, 3*sizeof(int));

    cudaMemcpy(gpdsrc1, pdsrc1, pnsz[0]*pnsz[1]*pnsz[2]*sizeof(float), cudaMemcpyHostToDevice); // destination, source, memory size, direction
    cudaMemcpy(gpdsrc2, pdsrc2, pnsz[0]*pnsz[1]*pnsz[2]*sizeof(float), cudaMemcpyHostToDevice); // destination, source, memory size, direction
    cudaMemcpy(gpnsz, pnsz, 3*sizeof(int), cudaMemcpyHostToDevice); // destination, source, memory size, direction

    int nthread = 8;
    dim3 nblock(nthread, nthread, nthread);
    dim3 ngrid ((pnsz[0] + nthread - 1) / nthread,
                (pnsz[1] + nthread - 1) / nthread,
                (pnsz[2] + nthread - 1) / nthread);

    mul_mat_kernel<<<ngrid, nblock>>>(gpddst, gpdsrc1, gpdsrc2, gpnsz);

    cudaMemcpy(pddst, gpddst, pnsz[0]*pnsz[1]*pnsz[2]*sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(gpddst);
    cudaFree(gpdsrc1);
    cudaFree(gpdsrc2);
    cudaFree(gpnsz);

    gpddst = 0;
    gpdsrc1 = 0;
    gpdsrc2 = 0;
    gpnsz = 0;

    return ;
}


// DLLEXPORT void add_mat(float *pddst, float *pdsrc1, float *pdsrc2, int *pnsz){ // window only
extern "C" void add_mat(float *pddst, float *pdsrc1, float *pdsrc2, int *pnsz){ // available on window and linux
    float *gpddst = 0;
    float *gpdsrc1 = 0;
    float *gpdsrc2 = 0;
    int *gpnsz = 0;

    cudaMalloc((void **)&gpddst, pnsz[0]*pnsz[1]*pnsz[2]*sizeof(float));
    cudaMalloc((void **)&gpdsrc1, pnsz[0]*pnsz[1]*pnsz[2]*sizeof(float));
    cudaMalloc((void **)&gpdsrc2, pnsz[0]*pnsz[1]*pnsz[2]*sizeof(float));
    cudaMalloc((void **)&gpnsz, 3*sizeof(int));


    cudaMemset(gpddst, 0, pnsz[0]*pnsz[1]*pnsz[2]*sizeof(float));
    cudaMemset(gpdsrc1, 0, pnsz[0]*pnsz[1]*pnsz[2]*sizeof(float));
    cudaMemset(gpdsrc2, 0, pnsz[0]*pnsz[1]*pnsz[2]*sizeof(float));
    cudaMemset(gpnsz, 0, 3*sizeof(int));

    cudaMemcpy(gpdsrc1, pdsrc1, pnsz[0]*pnsz[1]*pnsz[2]*sizeof(float), cudaMemcpyHostToDevice); // destination, source, memory size, direction
    cudaMemcpy(gpdsrc2, pdsrc2, pnsz[0]*pnsz[1]*pnsz[2]*sizeof(float), cudaMemcpyHostToDevice); // destination, source, memory size, direction
    cudaMemcpy(gpnsz, pnsz, 3*sizeof(int), cudaMemcpyHostToDevice); // destination, source, memory size, direction

    int nthread = 8;
    dim3 nblock(nthread, nthread, nthread);
    dim3 ngrid ((pnsz[0] + nthread - 1) / nthread,
                (pnsz[1] + nthread - 1) / nthread,
                (pnsz[2] + nthread - 1) / nthread);

    add_mat_kernel<<<ngrid, nblock>>>(gpddst, gpdsrc1, gpdsrc2, gpnsz);

    cudaMemcpy(pddst, gpddst, pnsz[0]*pnsz[1]*pnsz[2]*sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(gpddst);
    cudaFree(gpdsrc1);
    cudaFree(gpdsrc2);
    cudaFree(gpnsz);

    gpddst = 0;
    gpdsrc1 = 0;
    gpdsrc2 = 0;
    gpnsz = 0;

    return ;
}
```

- compile math_cu_linux.cu -> libmath_cu.so

```bash
nvcc -Xcompiler -fPIC math_cu_linux.cu -shared -o libmath_cu.so
# nvcc error   : 'cudafe++' died with status 0xC0000005 (ACCESS_VIOLATION).. need check compiler
```

- .py

```python
## python, c, cu

import os, copy
import numpy as np
import ctypes
from ctypes import *
import sys
import time

if sys.platform.startswith('win'): 
    print('platform is window')
    platform = 'window'
elif sys.platform.startswith('linux'): 
    print('platform is linux')
    platform = 'linux'
else: 
    print('platform is others')
    platform = 'other platform'

pnsz = np.asarray([300, 1024, 760], dtype = np.int32)
mul = np.random.randn()
add = np.random.randn()
src = np.random.randn(pnsz[0], pnsz[1], pnsz[2]).astype(dtype=np.float32)

## CU in GPU

if platform == 'window':
    cu_file = os.path.join(os.path.dirname(__file__), 'libmath_cu.so')
elif platform == 'linux':
    cu_file = os.path.join(os.getcwd(), 'libmath_cu.so')

_math_cu = ctypes.CDLL(cu_file)

__mul_const_cu = _math_cu.mul_const
__add_const_cu = _math_cu.add_const
__mul_mat_cu = _math_cu.mul_mat
__add_mat_cu = _math_cu.add_mat

__mul_const_cu.argtypes = (POINTER(c_float), POINTER(c_float), c_float, POINTER(c_int))
__mul_const_cu.restypes = c_void_p
__add_const_cu.argtypes = (POINTER(c_float), POINTER(c_float), c_float, POINTER(c_int))
__add_const_cu.restypes = c_void_p
__mul_mat_cu.argtypes = (POINTER(c_float), POINTER(c_float), POINTER(c_float), POINTER(c_int))
__mul_mat_cu.restypes = c_void_p
__add_mat_cu.argtypes = (POINTER(c_float), POINTER(c_float), POINTER(c_float), POINTER(c_int))
__add_mat_cu.restypes = c_void_p

c_float_p = lambda x: x.ctypes.data_as(POINTER(c_float))
c_int_p = lambda x: x.ctypes.data_as(POINTER(c_int))
pre_time_cu = time.time()
src_cu = copy.deepcopy(src)
dst_cu = src_cu

__mul_const_cu(c_float_p(dst_cu), c_float_p(dst_cu), mul, c_int_p(pnsz))
__add_const_cu(c_float_p(dst_cu), c_float_p(dst_cu), add, c_int_p(pnsz))
__mul_mat_cu(c_float_p(dst_cu), c_float_p(dst_cu), c_float_p(dst_cu), c_int_p(pnsz))
__add_mat_cu(c_float_p(dst_cu), c_float_p(dst_cu), c_float_p(dst_cu), c_int_p(pnsz))

```


# pycuda

- .cu에서는 interface를 위한 코드가 있지만 여기선 없다.
- memcopy를 cuda.in, cuda.out으로 간단히 처리했다.

```python
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


```

# 참고
- [https://cuda.readthedocs.io/ko/latest/PyCUDA_int/](https://cuda.readthedocs.io/ko/latest/PyCUDA_int/)
- [https://www.cudahandbook.com/sample-chapters/](https://www.cudahandbook.com/sample-chapters/)
- [https://www.youtube.com/watch?v=X9mflbX1NL8 Boston University - PyOpenCL, PyCUDA GPU programming](https://www.youtube.com/watch?v=X9mflbX1NL8)