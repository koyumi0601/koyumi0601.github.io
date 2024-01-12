---
layout: single
title: "python and cpp dll, cu dll"
categories: cuda
tags: [language, programming, cpp, cuda, python]
toc: true
author_profile: false
sidebar:
    nav: "docs"
search: true

---

*cpp dll, cu dll and python*

- see practice_020_dll

# generate dll
- run visual studio
- create new project > (template) Dynamic Link Library
    - framework.h, pch.h, pch.cpp는 그대로 사용한다.
    - dllmain.cpp는 삭제한다
    - math_clang.cpp, math_clang.h 생성한다.

- math_clang.cpp
    - 원래의 코드에서 #include "pch.h" #include "math_clang.h"만 추가되었다.

```cpp
#include "pch.h" // added, use stdafx.h in Visual Studio 2017 and earlier
#include "math_clang.h" // added
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

void mul_const(float* pddst, float* pdsrc, float dconst, int* pnsz) {
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

  return;
}


/*
 * Adds a constant value to each element of the source array.
 *
 * @param pddst     Destination array to store the result.
 * @param pdsrc     Source array to which the constant is added.
 * @param dconst    Constant value to be added to each element.
 * @param pnsz      Integer array containing the dimensions [nx, ny, nz].
 */
void add_const(float* pddst, float* pdsrc, float dconst, int* pnsz) {
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

  return;

}

/*
 * Multiplies two arrays element-wise and stores the result in the destination array.
 *
 * @param pddst     Destination array to store the result.
 * @param pdsrc     Source array to be multiplied with another source array.
 * @param pdsrc2    Second source array for element-wise multiplication.
 * @param pnsz      Integer array containing the dimensions [nx, ny, nz].
 */
void mul_mat(float* pddst, float* pdsrc, float* pdsrc2, int* pnsz) {
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

  return;

}

/*
 * Adds two arrays element-wise and stores the result in the destination array.
 *
 * @param pddst     Destination array to store the result.
 * @param pdsrc     Source array to which the second source array is added.
 * @param pdsrc2    Second source array for element-wise addition.
 * @param pnsz      Integer array containing the dimensions [nx, ny, nz].
 */
void add_mat(float* pddst, float* pdsrc, float* pdsrc2, int* pnsz) {
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

  return;

}

// gcc -c -fPIC math_clang.c -o math_clang.o
// gcc -shared math_clang.o -o libmath_clang.so
```


- math_clang.h
    - __declspec(dllexport), __declspec(dllimport)가 추가되었다.

```cpp
#pragma once

#ifdef MATHLIBRARY_EXPORTS
#define MATHLIBRARY_API __declspec(dllexport)
#else
#define MATHLIBRARY_API __declspec(dllimport)
#endif

extern "C" MATHLIBRARY_API void mul_const(float* pddst, float* pdsrc, float dconst, int* pnsz);
extern "C" MATHLIBRARY_API void add_const(float* pddst, float* pdsrc, float dconst, int* pnsz);
extern "C" MATHLIBRARY_API void mul_mat(float* pddst, float* pdsrc, float* pdsrc2, int* pnsz);
extern "C" MATHLIBRARY_API void add_mat(float* pddst, float* pdsrc, float* pdsrc2, int* pnsz);
```

- build
    - math_clang.dll, math_clang.pdb, math_clang.exp, math_clang.lib 네 개가 모두 나와야 구성이 잘 된 것.
    - 여기서 math_clang.dll만 있으면 가져다가 쓸 수 있다. python과 동일 경로에 이동한다.


- python
    - ctypes를 통해서 가져온다
    - 기존의 linux에서 so를 썼던 것과 동일하다. dll 생성해내는 것이 문제였다.


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

## numpy in CPU
pre_time_numpy = time.time()
src_numpy = copy.deepcopy(src)
dst_numpy = src_numpy
dst_numpy = dst_numpy * mul
dst_numpy = dst_numpy + add
dst_numpy = dst_numpy * dst_numpy
dst_numpy = dst_numpy + dst_numpy
after_time_numpy = time.time()
print(f'elapsed time numpy: {after_time_numpy - pre_time_numpy}')
print("Result: ", dst_numpy[:4,0,0])

## Clang in CPU
if platform == 'window':
    clang_file = os.path.join(os.path.dirname(__file__), 'math_clang.dll')
elif platform == 'linux':
    clang_file = os.path.join(os.getcwd(), 'libmath_clang.so')

_math_clang = ctypes.CDLL(clang_file)

__mul_const_clang = _math_clang.mul_const
__add_const_clang = _math_clang.add_const
__mul_mat_clang = _math_clang.mul_mat
__add_mat_clang = _math_clang.add_mat

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
pre_time_clang = time.time()
src_clang = copy.deepcopy(src)
dst_clang = src_clang
__mul_const_clang(c_float_p(dst_clang), c_float_p(dst_clang), mul, c_int_p(pnsz))
__add_const_clang(c_float_p(dst_clang), c_float_p(dst_clang), add, c_int_p(pnsz))
__mul_mat_clang(c_float_p(dst_clang), c_float_p(dst_clang), c_float_p(dst_clang), c_int_p(pnsz))
__add_mat_clang(c_float_p(dst_clang), c_float_p(dst_clang), c_float_p(dst_clang), c_int_p(pnsz))

after_time_clang = time.time()
print(f'elapsed time clang: {after_time_clang - pre_time_clang}')
print("Result: ", dst_clang[:4,0,0])


```