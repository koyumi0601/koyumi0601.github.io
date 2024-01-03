## python, c, cu

import os, copy
import numpy as np
import ctypes
from ctypes import *


pnsz = np.asarray([100, 50, 30], dtype = np.int32)

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

## Clang in CPU
clang_file = os.path.join(os.path.dirname(__file__), 'libmath_clang.so')
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

src_clang = copy.deepcopy(src)
dst_clang = src_clang
__mul_const_clang(c_float_p(dst_clang), c_float_p(dst_clang), mul, c_int_p(pnsz))
__add_const_clang(c_float_p(dst_clang), c_float_p(dst_clang), add, c_int_p(pnsz))
__mul_mat_clang(c_float_p(dst_clang), c_float_p(dst_clang), c_float_p(dst_clang), c_int_p(pnsz))
__add_mat_clang(c_float_p(dst_clang), c_float_p(dst_clang), c_float_p(dst_clang), c_int_p(pnsz))



##
print(dst_numpy[:10, 0, 0])
print(dst_clang[:10, 0, 0])
print(np.sum((dst_numpy - dst_clang)))



## CU in GPU

cu_file = os.path.join(os.path.dirname(__file__), 'libmath_cu.so')
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

src_cu = copy.deepcopy(src)
dst_cu = src_cu

__mul_const_cu(c_float_p(dst_cu), c_float_p(dst_cu), mul, c_int_p(pnsz))
__add_const_cu(c_float_p(dst_cu), c_float_p(dst_cu), add, c_int_p(pnsz))
__mul_mat_cu(c_float_p(dst_cu), c_float_p(dst_cu), c_float_p(dst_cu), c_int_p(pnsz))
__add_mat_cu(c_float_p(dst_cu), c_float_p(dst_cu), c_float_p(dst_cu), c_int_p(pnsz))

print(dst_numpy[:10, 0, 0])
print(dst_cu[:10, 0, 0])
print(np.sum(dst_numpy - dst_cu))

# https://www.youtube.com/watch?v=LO2qKHp2jLg 27:33
