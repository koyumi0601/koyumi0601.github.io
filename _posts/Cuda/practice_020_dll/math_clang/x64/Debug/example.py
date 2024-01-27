import os, copy
import numpy as np
import ctypes
from ctypes import *
import sys
import time

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

lib = ctypes.cdll.LoadLibrary('./math_clang.dll')
#또는 lib = ctypes.CDLL('./mydll.dll')

__mul_const_clang = lib.mul_const
print(__mul_const_clang)
__mul_const_clang.argtypes = (POINTER(c_float), POINTER(c_float), c_float, POINTER(c_int)) # argument
__mul_const_clang.restypes = c_void_p # return variable
c_float_p = lambda x: x.ctypes.data_as(POINTER(c_float))
c_int_p = lambda x: x.ctypes.data_as(POINTER(c_int))

pre_time_clang = time.time()
src_clang = copy.deepcopy(src)
dst_clang = src_clang
__mul_const_clang(c_float_p(dst_clang), c_float_p(dst_clang), mul, c_int_p(pnsz))


after_time_clang = time.time()
print(f'elapsed time clang: {after_time_clang - pre_time_clang}')
print("Result: ", dst_clang[:4,0,0])
