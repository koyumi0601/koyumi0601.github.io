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
    #clang_file = os.path.join(os.path.dirname(__file__), 'libmath_clang_win.so')
    clang_file = os.path.join(os.path.dirname(__file__), 'mydll.dll')
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
