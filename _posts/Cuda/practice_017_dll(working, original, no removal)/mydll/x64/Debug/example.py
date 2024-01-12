import ctypes
import time
from array import array
lib = ctypes.cdll.LoadLibrary('./mydll.dll')
#또는 lib = ctypes.CDLL('./mydll.dll')

#Python만으로 구성된 elementwise list add
def py_add(_a, _b):
    temp = [None for i in _a]
    cnt = 0
    for i, j in zip(_a, _b):
        temp[cnt] = i + j
        cnt += 1
    return temp

cnt = 100000000
temp1 = [i for i in range(cnt)]
temp2 = [i for i in range(cnt)]
st = time.time()
ret_py = py_add(temp1, temp2)
print("python def time : ", time.time() - st)

#Python List를 Ctypes를 이용해서 Ctypes Array로 변환합니다.
#이때 바로 List -> Ctypes Array로 변환할 경우 엄청난 병목이 발생합니다.
#때문에 Python List -> Python Array -> Ctypes Array 순으로 변환해줍니다.
temp1_1 = array('I', temp1)
temp1_1 = (ctypes.c_int * len(temp1)).from_buffer(temp1_1)
temp2_1 = array('I', temp2)
temp2_1 = (ctypes.c_int * len(temp2)).from_buffer(temp2_1)
#반환값의 형태를 미리 설정해 줍니다.
#반환타입 설정 없이 Pointer를 return할 경우 Python에서는 반환값이 Int로 강제 형변환이 발생합니다.
lib.my_list_add.restype = ctypes.POINTER(ctypes.c_int)
ret_array = lib.my_list_add(temp1_1, temp2_1, len(temp1_1))
print("ctypes time : ", time.time() - st)
#ret_array는 Ctypes Pointer/Array이면서, [:] Slice 방법으로 List와같은 접근이 가능합니다.
#하지만 Pointer의 끝을 알 방법이 없기 때문에, ret_array[:cnt]와 같은 형태로 접근이 필요합니다.
#ret_array[cnt+1]로 접근할 경우 메모리 Access위반이 발생하여 Python Run Time이 종료됩니다.
