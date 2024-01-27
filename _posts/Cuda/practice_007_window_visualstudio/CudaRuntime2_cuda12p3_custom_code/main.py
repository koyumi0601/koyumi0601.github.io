print("Hello")
import platform
print(platform.architecture())

import os
import ctypes

# CUDA 커널 파일(.obj) 경로
cuda_kernel_file = os.path.abspath('D:/GitHub_Project/koyumi0601.github.io/_posts/Cuda/practice_007_window_visualstudio/CudaRuntime2_cuda12p3_custom_code/x64/Release/kernel.cu.dll')

# CUDA 커널 파일을 로드합니다.
cuda_kernel = ctypes.CDLL(cuda_kernel_file)

# # CUDA 커널 함수의 인자 타입을 설정합니다.
# # 예를 들어, 함수의 인자 타입이 float* 인 경우 c_float_p를 사용하여 넘겨줍니다.
# c_float_p = ctypes.POINTER(ctypes.c_float)
# cuda_kernel.some_cuda_function.argtypes = (c_float_p, c_float_p, ctypes.c_int)

# # CUDA 커널 함수를 호출할 데이터를 준비합니다.
# # 예를 들어, 입력 데이터와 결과 데이터를 생성합니다.
# input_data = [1.0, 2.0, 3.0]  # 예시 데이터
# input_size = len(input_data)
# output_data = [0.0] * input_size

# # CUDA 커널 함수를 호출합니다.
# cuda_kernel.some_cuda_function(input_data, output_data, input_size)

# # 결과를 출력합니다.
# print("CUDA 결과:", output_data)

# import os
# import ctypes

# # 현재 파이썬 스크립트의 디렉토리를 구합니다.
# current_directory = os.path.dirname(os.path.abspath(__file__))

# # CUDA 커널 파일(.obj)의 경로를 구합니다.
# cuda_kernel_file = os.path.join(current_directory, 'x64', 'Release', 'kernel.cu.obj')

# # CUDA 커널 파일을 로드합니다.
# cuda_kernel = ctypes.CDLL(cuda_kernel_file)