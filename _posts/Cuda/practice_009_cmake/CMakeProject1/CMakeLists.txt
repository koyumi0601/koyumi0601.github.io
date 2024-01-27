cmake_minimum_required(VERSION 3.20)
project(CMakeProject1) # project name

find_package(CUDAToolkit REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

find_package(PythonLibs REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS})

# add_subdirectory(src) # source code directory

add_executable(my_executable python_script.py) #

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-O3 -arch=sm_52) # 원하는 GPU 아키텍처로 변경

file(GLOB CUDA_SOURCES "*.cu")
cuda_add_executable(my_cuda_executable ${CUDA_SOURCES})

file(GLOB PYTHON_SOURCES "*.cpp")
target_link_libraries(my_executable ${PYTHON_LIBRARIES})

#set(CMAKE_CUDA_COMPILER "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.3/bin/nvcc.exe")  # nvcc 컴파일러 경로 설정
#set(CUDA_TOOLKIT_ROOT_DIR "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.3")
#set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -arch=sm_52)  # 원하는 GPU 아키텍처에 맞게 설정



#project(CPP_WITH_CU LANGUAGES CXX CUDA)


#target_link_libraries(my_cuda_program PRIVATE CUDA::cudart)

#add_executable(my_cuda_program cuda_code.cu)

#WITH_CUDA = ON


#if(COMMAND cmake_policy)
#    cmake_policy(SET CMP0003 NEW)
#endif(COMMAND cmake_policy)