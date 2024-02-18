cmake_minimum_required(VERSION 3.10)
project(cuda_project)

find_package(CUDA REQUIRED)
find_package(CUDAToolkit REQUIRED)

cuda_add_executable(myProgram main.cpp kernel.cu)