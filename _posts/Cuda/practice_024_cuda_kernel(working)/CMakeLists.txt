cmake_minimum_required(VERSION 3.10)
project(cuda_project)

find_package(CUDA REQUIRED)

cuda_add_executable(main main.cu)