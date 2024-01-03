CMAKE_MINIMUM_REQUIRED(VERSION 3.26)
project(CPP_WITH_CU LANGUAGES CXX CUDA)

add_executable(main 
    src/main.cpp
    src/matmul.cu
)

target_include_directories(main PUBLIC
    include
)