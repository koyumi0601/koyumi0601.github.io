project(CMakeProject1)

find_package(CUDA REQUIRED)

cuda_add_executable(my_cuda_program cuda_code.cu)

target_link_libraries(my_cuda_program CUDA::cudart)