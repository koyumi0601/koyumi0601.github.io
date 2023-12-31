#include <iostream>
#include <cuda_runtime.h>

int main() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for (int deviceIdx = 0; deviceIdx < deviceCount; ++deviceIdx) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, deviceIdx);

        std::cout << "Device " << deviceIdx << ": " << deviceProp.name << std::endl;
        std::cout << "  Compute Capability: " << deviceProp.major << "." << deviceProp.minor << std::endl;
        std::cout << "  Multiprocessors: " << deviceProp.multiProcessorCount << std::endl;
        std::cout << "  CUDA Cores per Multiprocessor: " << deviceProp.warpSize << std::endl;
        std::cout << "  Global Memory: " << deviceProp.totalGlobalMem / (1024 * 1024) << " MB" << std::endl;
        std::cout << "  Shared Memory per Block: " << deviceProp.sharedMemPerBlock / 1024 << " KB" << std::endl;
        std::cout << "  Max Threads per Block: " << deviceProp.maxThreadsPerBlock << std::endl;
        std::cout << "  Max Threads per Dimension: (" << deviceProp.maxThreadsDim[0] << ", "
                  << deviceProp.maxThreadsDim[1] << ", " << deviceProp.maxThreadsDim[2] << ")" << std::endl;
        std::cout << "  Max Grid Size: (" << deviceProp.maxGridSize[0] << ", " << deviceProp.maxGridSize[1] << ", "
                  << deviceProp.maxGridSize[2] << ")" << std::endl;
        // Tensor Core 지원 여부 확인
        if (deviceProp.major >= 7 && deviceProp.minor >= 0) {
            std::cout << "  Tensor Core 지원: Yes" << std::endl;
        } else {
            std::cout << "  Tensor Core 지원: No" << std::endl;
        }          

        std::cout << std::endl;
    }

    return 0;
}