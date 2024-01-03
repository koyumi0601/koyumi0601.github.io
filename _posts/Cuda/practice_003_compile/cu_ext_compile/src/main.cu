#include <iostream>
#include <cuda_runtime.h>
#include "add_kernel.cuh" 


int main() {
    int dataSize = 1000;
    std::cout << "Hello!" << std::endl;
    float *a, *b, *c;

    // 데이터 할당 및 초기화
    cudaMallocManaged(&a, dataSize * sizeof(float));
    cudaMallocManaged(&b, dataSize * sizeof(float));
    cudaMallocManaged(&c, dataSize * sizeof(float));
    
    for (int i = 0; i < dataSize; ++i) {
        a[i] = 1.0f;
        b[i] = 2.0f;
    }

    // GPU에서 커널 실행
    int block_size = 256;
    int grid_size = (dataSize + block_size - 1) / block_size;
    add_kernel<<<grid_size, block_size>>>(a, b, c, dataSize);
    // cudaDeviceSynchronize();

    // // 결과 출력
    // std::cout << "Result: ";
    // for (int i = 0; i < dataSize; ++i) {
    //     std::cout << c[i] << " ";
    // }
    // std::cout << std::endl;

    // // 메모리 해제
    // cudaFree(a);
    // cudaFree(b);
    // cudaFree(c);

    return 0;
}