#ifndef KERNEL_CUH_
#define KERNEL_CUH_

#include <cuda_runtime.h>

void calculateAverage(int N, int M, int L, dim3 numBlocks, dim3 blockSize, float *inputData, float *outputData);

#endif // KERNEL_CUH_