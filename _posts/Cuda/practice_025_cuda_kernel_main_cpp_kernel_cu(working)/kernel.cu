#include "kernel.cuh"

__global__ void calculateAverageKernel(int N, int M, int L, float *inputData, float *outputData) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    int m = blockIdx.y * blockDim.y + threadIdx.y;

    if (n < N && m < M) {
        float sum = 0.0f;
        for (int l = 0; l < L; ++l) {
            sum += inputData[(n * M * L) + (m * L) + l]; // l 방향으로의 값 더하기
        }
        outputData[(n * M) + m] = sum / L; // 평균 값 계산
    }
}

void calculateAverage(int N, int M, int L, dim3 numBlocks, dim3 blockSize, float *inputData, float *outputData) {
    calculateAverageKernel<<<numBlocks, blockSize>>>(N, M, L, inputData, outputData);
}