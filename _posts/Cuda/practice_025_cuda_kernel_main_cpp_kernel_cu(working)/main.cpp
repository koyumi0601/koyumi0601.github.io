#include <stdio.h>
#include <stdlib.h> // malloc, free 함수를 사용하기 위해
#include <cuda_runtime.h> // CUDA 런타임 함수를 사용하기 위해
#include "kernel.cuh"

#define N 3 // n 차원의 크기
#define M 3 // m 차원의 크기
#define L 3 // l 차원의 크기

int main() {
    float inputData[N][M][L] = {
        {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}},
        {{2, 4, 6}, {8, 10, 12}, {14, 16, 18}},
        {{3, 6, 9}, {12, 15, 18}, {21, 24, 27}}
    };

    float *outputData, *d_inputData, *d_outputData;

    // 호스트 메모리 할당
    outputData = (float*)malloc(N * M * sizeof(float));

    // GPU 메모리 할당
    cudaMalloc((void**)&d_inputData, N * M * L * sizeof(float));
    cudaMalloc((void**)&d_outputData, N * M * sizeof(float));

    // 입력 데이터를 GPU로 복사
    cudaMemcpy(d_inputData, inputData, N * M * L * sizeof(float), cudaMemcpyHostToDevice);

    // 블록과 스레드 구성
    dim3 blockSize(16, 16);
    dim3 numBlocks((N + blockSize.x - 1) / blockSize.x, (M + blockSize.y - 1) / blockSize.y);

    // CUDA 커널 실행
    calculateAverage(N, M, L, numBlocks, blockSize, d_inputData, d_outputData);
    cudaDeviceSynchronize();

    // 결과를 호스트로 복사
    cudaMemcpy(outputData, d_outputData, N * M * sizeof(float), cudaMemcpyDeviceToHost);

    // 결과 출력
    printf("Average values:\n");
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < M; ++j) {
            printf("%f ", outputData[i * M + j]);
        }
        printf("\n");
    }

    // 메모리 해제
    free(outputData);
    cudaFree(d_inputData);
    cudaFree(d_outputData);

    return 0;
}