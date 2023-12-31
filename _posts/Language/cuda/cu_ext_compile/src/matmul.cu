#include "matmul.cuh"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <chrono>
#include <cmath>

#include <stdio.h>
#include <stdlib.h>

__global__ void matMul(int* dA, int* dB, int* dC, int m, int n, int k)
{
	int row = blockDim.x * blockIdx.x + threadIdx.x;
	int col = blockDim.y * blockIdx.y + threadIdx.y;
	int index = row * n + col;
	if (row >= m || col >= n)
	{
		return;
	}
	int sum = 0;
	for (int i = 0; i < k; ++i)
	{
		sum += dA[row * k + i] * dB[n * i + col];
	}
	dC[index] = sum;
}

void matMulWrapper(int* A, int* B, int* C, int m, int n, int k)
{
    std::chrono::system_clock::time_point start = std::chrono::system_clock::now();
    int* dA, * dB, * dC;
	cudaMalloc(&dA, m * k * sizeof(int));
	cudaMemset(dA, 0, m * k * sizeof(int));
	cudaMalloc(&dB, k * n * sizeof(int));
	cudaMemset(dB, 0, k * n * sizeof(int));
	cudaMalloc(&dC, m * n * sizeof(int));
	cudaMemset(dC, 0, m * n * sizeof(int));
    std::chrono::duration<double> timeCudaMalloc = std::chrono::system_clock::now() - start;

    start = std::chrono::system_clock::now();
	cudaMemcpy(dA, A, m * k * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dB, B, k * n * sizeof(int), cudaMemcpyHostToDevice);
	std::chrono::duration<double> timeCudaMemcpyToDevice = std::chrono::system_clock::now() - start;

	int blockSize = 32;
	dim3 gridDim(ceil(static_cast<float>(m) / blockSize), ceil(static_cast<float>(n) / blockSize));
	dim3 blockDim(blockSize, blockSize);
	printf("Grid(%d, %d), Block(%d, %d)\n", gridDim.x, gridDim.y, blockDim.x, blockDim.y);

    start = std::chrono::system_clock::now();
	matMul <<<gridDim, blockDim >>> (dA, dB, dC, m, n, k);
	cudaDeviceSynchronize();
    std::chrono::duration<double> timeCudaMatmul = std::chrono::system_clock::now() - start;

    start = std::chrono::system_clock::now();
	cudaMemcpy(C, dC, m * n * sizeof(int), cudaMemcpyDeviceToHost);
    std::chrono::duration<double> timeCudaMemcpyToHost = std::chrono::system_clock::now() - start;

    cudaFree(dA);
	cudaFree(dB);
	cudaFree(dC);

    printf("cuda malloc : %lf(ms), cuda memcpy to device : %lf(ms), cuda matmul : %lf(ms), cuda memcpy to host : %lf(ms)\n", timeCudaMalloc*1000, timeCudaMemcpyToDevice*1000, timeCudaMatmul*1000, timeCudaMemcpyToHost*1000);   
}