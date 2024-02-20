#include "kernel.cuh"

__global__ void kernel(int N, float* input, float* output) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    output[idx] = input[idx] * 2; // 예시: 입력 값을 2배로 변경
  }
}

void cudaKernel(int N, float* input, float* output) {
  float* d_input, * d_output;

  // CUDA 메모리 할당
  cudaMalloc(&d_input, N * sizeof(float));
  cudaMalloc(&d_output, N * sizeof(float));

  // 입력 데이터 복사
  cudaMemcpy(d_input, input, N * sizeof(float), cudaMemcpyHostToDevice);

  // CUDA 커널 실행
  kernel << <(N + 255) / 256, 256 >> > (N, d_input, d_output);

  // 결과 데이터 복사
  cudaMemcpy(output, d_output, N * sizeof(float), cudaMemcpyDeviceToHost);

  // CUDA 메모리 해제
  cudaFree(d_input);
  cudaFree(d_output);
}