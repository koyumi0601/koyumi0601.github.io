#include <iostream>
#include "kernel.cuh" // CUDA 커널 선언이 있는 헤더 파일을 포함합니다.

int main() {
  const int N = 10;
  float* h_input = new float[N];
  float* h_output = new float[N];

  // 입력 데이터 초기화
  for (int i = 0; i < N; ++i) {
    h_input[i] = i;
  }

  // CUDA 커널 호출
  cudaKernel(N, h_input, h_output);

  // 결과 출력
  std::cout << "Output: ";
  for (int i = 0; i < N; ++i) {
    std::cout << h_output[i] << " ";
  }
  std::cout << std::endl;

  delete[] h_input;
  delete[] h_output;

  return 0;
}