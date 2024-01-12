#include <stdio.h>
#include <iostream>  // <iostream> 헤더 추가

__global__ void cuda_hello() {
  printf("Hello World from GPU!\n");
}

int main() {
  cuda_hello << <10, 1 >> > ();
  cudaDeviceSynchronize(); // sync하지 않으면 아래의 Press Enter to continue가 먼저 실행되기도 함
  std::cout << "Press Enter to continue...";
  std::cin.get();
  return 0;
}