#include <iostream>
#include "utils.h"
#include <vector>
#include "example01.h"
#include "kernel.cuh"


int main(int argc, char* argv[]) {
//  int main() {

  // exampleVector();
  printCudaDevicesInfo();
  // MatMulUnder1024_Helper();
  // MatMulUnder1024_9_1_sharedMemory_Helper();
  //
  //MatMulOver1024_Helper();
  //MatMulOver1024_10_2_SharedMemory_Helper();
  //MatMulOver1024_Helper();
  //MatMulOver1024_10_2_SharedMemory_Helper();
  //MatMulOver1024_Helper();
  //MatMulOver1024_10_2_SharedMemory_Helper();
  //MatMulOver1024_Helper();
  //MatMulOver1024_10_2_SharedMemory_Helper();
  //std::cout << "Hello1?\n";
  //MatMulOver1024_Helper();
  //MatMulOver1024_10_2_SharedMemory_Helper();
  //std::cout << "Hello?\n";
  //cuBLAS_MatMul();
  //std::cout << "Hello?\n";
  //cuBLAS_MatMul();
  //std::cout << "Hello?\n";
  //cuBLAS_MatMul();
  //std::cout << "Hello?\n";
//  MatMulOver1024_Helper();
  //MatMulOver1024_Helper();
  //MatMulOver1024_Helper();
  //MatMulOver1024_Helper();
  //MatMulOver1024_Helper();
  //MatMulOver1024_Helper();
//  cuBLAS_MatMul();
  //cuBLAS_MatMul();
  //cuBLAS_MatMul();
  //cuBLAS_MatMul();
  //cuBLAS_MatMul();
  //cuBLAS_MatMul();
  //MatMulOver1024_11_5_SharedMemory_NoBankConflict_Helper();
  //MatMulOver1024_11_5_SharedMemory_NoBankConflict_Helper();
  //MatMulOver1024_11_5_SharedMemory_NoBankConflict_Helper();
  //MatMulOver1024_11_5_SharedMemory_NoBankConflict_Helper();
  //MatMulOver1024_11_5_SharedMemory_NoBankConflict_Helper();

  //MatMulOver1024_10_2_SharedMemory_Helper();
  //std::cout << "Hello\n";
  //MatMulOver1024_10_2_SharedMemory_Helper();
  //MatMulOver1024_10_2_SharedMemory_Helper();
  //MatMulOver1024_10_2_SharedMemory_Helper();
  //MatMulOver1024_10_2_SharedMemory_Helper();

  //MatMulOver1024_10_2_SharedMemory_Helper();
  //MatMulOver1024_10_2_SharedMemory_Helper();
  //MatMulOver1024_10_2_SharedMemory_Helper();
  //MatMulOver1024_10_2_SharedMemory_Helper();
  //MatMulOver1024_Helper();
  //MatMulOver1024_Helper();
  //MatMulOver1024_Helper();
  //MatMulOver1024_Helper();
  //MatMulOver1024_Helper();
  //MatMulOver1024_10_2_SharedMemory_Helper();
  //MatMulOver1024_10_2_SharedMemory_Helper();
  //MatMulOver1024_10_2_SharedMemory_Helper();
  //MatMulOver1024_10_2_SharedMemory_Helper();
  //MatMulOver1024_10_2_SharedMemory_Helper();
  //MatMulOver1024_xyswap_Helper();
  //MatMulOver1024_xyswap_Helper();
  //MatMulOver1024_xyswap_Helper();
  //MatMulOver1024_xyswap_Helper();
  //MatMulOver1024_xyswap_Helper();

  syncWarp_helper();

  return 0;
}
