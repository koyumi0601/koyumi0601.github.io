#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include "pch.h"

__global__ void addKernel(int* c, const int* a, const int* b, int size)
{
  int i = threadIdx.x;
  if (i < size) {
    c[i] = a[i] + b[i];
  }
}

extern "C" __declspec(dllexport) void addWithCuda(int* c, const int* a, const int* b, int size) {
  addKernel <<<1, size >>> (c, a, b, size);
  cudaDeviceSynchronize();
}