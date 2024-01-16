// kernel.cu
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ void addKernel(int* c, const int* a, const int* b, unsigned int size)
{
  int i = threadIdx.x;
  if (i < size) {
    c[i] = a[i] + b[i];
  }
}

extern "C" __declspec(dllexport) void addWithCuda(int* c, const int* a, const int* b, unsigned int size)
{
  int* dev_a = 0;
  int* dev_b = 0;
  int* dev_c = 0;

  cudaMalloc((void**)&dev_a, size * sizeof(int));
  cudaMalloc((void**)&dev_b, size * sizeof(int));
  cudaMalloc((void**)&dev_c, size * sizeof(int));

  cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);

  addKernel << <1, size >> > (dev_c, dev_a, dev_b, size);

  cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);

  cudaFree(dev_a);
  cudaFree(dev_b);
  cudaFree(dev_c);
}