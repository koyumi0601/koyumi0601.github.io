#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

// Kernel function to add two vectors on GPU
__global__ void addKernel(int* c, const int* a, const int* b, int size)
{
  int i = threadIdx.x;
  if (i < size)
    c[i] = a[i] + b[i];
}

// Helper function for using CUDA to add vectors in parallel.
void addWithCuda(int* c, const int* a, const int* b, int size)
{
  int* dev_a = 0;
  int* dev_b = 0;
  int* dev_c = 0;

  // Allocate GPU memory for a, b, and c
  cudaMalloc((void**)&dev_a, size * sizeof(int));
  cudaMalloc((void**)&dev_b, size * sizeof(int));
  cudaMalloc((void**)&dev_c, size * sizeof(int));

  // Copy input vectors from host memory to GPU buffers
  cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);

  // Launch a kernel on the GPU with one thread for each element.
  addKernel << <1, size >> > (dev_c, dev_a, dev_b, size);

  // Copy result from GPU buffer to host memory
  cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);

  // Free GPU memory
  cudaFree(dev_c);
  cudaFree(dev_a);
  cudaFree(dev_b);
}

int main()
{
  const int arraySize = 5;
  const int a[arraySize] = { 1, 2, 3, 4, 5 };
  const int b[arraySize] = { 10, 20, 30, 40, 50 };
  int c[arraySize] = { 0 };

  addWithCuda(c, a, b, arraySize);

  printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
    c[0], c[1], c[2], c[3], c[4]);

  return 0;
}