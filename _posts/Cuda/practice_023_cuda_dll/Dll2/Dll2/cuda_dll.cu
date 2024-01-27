#include <stdio.h>
#include <stdlib.h>
#include "cuda_dll.h"
#include "pch.h"

__global__ void kernel(int n)
{
  printf("kernel: n = %d\n", n);
}

void wrapper(int n)
{
  printf("wrapper: calling kernel()\n");
  kernel << <1, 1 >> > (n);
  cudaDeviceSynchronize();
}