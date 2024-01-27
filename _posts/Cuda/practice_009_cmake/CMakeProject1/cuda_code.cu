#include <stdio.h>

__global__ void addVectors(int* a, int* b, int* c, int n) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < n) {
    c[idx] = a[idx] + b[idx];
  }
}