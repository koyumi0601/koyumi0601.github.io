#include "add_kernel.cuh"
#include <stdio.h>

__global__ void add_kernel(float *a, float *b, float *c, int n) {
    // __global__ 키워드를 붙이면 Device에서 작동된다.
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}