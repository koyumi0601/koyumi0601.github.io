// add_kernel.h

#ifndef ADD_KERNEL_H
#define ADD_KERNEL_H

extern "C" __global__ void add_kernel(float *a, float *b, float *c, int n);

#endif  // ADD_KERNEL_H