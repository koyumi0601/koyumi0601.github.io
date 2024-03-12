#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <cmath> // ceil
#include <iostream>
#include <helper_cuda.h> // _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor). path: D:\Github_Blog\cuda-samples\Common


void printCudaDevicesInfo();

//__global__ void MatMul(float* deviceMatAPtr, float* deviceMatBPtr, float* deviceMatCPtr, int size_n, int size_k, int size_m);
__global__ void MatMulUnder1024(float* deviceMatAPtr, float* deviceMatBPtr, float* deviceMatCPtr, int m, int n, int k);
__global__ void MatMulOver1024(float* deviceMatAPtr, float* deviceMatBPtr, float* deviceMatCPtr, int m, int n, int k);
void MatMulUnder1024_Helper(char* argv[]);
//
//cudaError_t MatAdd_G2D_B2D_Helper(int* c, const int* a, const int* b, unsigned int COL_SIZE, unsigned int ROW_SIZE);
//__global__ void MatAdd_G2D_B2D(float* MatA, float* MatB, float* MatC, int ROW_SIZE, int COL_SIZE);