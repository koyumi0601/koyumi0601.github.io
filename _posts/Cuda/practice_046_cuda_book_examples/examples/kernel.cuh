#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <cmath> // ceil
#include <iostream>
#include <helper_cuda.h> // _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor). path: D:\Github_Blog\cuda-samples\Common
#include "utils.h"
#include <cublas_v2.h>


void printCudaDevicesInfo();

//__global__ void MatMul(float* deviceMatAPtr, float* deviceMatBPtr, float* deviceMatCPtr, int size_n, int size_k, int size_m);
__global__ void MatMulUnder1024(float* deviceMatAPtr, float* deviceMatBPtr, float* deviceMatCPtr, int m, int n, int k);
__global__ void MatMulUnder1024_9_1_sharedMemory(float* deviceMatAPtr, float* deviceMatBPtr, float* deviceMatCPtr);
__global__ void MatMulUnder1024_9_1_globalMemory(float* deviceMatAPtr, float* deviceMatBPtr, float* deviceMatCPtr);


__global__ void MatMulOver1024(float* deviceMatAPtr, float* deviceMatBPtr, float* deviceMatCPtr, int m, int n, int k);
__global__ void MatMulOver1024_10_2_sharedMemory(float* deviceMatAPtr, float* deviceMatBPtr, float* deviceMatCPtr, int m, int n, int k);
__global__ void MatMulOver1024_11_5_sharedMemory_NoBankConflict(float* deviceMatAPtr, float* deviceMatBPtr, float* deviceMatCPtr, int m, int n, int k);
__global__ void MatMulOver1024_xyswap(float* deviceMatAPtr, float* deviceMatBPtr, float* deviceMatCPtr, int m, int n, int k);


//__global__ void MatMulOver1024(float* deviceMatAPtr, float* deviceMatBPtr, float* deviceMatCPtr, int m, int n, int k);
void MatMulUnder1024_Helper();
void MatMulUnder1024_9_1_sharedMemory_Helper();
void MatMulOver1024_Helper();
void MatMulOver1024_10_2_SharedMemory_Helper();
void MatMulOver1024_11_5_SharedMemory_NoBankConflict_Helper();
void MatMulOver1024_xyswap_Helper();

//__global__ void sharedMemory_Static_Kernel(void);
//
//cudaError_t MatAdd_G2D_B2D_Helper(int* c, const int* a, const int* b, unsigned int COL_SIZE, unsigned int ROW_SIZE);
//__global__ void MatAdd_G2D_B2D(float* MatA, float* MatB, float* MatC, int ROW_SIZE, int COL_SIZE);

void cuBLAS_MatMul();

__global__ void syncWarpKernel();
void syncWarp_helper();