#include "stdio.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <cuda.h>

//__global__ void addArrays(int* c, const int* a, const int* b, unsigned int size);


__global__ void mul_const_kernel(float* pddst, float* pdsrc, float dconst, int* pnsz) {
  int nx = pnsz[0];
  int ny = pnsz[1];
  int nz = pnsz[2];
  int id = 0;

  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  int idy = blockDim.y * blockIdx.y + threadIdx.y;
  int idz = blockDim.z * blockIdx.z + threadIdx.z;

  if (idz >= nz || idy >= ny || idx >= nx) return;

  id = ny * nx * idz + nx * idy + idx;
  pddst[id] = pdsrc[id] * dconst;


  return;
}


__global__ void add_const_kernel(float* pddst, float* pdsrc, float dconst, int* pnsz) {
  int nx = pnsz[0];
  int ny = pnsz[1];
  int nz = pnsz[2];
  int id = 0;

  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  int idy = blockDim.y * blockIdx.y + threadIdx.y;
  int idz = blockDim.z * blockIdx.z + threadIdx.z;

  if (idz >= nz || idy >= ny || idx >= nx) return;

  id = ny * nx * idz + nx * idy + idx;
  pddst[id] = pdsrc[id] + dconst;

  return;

}


__global__ void mul_mat_kernel(float* pddst, float* pdsrc, float* pdsrc2, int* pnsz) {
  int nx = pnsz[0];
  int ny = pnsz[1];
  int nz = pnsz[2];
  int id = 0;

  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  int idy = blockDim.y * blockIdx.y + threadIdx.y;
  int idz = blockDim.z * blockIdx.z + threadIdx.z;

  if (idz >= nz || idy >= ny || idx >= nx) return;

  id = ny * nx * idz + nx * idy + idx;
  pddst[id] = pdsrc[id] * pdsrc2[id];

  return;

}


__global__ void add_mat_kernel(float* pddst, float* pdsrc, float* pdsrc2, int* pnsz) {
  int nx = pnsz[0];
  int ny = pnsz[1];
  int nz = pnsz[2];
  int id = 0;

  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  int idy = blockDim.y * blockIdx.y + threadIdx.y;
  int idz = blockDim.z * blockIdx.z + threadIdx.z;

  if (idz >= nz || idy >= ny || idx >= nx) return;

  id = ny * nx * idz + nx * idy + idx;
  pddst[id] = pdsrc[id] + pdsrc2[id];

  return;

}


// cpu interface
// DLLEXPORT void mul_const(float *pddst, float *pdsrc, float dconst, int *pnsz){ // window only expression
extern "C" void mul_const(float* pddst, float* pdsrc, float dconst, int* pnsz) { // available on linux, window
  float* gpddst = 0;
  float* gpdsrc = 0;
  int* gpnsz = 0;

  cudaMalloc((void**)&gpddst, pnsz[0] * pnsz[1] * pnsz[2] * sizeof(float));
  cudaMalloc((void**)&gpdsrc, pnsz[0] * pnsz[1] * pnsz[2] * sizeof(float));
  cudaMalloc((void**)&gpnsz, 3 * sizeof(int));

  cudaMemset(gpddst, 0, pnsz[0] * pnsz[1] * pnsz[2] * sizeof(float));
  cudaMemset(gpdsrc, 0, pnsz[0] * pnsz[1] * pnsz[2] * sizeof(float));
  cudaMemset(gpnsz, 0, 3 * sizeof(float));

  cudaMemcpy(gpdsrc, pdsrc, pnsz[0] * pnsz[1] * pnsz[2] * sizeof(float), cudaMemcpyHostToDevice); // destination, source, memory size, direction
  cudaMemcpy(gpnsz, pnsz, 3 * sizeof(int), cudaMemcpyHostToDevice); // destination, source, memory size, direction

  int nthread = 8;
  dim3 nblock(nthread, nthread, nthread);
  dim3 ngrid((pnsz[0] + nthread - 1) / nthread,
    (pnsz[1] + nthread - 1) / nthread,
    (pnsz[2] + nthread - 1) / nthread);

  mul_const_kernel <<<ngrid, nblock >>> (gpddst, gpdsrc, dconst, gpnsz);

  cudaMemcpy(pddst, gpddst, pnsz[0] * pnsz[1] * pnsz[2] * sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(gpddst);
  cudaFree(gpdsrc);
  cudaFree(gpnsz);

  gpddst = 0;
  gpdsrc = 0;
  gpnsz = 0;

  return;
}


// DLLEXPORT void add_const(float *pddst, float *pdsrc, float dconst, int *pnsz){ // window only
extern "C" void add_const(float* pddst, float* pdsrc, float dconst, int* pnsz) { // available on window and linux 
  float* gpddst = 0;
  float* gpdsrc = 0;
  int* gpnsz = 0;

  cudaMalloc((void**)&gpddst, pnsz[0] * pnsz[1] * pnsz[2] * sizeof(float));
  cudaMalloc((void**)&gpdsrc, pnsz[0] * pnsz[1] * pnsz[2] * sizeof(float));
  cudaMalloc((void**)&gpnsz, 3 * sizeof(int));

  cudaMemset(gpddst, 0, pnsz[0] * pnsz[1] * pnsz[2] * sizeof(float));
  cudaMemset(gpdsrc, 0, pnsz[0] * pnsz[1] * pnsz[2] * sizeof(float));
  cudaMemset(gpnsz, 0, 3 * sizeof(float));

  cudaMemcpy(gpdsrc, pdsrc, pnsz[0] * pnsz[1] * pnsz[2] * sizeof(float), cudaMemcpyHostToDevice); // destination, source, memory size, direction
  cudaMemcpy(gpnsz, pnsz, 3 * sizeof(int), cudaMemcpyHostToDevice); // destination, source, memory size, direction

  int nthread = 8;
  dim3 nblock(nthread, nthread, nthread);
  dim3 ngrid((pnsz[0] + nthread - 1) / nthread,
    (pnsz[1] + nthread - 1) / nthread,
    (pnsz[2] + nthread - 1) / nthread);

  add_const_kernel << <ngrid, nblock >> > (gpddst, gpdsrc, dconst, gpnsz);

  cudaMemcpy(pddst, gpddst, pnsz[0] * pnsz[1] * pnsz[2] * sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(gpddst);
  cudaFree(gpdsrc);
  cudaFree(gpnsz);

  gpddst = 0;
  gpdsrc = 0;
  gpnsz = 0;

  return;
}



// DLLEXPORT void mul_mat(float *pddst, float *pdsrc1, float *pdsrc2, int *pnsz){ // window only
extern "C" void mul_mat(float* pddst, float* pdsrc1, float* pdsrc2, int* pnsz) { // available on window and linux
  float* gpddst = 0;
  float* gpdsrc1 = 0;
  float* gpdsrc2 = 0;
  int* gpnsz = 0;

  cudaMalloc((void**)&gpddst, pnsz[0] * pnsz[1] * pnsz[2] * sizeof(float));
  cudaMalloc((void**)&gpdsrc1, pnsz[0] * pnsz[1] * pnsz[2] * sizeof(float));
  cudaMalloc((void**)&gpdsrc2, pnsz[0] * pnsz[1] * pnsz[2] * sizeof(float));
  cudaMalloc((void**)&gpnsz, 3 * sizeof(int));


  cudaMemset(gpddst, 0, pnsz[0] * pnsz[1] * pnsz[2] * sizeof(float));
  cudaMemset(gpdsrc1, 0, pnsz[0] * pnsz[1] * pnsz[2] * sizeof(float));
  cudaMemset(gpdsrc2, 0, pnsz[0] * pnsz[1] * pnsz[2] * sizeof(float));
  cudaMemset(gpnsz, 0, 3 * sizeof(int));

  cudaMemcpy(gpdsrc1, pdsrc1, pnsz[0] * pnsz[1] * pnsz[2] * sizeof(float), cudaMemcpyHostToDevice); // destination, source, memory size, direction
  cudaMemcpy(gpdsrc2, pdsrc2, pnsz[0] * pnsz[1] * pnsz[2] * sizeof(float), cudaMemcpyHostToDevice); // destination, source, memory size, direction
  cudaMemcpy(gpnsz, pnsz, 3 * sizeof(int), cudaMemcpyHostToDevice); // destination, source, memory size, direction

  int nthread = 8;
  dim3 nblock(nthread, nthread, nthread);
  dim3 ngrid((pnsz[0] + nthread - 1) / nthread,
    (pnsz[1] + nthread - 1) / nthread,
    (pnsz[2] + nthread - 1) / nthread);

  mul_mat_kernel << <ngrid, nblock >> > (gpddst, gpdsrc1, gpdsrc2, gpnsz);

  cudaMemcpy(pddst, gpddst, pnsz[0] * pnsz[1] * pnsz[2] * sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(gpddst);
  cudaFree(gpdsrc1);
  cudaFree(gpdsrc2);
  cudaFree(gpnsz);

  gpddst = 0;
  gpdsrc1 = 0;
  gpdsrc2 = 0;
  gpnsz = 0;

  return;
}


// DLLEXPORT void add_mat(float *pddst, float *pdsrc1, float *pdsrc2, int *pnsz){ // window only
extern "C" void add_mat(float* pddst, float* pdsrc1, float* pdsrc2, int* pnsz) { // available on window and linux
  float* gpddst = 0;
  float* gpdsrc1 = 0;
  float* gpdsrc2 = 0;
  int* gpnsz = 0;

  cudaMalloc((void**)&gpddst, pnsz[0] * pnsz[1] * pnsz[2] * sizeof(float));
  cudaMalloc((void**)&gpdsrc1, pnsz[0] * pnsz[1] * pnsz[2] * sizeof(float));
  cudaMalloc((void**)&gpdsrc2, pnsz[0] * pnsz[1] * pnsz[2] * sizeof(float));
  cudaMalloc((void**)&gpnsz, 3 * sizeof(int));


  cudaMemset(gpddst, 0, pnsz[0] * pnsz[1] * pnsz[2] * sizeof(float));
  cudaMemset(gpdsrc1, 0, pnsz[0] * pnsz[1] * pnsz[2] * sizeof(float));
  cudaMemset(gpdsrc2, 0, pnsz[0] * pnsz[1] * pnsz[2] * sizeof(float));
  cudaMemset(gpnsz, 0, 3 * sizeof(int));

  cudaMemcpy(gpdsrc1, pdsrc1, pnsz[0] * pnsz[1] * pnsz[2] * sizeof(float), cudaMemcpyHostToDevice); // destination, source, memory size, direction
  cudaMemcpy(gpdsrc2, pdsrc2, pnsz[0] * pnsz[1] * pnsz[2] * sizeof(float), cudaMemcpyHostToDevice); // destination, source, memory size, direction
  cudaMemcpy(gpnsz, pnsz, 3 * sizeof(int), cudaMemcpyHostToDevice); // destination, source, memory size, direction

  int nthread = 8;
  dim3 nblock(nthread, nthread, nthread);
  dim3 ngrid((pnsz[0] + nthread - 1) / nthread,
    (pnsz[1] + nthread - 1) / nthread,
    (pnsz[2] + nthread - 1) / nthread);

  add_mat_kernel << <ngrid, nblock >> > (gpddst, gpdsrc1, gpdsrc2, gpnsz);

  cudaMemcpy(pddst, gpddst, pnsz[0] * pnsz[1] * pnsz[2] * sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(gpddst);
  cudaFree(gpdsrc1);
  cudaFree(gpdsrc2);
  cudaFree(gpnsz);

  gpddst = 0;
  gpdsrc1 = 0;
  gpdsrc2 = 0;
  gpnsz = 0;

  return;
}


//
//
//int main()
//{
//  const int arraySize = 5;
//  const int a[arraySize] = { 1, 2, 3, 4, 5 };
//  const int b[arraySize] = { 10, 20, 30, 40, 50 };
//  int c[arraySize] = { 0 };
//  int* dev_a, * dev_b, * dev_c;  // GPU 메모리 포인터
//
//  // GPU 메모리 할당
//  cudaMalloc((void**)&dev_a, arraySize * sizeof(int));
//  cudaMalloc((void**)&dev_b, arraySize * sizeof(int));
//  cudaMalloc((void**)&dev_c, arraySize * sizeof(int));
//
//  // CPU에서 GPU로 데이터 복사
//  cudaMemcpy(dev_a, a, arraySize * sizeof(int), cudaMemcpyHostToDevice);
//  cudaMemcpy(dev_b, b, arraySize * sizeof(int), cudaMemcpyHostToDevice);
//
//  // CUDA 커널 실행
//  addArrays <<<1, arraySize >>> (dev_c, dev_a, dev_b, arraySize);
//
//  // GPU에서 CPU로 결과 복사
//  cudaMemcpy(c, dev_c, arraySize * sizeof(int), cudaMemcpyDeviceToHost);
//
//  // GPU 메모리 해제
//  cudaFree(dev_a);
//  cudaFree(dev_b);
//  cudaFree(dev_c);
//
//  printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
//    c[0], c[1], c[2], c[3], c[4]);
//
//  return 0;
//}
//
//__global__ void addArrays(int* c, const int* a, const int* b, unsigned int size)
//{
//  int i = threadIdx.x;
//  c[i] = a[i] + b[i];
//}