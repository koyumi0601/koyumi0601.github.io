
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <cuda.h>
// #define DLLEXPORT extern "C" __declspec(dllexport) // window


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

  //mul_const_kernel <<<ngrid, nblock >>> (gpddst, gpdsrc, dconst, gpnsz);
  mul_const_kernel << <ngrid, nblock >> > (gpddst, gpdsrc, dconst, gpnsz);

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




// nvcc -Xcompiler -fPIC math_cu.cu -shared -o libmath_cu.so // linux
// nvcc -o math_cu_win.dll math_cu_linux.cu --shared // window
// nvcc error   : 'cudafe++' died with status 0xC0000005 (ACCESS_VIOLATION)



//
//cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);
//
//__global__ void addKernel(int *c, const int *a, const int *b)
//{
//    int i = threadIdx.x;
//    c[i] = a[i] + b[i];
//}
//
//int main()
//{
//    const int arraySize = 5;
//    const int a[arraySize] = { 1, 2, 3, 4, 5 };
//    const int b[arraySize] = { 10, 20, 30, 40, 50 };
//    int c[arraySize] = { 0 };
//
//    // Add vectors in parallel.
//    cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "addWithCuda failed!");
//        return 1;
//    }
//
//    printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
//        c[0], c[1], c[2], c[3], c[4]);
//
//    // cudaDeviceReset must be called before exiting in order for profiling and
//    // tracing tools such as Nsight and Visual Profiler to show complete traces.
//    cudaStatus = cudaDeviceReset();
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaDeviceReset failed!");
//        return 1;
//    }
//
//    return 0;
//}
//
//// Helper function for using CUDA to add vectors in parallel.
//cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
//{
//    int *dev_a = 0;
//    int *dev_b = 0;
//    int *dev_c = 0;
//    cudaError_t cudaStatus;
//
//    // Choose which GPU to run on, change this on a multi-GPU system.
//    cudaStatus = cudaSetDevice(0);
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
//        goto Error;
//    }
//
//    // Allocate GPU buffers for three vectors (two input, one output)    .
//    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMalloc failed!");
//        goto Error;
//    }
//
//    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMalloc failed!");
//        goto Error;
//    }
//
//    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMalloc failed!");
//        goto Error;
//    }
//
//    // Copy input vectors from host memory to GPU buffers.
//    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMemcpy failed!");
//        goto Error;
//    }
//
//    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMemcpy failed!");
//        goto Error;
//    }
//
//    // Launch a kernel on the GPU with one thread for each element.
//    addKernel<<<1, size>>>(dev_c, dev_a, dev_b);
//
//    // Check for any errors launching the kernel
//    cudaStatus = cudaGetLastError();
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
//        goto Error;
//    }
//    
//    // cudaDeviceSynchronize waits for the kernel to finish, and returns
//    // any errors encountered during the launch.
//    cudaStatus = cudaDeviceSynchronize();
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
//        goto Error;
//    }
//
//    // Copy output vector from GPU buffer to host memory.
//    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMemcpy failed!");
//        goto Error;
//    }
//
//Error:
//    cudaFree(dev_c);
//    cudaFree(dev_a);
//    cudaFree(dev_b);
//    
//    return cudaStatus;
//}
