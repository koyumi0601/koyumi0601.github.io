#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "kernel.cuh"
#include <numeric>
#include <iostream>


__global__ void averageVectorForKernels(unsigned char* deviceOutputPlane, unsigned char** deviceVecVolSlices, int numElements, int numVectors)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < numElements) {
    unsigned int sum = 0;
    for (int vec = 0; vec < numVectors; ++vec) {
      sum += deviceVecVolSlices[vec][i];
    }
    deviceOutputPlane[i] = sum / numVectors;
  }
}


cudaError_t averageVectorForWithCuda(std::vector<unsigned char>& outputPlane, std::vector<unsigned char>& vecVol, unsigned int dim1Size, unsigned int dim2Size, unsigned int dim3Size) {
  unsigned char* deviceOutputPlane;
  std::vector<unsigned char*> deviceVecVolSlicesHost(dim3Size, nullptr); // Host-side vector slice pointers
  unsigned char** deviceVecVolSlices; // Device-side pointer array
  cudaError_t cudaStatus;

  // Select GPU
  cudaStatus = cudaSetDevice(0);
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaSetDevice failed! Do you have a CUDA-capable GPU installed?");
    goto Error;
  }

  // Allocate GPU buffer
  cudaStatus = cudaMalloc(&deviceOutputPlane, outputPlane.size() * sizeof(unsigned char));
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaMalloc failed for deviceOutputPlane!");
    goto Error;
  }

  // Allocate memory for device-side pointer array
  cudaStatus = cudaMalloc(&deviceVecVolSlices, sizeof(unsigned char*) * dim3Size);
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaMalloc failed for deviceVecVolSlices!");
    goto Error;
  }

  for (int i = 0; i < dim3Size; ++i) {
    std::vector<unsigned char> vecVolSlice(vecVol.begin() + i * dim1Size * dim2Size, vecVol.begin() + (i + 1) * dim1Size * dim2Size);
    cudaStatus = cudaMalloc(&deviceVecVolSlicesHost[i], vecVolSlice.size() * sizeof(unsigned char));
    if (cudaStatus != cudaSuccess) {
      fprintf(stderr, "cudaMalloc failed for deviceVecVolSlicesHost[%d]!", i);
      goto Error;
    }
    cudaStatus = cudaMemcpy(deviceVecVolSlicesHost[i], vecVolSlice.data(), vecVolSlice.size() * sizeof(unsigned char), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
      fprintf(stderr, "cudaMemcpy failed at host to device for slice %d!", i);
      goto Error;
    }
  }

  // Copy pointer array from host to device
  cudaStatus = cudaMemcpy(deviceVecVolSlices, deviceVecVolSlicesHost.data(), sizeof(unsigned char*) * dim3Size, cudaMemcpyHostToDevice);
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaMemcpy failed for deviceVecVolSlices to device!");
    goto Error;
  }

  // Launch kernel
  dim3 threadsPerBlock(256);
  dim3 numBlocks((outputPlane.size() + threadsPerBlock.x - 1) / threadsPerBlock.x);
  averageVectorForKernels << <numBlocks, threadsPerBlock >> > (deviceOutputPlane, deviceVecVolSlices, dim1Size * dim2Size, dim3Size);

  // Check for kernel launch errors
  cudaStatus = cudaGetLastError();
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "averageVectorNKernels launch failed: %s\n", cudaGetErrorString(cudaStatus));
    goto Error;
  }

  // Synchronize kernel and check for errors
  cudaStatus = cudaDeviceSynchronize();
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching averageVectorNKernels!\n", cudaStatus);
    goto Error;
  }

  // Copy output vector from GPU buffer to host memory
  outputPlane.resize(dim1Size * dim2Size);
  cudaStatus = cudaMemcpy(outputPlane.data(), deviceOutputPlane, dim1Size * dim2Size * sizeof(unsigned char), cudaMemcpyDeviceToHost);
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaMemcpy failed for deviceOutputPlane to host!");
    goto Error;
  }

Error:
  cudaFree(deviceOutputPlane);
  for (int i = 0; i < dim3Size; ++i) {
    cudaFree(deviceVecVolSlicesHost[i]);
  }
  cudaFree(deviceVecVolSlices);
  return cudaStatus;
}