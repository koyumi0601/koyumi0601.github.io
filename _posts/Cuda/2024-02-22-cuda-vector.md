---
layout: single
title: "cuda <vector> 우회하여 사용하기"
categories: cuda
tags: [language, programming, cpp, cuda]
toc: true
author_profile: false
sidebar:
    nav: "docs"
search: true

---

*cuda <vector> 우회하여 사용하기*


# 문제의 코드

```cpp
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

  cudaStatus = cudaMalloc(&deviceOutputPlane, outputPlane.size() * sizeof(unsigned char)); // Allocate GPU buffer
  cudaStatus = cudaMalloc(&deviceVecVolSlices, sizeof(unsigned char*) * dim3Size); // Allocate memory for device-side pointer array
  for (int i = 0; i < dim3Size; ++i) 
  {
    std::vector<unsigned char> vecVolSlice(vecVol.begin() + i * dim1Size * dim2Size, vecVol.begin() + (i + 1) * dim1Size * dim2Size);
    cudaStatus = cudaMalloc(&deviceVecVolSlicesHost[i], vecVolSlice.size() * sizeof(unsigned char));
    cudaStatus = cudaMemcpy(deviceVecVolSlicesHost[i], vecVolSlice.data(), vecVolSlice.size() * sizeof(unsigned char), cudaMemcpyHostToDevice);
  }
  cudaStatus = cudaMemcpy(deviceVecVolSlices, deviceVecVolSlicesHost.data(), sizeof(unsigned char*) * dim3Size, cudaMemcpyHostToDevice); // Copy pointer array from host to device
  // Launch kernel
  dim3 threadsPerBlock(256);
  dim3 numBlocks((outputPlane.size() + threadsPerBlock.x - 1) / threadsPerBlock.x);
  averageVectorForKernels << <numBlocks, threadsPerBlock >> > (deviceOutputPlane, deviceVecVolSlices, dim1Size * dim2Size, dim3Size);
  cudaStatus = cudaGetLastError(); // Check for kernel launch errors
  cudaStatus = cudaDeviceSynchronize(); // Synchronize kernel and check for errors
  outputPlane.resize(dim1Size * dim2Size); // Copy output vector from GPU buffer to host memory
  cudaStatus = cudaMemcpy(outputPlane.data(), deviceOutputPlane, dim1Size * dim2Size * sizeof(unsigned char), cudaMemcpyDeviceToHost);

Error:
  cudaFree(deviceOutputPlane);
  for (int i = 0; i < dim3Size; ++i) {
    cudaFree(deviceVecVolSlicesHost[i]);
  }
  cudaFree(deviceVecVolSlices);
  return cudaStatus;
}
```

- std::vector의 데이터를 디바이스 메모리로 직접 전달할 수 없는 문제

# 해결 방법

- std::vector<unsigned char*>에서 포인터들을 추출:

```cpp
std::vector<unsigned char*> deviceVecVolSlicesHost(dim3Size, nullptr);
```

- 여기서 deviceVecVolSlicesHost는 각 볼륨 슬라이스에 대한 디바이스 메모리 포인터를 저장하기 위한 std::vector입니다. 각 슬라이스는 별도의 cudaMalloc 호출을 통해 디바이스 메모리에 할당됩니다.

- 이 배열을 CUDA 메모리에 복사:

```cpp
for (int i = 0; i < dim3Size; ++i) {
    // ...
    cudaStatus = cudaMemcpy(deviceVecVolSlicesHost[i], vecVolSlice.data(), vecVolSlice.size() * sizeof(unsigned char), cudaMemcpyHostToDevice);
    // ...
}
```

- 각 볼륨 슬라이스의 데이터를 호스트에서 디바이스로 복사합니다. vecVolSlice는 vecVol에서 추출한 하나의 슬라이스를 나타내며, 이 슬라이스의 데이터는 cudaMemcpy을 사용하여 해당 디바이스 메모리 포인터(deviceVecVolSlicesHost[i])로 복사됩니다.

- CUDA 커널을 호출할 때 이 배열을 인자로 전달:

```cpp
unsigned char** deviceVecVolSlices;
cudaStatus = cudaMalloc(&deviceVecVolSlices, sizeof(unsigned char*) * dim3Size);
// ...
cudaStatus = cudaMemcpy(deviceVecVolSlices, deviceVecVolSlicesHost.data(), sizeof(unsigned char*) * dim3Size, cudaMemcpyHostToDevice);
// ...
averageVectorNKernels<<<numBlocks, threadsPerBlock>>>(deviceOutputPlane, deviceVecVolSlices, dim1Size * dim2Size, dim3Size);
```

- deviceVecVolSlices는 디바이스 메모리에 할당된 포인터 배열로, 각 볼륨 슬라이스의 디바이스 메모리 포인터를 저장합니다. 이 포인터 배열은 cudaMalloc을 통해 디바이스 메모리에 할당되며, deviceVecVolSlicesHost에서 추출한 포인터들을 cudaMemcpy을 사용하여 이 배열로 복사합니다.


- 마지막으로, averageVectorNKernels 커널은 이 포인터 배열을 인자로 받아, 각 포인터를 통해 볼륨 슬라이스에 접근하고 연산을 수행합니다.