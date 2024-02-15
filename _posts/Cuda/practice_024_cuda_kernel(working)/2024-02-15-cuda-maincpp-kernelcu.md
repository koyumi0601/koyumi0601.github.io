---
layout: single
title: "CUDA example code 1. main.cpp, kernel.cu, kernel.cuh 구성으로 빌드하기, ubuntu"
categories: cuda
tags: [language, programming, cpp, cuda]
toc: true
author_profile: false
sidebar:
    nav: "docs"
search: true

---

*CUDA sample code*

# 참조 코드 경로
- practice_024_cuda_kernel(working)
- practice_025_cuda_kernel_main_cpp_kernel_cu(working)

# How to start
- cuda 설치 (CUDA Toolkit)

# 프로젝트 폴더
## main.cu만 사용할 때

```
├── build (folder)
├── CMakeLists.txt
├── howToStart.md
└── main.cu
```

- 프로젝트 폴더 생성
- CMakeLists.txt 생성

```bash
cmake_minimum_required(VERSION 3.10)
project(cuda_project)
find_package(CUDA REQUIRED)
cuda_add_executable(my_program my_cuda_code.cu)
```

- 빌드

```bash
mkdir build
cd build
cmake ..
make
```

- CUDA 컴파일러 (NVCC)가 gcc 버전 8 이상을 지원하지 않으므로 현재 설치된 9.4.0에서 문제가 발생하였음 -> gcc 8 버전을 다시 설치

```bash
sudo apt update
sudo apt install gcc-8 g++-8
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-8 100 --slave /usr/bin/g++ g++ /usr/bin/g++-8
```

- 실행

```bash
./my_program
```

## main.cpp, kernel.cu, kernel.cuh 일 때

- main.cpp에서는 <<< >>>와 같은 cuda 표현을 컴파일러가 인식하지 못한다. 따라서 함수처럼 써줘야하고 필요한 인자는 모두 넘겨줘야 한다
- 예를 들어 기존의 main.cu (하나의 파일에 작성)에서는 아래와 같이 커널을 선언한 후

```c
__global__ void calculateAverage(float *inputData, float *outputData) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    int m = blockIdx.y * blockDim.y + threadIdx.y;

    if (n < N && m < M) {
        float sum = 0.0f;
        for (int l = 0; l < L; ++l) {
            sum += inputData[(n * M * L) + (m * L) + l]; // l 방향으로의 값 더하기
        }
        outputData[(n * M) + m] = sum / L; // 평균 값 계산
    }
}

```

- 함수 실행에 <<< >>> 표현을 사용하였으나

```c
// CUDA 커널 실행
    calculateAverage<<<numBlocks, blockSize>>>(d_inputData, d_outputData);
```

- main.cpp로 작성하였을 때에는 함수 실행에 () 표현을 사용하여야 하며

```cpp
// CUDA 함수 실행
calculateAverage(N, M, L, numBlocks, blockSize, d_inputData, d_outputData);
```

- kernel.cu에 interface를 알려주고, N, M, L도 넘겨줬다.. -> 이건 다른 방법을 찾아볼 것.

```c
#include "kernel.cuh"

__global__ void calculateAverageKernel(int N, int M, int L, float *inputData, float *outputData) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    int m = blockIdx.y * blockDim.y + threadIdx.y;

    if (n < N && m < M) {
        float sum = 0.0f;
        for (int l = 0; l < L; ++l) {
            sum += inputData[(n * M * L) + (m * L) + l]; // l 방향으로의 값 더하기
        }
        outputData[(n * M) + m] = sum / L; // 평균 값 계산
    }
}

void calculateAverage(int N, int M, int L, dim3 numBlocks, dim3 blockSize, float *inputData, float *outputData) {
    calculateAverageKernel<<<numBlocks, blockSize>>>(N, M, L, inputData, outputData);
}
```


## 커맨드 라인에서 각각 컴파일 및 링크하여 빌드할 수도 있다.

```bash
nvcc -c main.cu -o main.o # 각각 빌드
nvcc -c kernel.cu -o kernel.o
nvcc main.o kernel.o -o my_program # 링크
./my_program # 실행
```