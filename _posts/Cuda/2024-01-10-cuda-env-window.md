---
layout: single
title: "CUDA Environment setup, Window"
categories: cuda
tags: [language, programming, cpp, cuda, window]
toc: true
author_profile: false
sidebar:
    nav: "docs"
search: true

---

*CUDA Environment Setup, window*

- 참고 
    - [https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html)
    - [https://docs.nvidia.com/cuda/cuda-quick-start-guide/index.html#windows](https://docs.nvidia.com/cuda/cuda-quick-start-guide/index.html#windows)


- Quick Install Guide
  - sample code [https://github.com/NVIDIA/cuda-samples/tree/master/Samples/5_Domain_Specific/nbody](https://github.com/NVIDIA/cuda-samples/tree/master/Samples/5_Domain_Specific/nbody)

    - sample code는 빌드 시 잘 동작.

- visual studio에서 project 생성 시, cuda toolkit 버전에 맞는 cuda runtim3 12.3 같은 템플릿을 선택해서 생성.



- main(entry point)가 없다면, 빌드(컴파일 및 실행)이 불가능. 컴파일까지는 가능. .obj 파일 생성 됨.
- python 파일 추가 시, 솔루션 탐색기 > 우클릭 > 추가 > 기존항목추가 > 터미널에서 실행 가능.



1.1 System Requirements
- NVIDIA GPU
    - nvidia-smi

```bash
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 546.33                 Driver Version: 546.33       CUDA Version: 12.3     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                     TCC/WDDM  | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  NVIDIA GeForce RTX 3060      WDDM  | 00000000:2B:00.0  On |                  N/A |
|  0%   34C    P8              21W / 170W |    505MiB / 12288MiB |      4%      Default |
|                                         |                      |                  N/A |
+-----------------------------------------+----------------------+----------------------+

+---------------------------------------------------------------------------------------+
| Processes:                                                                            |
|  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
|        ID   ID                                                             Usage      |
|=======================================================================================|
|    0   N/A  N/A       832    C+G   ...64__8wekyb3d8bbwe\CalculatorApp.exe    N/A      |
|    0   N/A  N/A      1304    C+G   C:\Windows\System32\dwm.exe               N/A      |
|    0   N/A  N/A      3004    C+G   ...siveControlPanel\SystemSettings.exe    N/A      |
|    0   N/A  N/A      4232    C+G   ...Programs\Microsoft VS Code\Code.exe    N/A      |
|    0   N/A  N/A      4352    C+G   C:\Windows\explorer.exe                   N/A      |
|    0   N/A  N/A      6100    C+G   ...2txyewy\StartMenuExperienceHost.exe    N/A      |
|    0   N/A  N/A      8108    C+G   ....Search_cw5n1h2txyewy\SearchApp.exe    N/A      |
|    0   N/A  N/A     10272    C+G   ...ekyb3d8bbwe\PhoneExperienceHost.exe    N/A      |
|    0   N/A  N/A     12312    C+G   ...oogle\Chrome\Application\chrome.exe    N/A      |
|    0   N/A  N/A     12536    C+G   ...CBS_cw5n1h2txyewy\TextInputHost.exe    N/A      |
|    0   N/A  N/A     13808    C+G   ...Desktop\app-3.3.6\GitHubDesktop.exe    N/A      |
+---------------------------------------------------------------------------------------+
```


- A supported version of Linux with a gcc compiler and toolchain
- NVIDIA CUDA Toolkit (available at https://developer.nvidia.com/cuda-downloads)
    - install 
      - 임시 압축 해제 경로, 직접 지정 D:\Users\user\AppData\Local\Temp\cuda
      - 환경변수 자동 추가 됨. C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.3
    - nvcc --version



```bash
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2023 NVIDIA Corporation
Built on Fri_Nov__3_17:51:05_Pacific_Daylight_Time_2023
Cuda compilation tools, release 12.3, V12.3.103        
Build cuda_12.3.r12.3/compiler.33492891_0
```


- Microsoft Windows 10, 11 21H2, 22H2-SV2, Server 2022, 2019
    - window 10


- Compiler MSVC Version 193x / IDE Visual Studio 2022 17.0 / 64 bit for CUDA 12.0 and later toolkit / 32 bit로 컴파일하려면, cuda 12.0 이전 버전을 사용할 것.
    - IDE: visual studio 2022


- 








# CPP 컴파일 환경 설정
- 프로젝트 디렉토리 설정
    - 프로젝트 디렉토리를 VS Code에서 연다
- 작업 환경
    - tasks.json: 빌드 작업 설정. 컴파일러 명령 및 링크 설정
    - c_cpp_properties.json: C++ 프로젝트의 intellisense 구성 설정. 컴파일러 경로와 표준 라이브러리 경로
    - .vscode/settings.json: 프로젝트 또는 사용자 설정 지정. 경로 관련 설정 추가 가능 



# CU 컴파일 환경 설정

- nvcc 설치

```bash
sudo apt-get update
sudo apt-get install nvidia-cuda-toolkit
// 확인
nvcc --version
nvidia-smi
```

- cpp 컴파일 환경 설정과 마찬가지로, 프로젝트 디렉토리에 .vccode 폴더 생성 후 tasks.json 파일을 만든다.

```
// tasks.json
{
    "version": "2.0.0",
    "tasks": [
        {
            "label": "build",
            "type": "shell",
            "command": "nvcc",
            "args": [
                "main.cu",          // .cu 파일 이름
                "-o",
                "my_cuda_program"  // 생성된 실행 파일 이름
            ],
            "group": {
                "kind": "build",
                "isDefault": true
            },
            "presentation": {
                "reveal": "always",
                "panel": "shared"
            },
            "problemMatcher": "$nvcc"
        }
    ]
}
```

- 컴파일

```
cntrl + shift + B
```

- 실행

```bash
./my_cuda_program
```

- 테스트 코드

```c
#include <iostream>
#include <cuda_runtime.h>

int main() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for (int deviceIdx = 0; deviceIdx < deviceCount; ++deviceIdx) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, deviceIdx);

        std::cout << "Device " << deviceIdx << ": " << deviceProp.name << std::endl;
        std::cout << "  Compute Capability: " << deviceProp.major << "." << deviceProp.minor << std::endl;
        std::cout << "  Multiprocessors: " << deviceProp.multiProcessorCount << std::endl;
        std::cout << "  CUDA Cores per Multiprocessor: " << deviceProp.warpSize << std::endl;
        std::cout << "  Global Memory: " << deviceProp.totalGlobalMem / (1024 * 1024) << " MB" << std::endl;
        std::cout << "  Shared Memory per Block: " << deviceProp.sharedMemPerBlock / 1024 << " KB" << std::endl;
        std::cout << "  Max Threads per Block: " << deviceProp.maxThreadsPerBlock << std::endl;
        std::cout << "  Max Threads per Dimension: (" << deviceProp.maxThreadsDim[0] << ", "
                  << deviceProp.maxThreadsDim[1] << ", " << deviceProp.maxThreadsDim[2] << ")" << std::endl;
        std::cout << "  Max Grid Size: (" << deviceProp.maxGridSize[0] << ", " << deviceProp.maxGridSize[1] << ", "
                  << deviceProp.maxGridSize[2] << ")" << std::endl;
        std::cout << std::endl;
    }

    return 0;
}
```

```c
Device 0: NVIDIA GeForce GTX 960
  Compute Capability: 5.2
  Multiprocessors: 8
  CUDA Cores per Multiprocessor: 32
  Global Memory: 4043 MB
  Shared Memory per Block: 48 KB
  Max Threads per Block: 1024
  Max Threads per Dimension: (1024, 1024, 64)
  Max Grid Size: (2147483647, 65535, 65535)
  Tensor Core 지원: No
```

- Ray tracing, 메모리 버스 너비, 클럭 속도, AI 및 machine learning 지원 여부 확인 요망