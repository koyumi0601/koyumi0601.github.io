---
layout: single
title: "CUDA Study Materials"
categories: cuda
tags: [language, programming, cpp, cuda]
toc: true
author_profile: false
sidebar:
    nav: "docs"
search: true

---

*CUDA Study Materials, Pdf, lecture, spec., sample code*



# Cuda c programming guide

- **Programming Guide** [https://docs.nvidia.com/cuda/cuda-c-programming-guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide)
- Quick Start Guide [https://docs.nvidia.com/cuda/cuda-quick-start-guide/](https://docs.nvidia.com/cuda/cuda-quick-start-guide/)
- CUDA Programming read docs. [https://cuda.readthedocs.io/ko/latest/rtx2080Ti/](https://cuda.readthedocs.io/ko/latest/rtx2080Ti/)
- 구글검색 nvidia ampere architecture white paper [https://www.nvidia.com/content/PDF/nvidia-ampere-ga-102-gpu-architecture-whitepaper-v2.1.pdf](https://www.nvidia.com/content/PDF/nvidia-ampere-ga-102-gpu-architecture-whitepaper-v2.1.pdf)


# Code 
- Samples: NVIDIA Github [https://github.com/NVIDIA/cuda-samples](https://github.com/NVIDIA/cuda-samples)
- 자주 쓰는 코드: https://blog.naver.com/PostView.nhn?blogId=lithium81&logNo=80143506571


# Compute Capability

- maxwell 5.x (https://docs.nvidia.com/cuda/archive/12.2.1/maxwell-compatibility-guide/index.html

# Specifications
- GTX 960
- RTX 3060 [https://www.nvidia.com/en-us/data-center/ampere-architecture/](https://www.nvidia.com/en-us/data-center/ampere-architecture/)
- Geforce 550 TI
```bash
  **CL_DEVICE_COMPUTE_CAPABILITY_NV: 2.1
  NUMBER OF MULTIPROCESSORS: 4** >> SM 4개
  **NUMBER OF CUDA CORES: 192** >> 전체 SP 192개. 하나의 SM당 SP는 192/4 = 48개. 제품에 써있는 스펙-멀티프로세서 숫자는 쿠다코어임.
  CL_DEVICE_REGISTERS_PER_BLOCK_NV: 32768 >> 레지스터 32768개
  CL_DEVICE_WARP_SIZE_NV: 32
  CL_DEVICE_GPU_OVERLAP_NV: CL_TRUE
  CL_DEVICE_KERNEL_EXEC_TIMEOUT_NV: CL_FALSE
  CL_DEVICE_INTEGRATED_MEMORY_NV: CL_FALSE
  CL_DEVICE_PREFERRED_VECTOR_WIDTH_<t> CHAR 1, SHORT 1, INT 1, LONG 1, FLOAT 1, DOUBLE 1
  Physical Limits for GPU Compute Capability:	2.0 # 최대 한계란 뜻
  Threads per Warp	32
  Warps per Multiprocessor	48
  Threads per Multiprocessor	1536
  Thread Blocks per Multiprocessor	8
  Total # of 32-bit registers per Multiprocessor	32768
  Register allocation unit size	64
  Register allocation granularity	warp
  Registers per Thread	63
  Shared Memory per Multiprocessor (bytes)	49152 # Shared Memory 49152 bytes
  Shared Memory Allocation unit size	128
  Warp allocation granularity	2
  Maximum Thread Block Size	1024
```

## 구조

# 6.1.1 스트리밍 멀티프로세서
- 페르미 아키텍처: 하나의 SM에 32개의 CUDA core를 가지고 있다

## Instruction Cache
- 명령어 1개가 워프 내 모든 스레드에 사용된다. SIMT (Single Instruction Multi Treads)

## Warp Scheduler
- 다음에 처리할 명령어를 결정하거나 명령을 내린다
## Dispatch Unit
- 명령어 전달 유닛


## **Register File**
- 32768 x 32 bit
- CORE들은 레지스터를 등분해서 가지고 있는다. 자신만의 작업책상이 있다.
- 따라서 문맥 교환 비용이 0이다.

## **CUDA Core**
- 간단히는, cuda core하나가 스레드 하나를 처리한다고 볼 수 있다
### Dispatch port
### Operand Collector
### FP Unit
- 실제 연산 유닛
### Int Unit
- 실제 연산 유닛
### Result Queue
## LD/ST
## SFU
## Interconnect Network
## 64kB **shared memory** / **L1 cache**

## Uniform cache

# 6.2 CUDA 스레드 계층과 GPU 하드웨어
- 스레드 -> 워프 -> 스레드 블록 -> 그리드 -> GPU
- 그리드 1개 = GPU 1개.
  - 멀티 GPU라고 하더라도 그리드 1개는 GPU 1개에 대응된다.
  - 반면, GPU 내에서 여러 개의 그리드는 실행 가능하다.

- https://www.youtube.com/watch?v=gSgZNdT9414 33:54

| 항목            | CUDA               | HW               |
|-----------------|--------------------|------------------|
| 연산 단위       | Thread             | SP or CUDA Core  |
| HW 점유 단위    | Block              | SM               |
| Shared Memory  | 48KB `__shared__`  | L1 Cache (16KB + 4KB) |
| Barrier        | `__syncthreads()`  | -                |
| 언어            | C/C++, ptr 사용 가능 | -                |


![img](https://t1.daumcdn.net/cfile/tistory/16282136509A061507)




# Terminology

- Streaming Processor (SP): SP는 한번에 4개의 스레드를 실행. 실제로는 1개 스레드당 1클럭, 시분할 방식으로 4클럭에 걸쳐 수행하지만, 4클럭을 1사이클로 보고, 사이클 단위로 처리량을 논하기 때문에 한 사이클에 4개 스레드를 실행한다고 함. 한 사이클에 몇 개의 스레드를 처리할지는 VGA HW마다 혹은 몇 클럭을 한 사이클로 보느냐에 따라 각기 다르다. 페르미(2.x) 아키텍쳐의 경우 2번의 클럭으로 2개의 스레드를 실행한다고 함.
- Streaming Multiprocessor (SM): 
  - SM은 한번에 32개의 스레드를 실행함. (1 warp = 32 thread)
  - G80의 경우 SM이 8개의 SP(Cuda core, thread)를 가지므로 8*4=32. 
  - GF100은 SM1개에 SP가 32개 있으므로,  한 번에 32*4=128개의 스레드를 실행한다.
  - warp scheduler가 4개인 경우 (ampere architecture 7.x) SM은 4 warp를 한번에 돌릴 수 있다.
- Thread
- Warp: 
  - 32개의 스레드를 묶어서 워프라고 함.
  - G80, GT200등의 SM은 한번에 1개의 워프를 실행 함. 즉, 32 threads = 1 warp
  - 페르미 아키텍처는 SM안에 워프 스케줄러가 1개에서 2개로 늘어났고, SP가 한 사이클에 2개의 스레드를 처리하며, SM은 32개 SP를 가지므로, 하나의 SM이 한 번에 2개의 워프를 실행 함. 
    - 2 * 32 = 64 threads = 2 warp
- Block
  - SM과 블록: 하나의 SM은 하나의 블록에 대응한다. compute capability 1.3까지는 블록 한 개가 최대 512개의 스레드를 가질 수 있으므로, SM은 한 번에 32개 스레드씩, 16번의 context switching을 통해 512개 스레드를 실행한다.
  - 스레드와 블록: Compute capability에 따라 다른데, 1.0~1.3까지는 한 블록당 최대 512개, 2.x는 최대 1024개의 스레드를 가진다. 블록은 3차원으로 구성이 가능한데, x * y * z의 값은 한 블록당 최대 허용 스레드 개수보다 작아야 한다.
- Grid: GPU
- Stream




# Optimization
- Occupancy 100%로 만드는 방법 
  - [https://ccode.tistory.com/184](https://ccode.tistory.com/184)
- 참고.
  - [http://developer.download.nvidia.com/CUDA/training/NVIDIA_GPU_Computing_Webinars_Further_CUDA_Optimization.pdf](http://developer.download.nvidia.com/CUDA/training/NVIDIA_GPU_Computing_Webinars_Further_CUDA_Optimization.pdf)
  - [http://www.nvidia.com/content/PDF/sc_2010/CUDA_Tutorial/SC10_Fundamental_Optimizations.pdf](http://www.nvidia.com/content/PDF/sc_2010/CUDA_Tutorial/SC10_Fundamental_Optimizations.pdf)
  - [http://developer.download.nvidia.com/CUDA/training/cuda_webinars_WarpsAndOccupancy.pdf](http://developer.download.nvidia.com/CUDA/training/cuda_webinars_WarpsAndOccupancy.pdf)
  - [http://nvidia.fullviewmedia.com/gtc2010/0922-a5-2238.html](http://nvidia.fullviewmedia.com/gtc2010/0922-a5-2238.html)
  - [http://www.cs.berkeley.edu/~volkov/volkov10-GTC.pdf](http://www.cs.berkeley.edu/~volkov/volkov10-GTC.pdf)



# 부동소수점 연산
- IEEE 754 표준의 부동소수점 표현방식
- [31] sign [30-23] exponent [22-0] fraction
## 허용오차 10-5
## float -> double
- 허용 오차와 함께 사용
## API
| Function Name         | Operation                                  | Description                                                         |
|-----------------------|--------------------------------------------|---------------------------------------------------------------------|
| `__fadd_rn(x, y)`     | `x + y` (Float, Round to Nearest)          | 가장 가까운 값으로 라운딩하여 더하기 연산을 수행합니다.                  |
| `__fadd_rz(x, y)`     | `x + y` (Float, Round Towards Zero)        | 0 방향으로 라운딩하여 더하기 연산을 수행합니다.                          |
| `__fadd_ru(x, y)`     | `x + y` (Float, Round Up)                  | 무조건 올림하여 더하기 연산을 수행합니다.                               |
| `__fadd_rd(x, y)`     | `x + y` (Float, Round Down)                | 무조건 내림하여 더하기 연산을 수행합니다.                               |
| `__fmul_rn(x, y)`     | `x * y` (Float, Round to Nearest)          | 가장 가까운 값으로 라운딩하여 곱하기 연산을 수행합니다.                   |
| `__fmul_rz(x, y)`     | `x * y` (Float, Round Towards Zero)        | 0 방향으로 라운딩하여 곱하기 연산을 수행합니다.                          |
| `__fmul_ru(x, y)`     | `x * y` (Float, Round Up)                  | 무조건 올림하여 곱하기 연산을 수행합니다.                                |
| `__fmul_rd(x, y)`     | `x * y` (Float, Round Down)                | 무조건 내림하여 곱하기 연산을 수행합니다.                                |
| `__fmaf_rn(x, y, z)`  | `x * y + z` (Float, Round to Nearest)      | 가장 가까운 값으로 라운딩하여 곱한 후 더하기 연산을 수행합니다.           |
| `__fmaf_rz(x, y, z)`  | `x * y + z` (Float, Round Towards Zero)    | 0 방향으로 라운딩하여 곱한 후 더하기 연산을 수행합니다.                   |
| `__fmaf_ru(x, y, z)`  | `x * y + z` (Float, Round Up)              | 무조건 올림하여 곱한 후 더하기 연산을 수행합니다.                         |
| `__fmaf_rd(x, y, z)`  | `x * y + z` (Float, Round Down)            | 무조건 내림하여 곱한 후 더하기 연산을 수행합니다.                         |
| `__frcp_rn(x)`        | `1/x` (Float, Round to Nearest)            | 가장 가까운 값으로 라운딩하여 역수를 계산합니다.                          |
| `__frcp_rz(x)`        | `1/x` (Float, Round Towards Zero)          | 0 방향으로 라운딩하여 역수를 계산합니다.                                 |
| `__frcp_ru(x)`        | `1/x` (Float, Round Up)                    | 무조건 올림하여 역수를 계산합니다.                                       |
| `__frcp_rd(x)`        | `1/x` (Float, Round Down)                  | 무조건 내림하여 역수를 계산합니다.                                       |
| `__fdiv_rn(x, y)`     | `x / y` (Float, Round to Nearest)          | 가장 가까운 값으로 라운딩하여 나누기 연산을 수행합니다.                    |
| `__fdiv_rz(x, y)`     | `x / y` (Float, Round Towards Zero)        | 0 방향으로 라운딩하여 나누기 연산을 수행합니다.                           |
| `__fdiv_ru(x, y)`     | `x / y` (Float, Round Up)                  | 무조건 올림하여 나누기 연산을 수행합니다.                                 |
| `__fdiv_rd(x, y)`     | `x / y` (Float, Round Down)                | 무조건 내림하여 나누기 연산을 수행합니다.                                 |
| `__fsqrt_rn(x)`       | `sqrt(x)` (Float, Round to Nearest)        | 가장 가까운 값으로 라운딩하여 제곱근을 계산합니다.                         |
| `__fsqrt_rz(x)`       | `sqrt(x)` (Float, Round Towards Zero)      | 0 방향으로 라운딩하여 제곱근을 계산합니다.                                |
| `__fsqrt_ru(x)`       | `sqrt(x)` (Float, Round Up)                | 무조건 올림하여 제곱근을 계산합니다.                                      |
| `__fsqrt_rd(x)`       | `sqrt(x)` (Float, Round Down)              | 무조건 내림하여 제곱근을 계산합니다.                                      |
| `__dadd_rn(x, y)`     | `x + y` (Double, Round to Nearest)         | 더블 정밀도로 가장 가까운 값으로 라운딩하여 더하기 연산을 수행합니다.        |
| `__dadd_rz(x, y)`     | `x + y` (Double, Round Towards Zero)       | 더블 정밀도로 0 방향으로 라운딩하여 더하기 연산을 수행합니다.               |
| `__dadd_ru(x, y)`     | `x + y` (Double, Round Up)                 | 더블 정밀도로 무조건 올림하여 더하기 연산을 수행합니다.                     |
| `__dadd_rd(x, y)`     | `x + y` (Double, Round Down)               | 더블 정밀도로 무조건 내림하여 더하기 연산을 수행합니다.                     |
| `__dmul_rn(x, y)`     | `x * y` (Double, Round to Nearest)         | 더블 정밀도로 가장 가까운 값으로 라운딩하여 곱하기 연산을 수행합니다.        |
| `__dmul_rz(x, y)`     | `x * y` (Double, Round Towards Zero)       | 더블 정밀도로 0 방향으로 라운딩하여 곱하기 연산을 수행합니다.               |
| `__dmul_ru(x, y)`     | `x * y` (Double, Round Up)                 | 더블 정밀도로 무조건 올림하여 곱하기 연산을 수행합니다.                     |
| `__dmul_rd(x, y)`     | `x * y` (Double, Round Down)               | 더블 정밀도로 무조건 내림하여 곱하기 연산을 수행합니다.                     |
| `__fma_rn(x, y, z)`   | `x * y + z` (Double, Round to Nearest)     | 더블 정밀도로 가장 가까운 값으로 라운딩하여 곱한 후 더하기 연산을 수행합니다. |
| `__fma_rz(x, y, z)`   | `x * y + z` (Double, Round Towards Zero)   | 더블 정밀도로 0 방향으로 라운딩하여 곱한 후 더하기 연산을 수행합니다.        |
| `__fma_ru(x, y, z)`   | `x * y + z` (Double, Round Up)             | 더블 정밀도로 무조건 올림하여 곱한 후 더하기 연산을 수행합니다.              |
| `__fma_rd(x, y, z)`   | `x * y + z` (Double, Round Down)           | 더블 정밀도로 무조건 내림하여 곱한 후 더하기 연산을 수행합니다.              |
| `__drcp_rn(x)`        | `1/x` (Double, Round to Nearest)           | 더블 정밀도로 가장 가까운 값으로 라운딩하여 역수를 계산합니다.               |
| `__drcp_rz(x)`        | `1/x` (Double, Round Towards Zero)         | 더블 정밀도로 0 방향으로 라운딩하여 역수를 계산합니다.                      |
| `__drcp_ru(x)`        | `1/x` (Double, Round Up)                   | 더블 정밀도로 무조건 올림하여 역수를 계산합니다.                            |
| `__drcp_rd(x)`        | `1/x` (Double, Round Down)                 | 더블 정밀도로 무조건 내림하여 역수를 계산합니다.                            |
| `__ddiv_rn(x, y)`     | `x / y` (Double, Round to Nearest)         | 더블 정밀도로 가장 가까운 값으로 라운딩하여 나누기 연산을 수행합니다.         |
| `__ddiv_rz(x, y)`     | `x / y` (Double, Round Towards Zero)       | 더블 정밀도로 0 방향으로 라운딩하여 나누기 연산을 수행합니다.                |
| `__ddiv_ru(x, y)`     | `x / y` (Double, Round Up)                 | 더블 정밀도로 무조건 올림하여 나누기 연산을 수행합니다.                      |
| `__ddiv_rd(x, y)`     | `x / y` (Double, Round Down)               | 더블 정밀도로 무조건 내림하여 나누기 연산을 수행합니다.                      |
| `__dsqrt_rn(x)`       | `sqrt(x)` (Double, Round to Nearest)       | 더블 정밀도로 가장 가까운 값으로 라운딩하여 제곱근을 계산합니다.              |
| `__dsqrt_rz(x)`       | `sqrt(x)` (Double, Round Towards Zero)     | 더블 정밀도로 0 방향으로 라운딩하여 제곱근을 계산합니다.                     |
| `__dsqrt_ru(x)`       | `sqrt(x)` (Double, Round Up)               | 더블 정밀도로 무조건 올림하여 제곱근을 계산합니다.                           |
| `__dsqrt_rd(x)`       | `sqrt(x)` (Double, Round Down)             | 더블 정밀도로 무조건 내림하여 제곱근을 계산합니다.                           |