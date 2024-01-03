import pycuda.driver as cuda
import pycuda.autoinit
from pycuda import gpuarray
from pycuda.compiler import SourceModule



# CUDA 드라이버 초기화
cuda.init()

# 사용 가능한 GPU 디바이스 개수 얻기
device_count = cuda.Device.count()
print(f"GPU 디바이스 개수: {device_count}")

for i in range(device_count):
    # 각 GPU 디바이스에 대한 정보 얻기
    device = cuda.Device(i)
    
    print(f"\nGPU {i + 1} 정보:")
    print(f"이름: {device.name()}")
    
    # SM 수 얻기
    sm_count = device.get_attribute(cuda.device_attribute.MULTIPROCESSOR_COUNT)
    print(f"SM 수: {sm_count}")
    
    # 각 SM의 워프 크기 얻기
    warp_size = device.get_attribute(cuda.device_attribute.WARP_SIZE)
    print(f"워프 크기, 워프당 스레드 수: {warp_size}") # CUDA C Programming Guide. Compute Capability, 32 for 7.0. 워프당 스레드 수 

    # 총 스레드 수 계산
    total_threads = sm_count * warp_size
    print(f"총 스레드 수: {total_threads}")


# GTX 960: Maxwell architecture, Cuda core 1024, 
    # Spec: https://www.nvidia.com/en-us/geforce/900-series/
    # Technical blog: https://developer.nvidia.com/blog/maxwell-most-advanced-cuda-gpu-ever-made/
    # Compute compatibility: 5.x https://docs.nvidia.com/cuda/archive/12.2.1/maxwell-compatibility-guide/index.html
# RTX 3060: Ampere architecture
    # Compute compatibility: https://docs.nvidia.com/cuda/archive/12.2.1/pdf/NVIDIA_Ampere_Compatibility_Guide.pdf
    # Ampere-compute capability 7.0
# A100: 
    
# Compute capability: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capabilities
# Fermi: 2.x
# Kepler: 3.x
# Maxwell: 5.x
# Pascal: 6.x
# Volta: 7.x
# Turing:  7.5
# Ampere: 8.x
    
    
# sample code 
    # https://developer.nvidia.com/cuda-code-samples
    # https://github.com/nvidia/cuda-samples
    # https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html


# CUDA 커널 코드 정의
# mod = SourceModule("""
#     __global__ void myKernel() {
#         int threadId = blockIdx.x * blockDim.x + threadIdx.x;
#         // 스레드 작업 수행
#         // ...
#         printf("블록 인덱스: %d, 스레드 인덱스: %d\\n", blockIdx.x, threadIdx.x);
#     }
# """)

# 블록 및 스레드 설정
numWarp = 32  
numSM = 28  
numThreadPerWarp = 32
numThread = numSM * numSM * numThreadPerWarp
vendor_streamProcessor = 3584

print(numThread)
print(vendor_streamProcessor/numSM/numWarp)
max_num_threads_per_block = 1024


#
# Guidelines for grid and block size
# - 블록당 스레드의 수는 warp size(32)의 배수로 유지
# - 작은 block size는 피한다. 블록 당 스레드 수는 적어도 128이나 256부터 시작
# - 커널의 리소스 사용량에 맞게 block 크기를 조절한다.
# - 디바이스에 충분한 병렬 처리를 제공하기 위해서 SM 수보다 블록 수를 더 많이 유지한다.
# - 실험을 통해 최적의 execution configuration과 resource usage를 찾는다.

## blog 
# import numpy as np
# WarpsPerBlock = np.ceil(ThreadsPerBlock/warpSize)





# https://jihunlee25.tistory.com/entry/Introduction-of-CUDA-Programming-1 구조
# 다나와 RTX 2080 Ti D6 11GB
# RTX 2080 Ti / 12nm / 스트림 프로세서: 4352개 / PCIe3.0x16 / GDDR6(DDR6) / 구매 시 주의사항: 쿨링팬 수(선택), OC(선택), 채굴 여부 판매자 별도 문의 요망
# Specification
# Device : "GeForce RTX 2080 Ti"
# driverVersion : 10010
# runtimeVersion : 10000
#         CUDA Driver Version / Runtime Version  10.1 / 10.0
#         CUDA Capability Major/Minor version number : 7.5
#         Total amount of global memory : 10.73 GBytes (11523260416 bytes)
#         GPU Clock rate :        1545 MHz(1.54 GHz)
#         Memory Clock rate :     7000 Mhz
#         Memory Bus Width :      352-bit
#         L2 Cache Size:  5767168 bytes
#         Total amount of constant memory:        65536 bytes
#         Total amount of shared memory per block:        49152 bytes
#         Total number of registers available per block:  65536
#         Warp Size:      32
#         Maximum number of threads per multiprocessor:   1024
#         Maximum number of thread per block:     1024
#         Maximum sizes of each dimension of a block:     1024 x 1024 x 64
#         Maximum sizes of each dimension of a grid:      2147483647 x 65535 x 65535
# RTX 2080 Ti Engine Specs:	-
# CUDA Cores	4352
# RTX-OPS	76T
# Giga Rays/s	10
# Base Clock (MHz)	1545
# Boost Clock (MHz)	1350
# GTX TITAN X Memory Specs:	-
# Memory Clock	14.0 Gbps
# Standard Memory Config	11 GB GDDR6
# Memory Interface Width	352-bit
# Memory Bandwidth (GB/sec)	616 GB/s
# Display Support:	-
# Maximum Digital Resolution*	7680x4320
# Standard Display Connectors	DisplayPort, HDMI, USB Type-C
# Multi Monitor	4 displays
# HDCP	2.2
# GTX TITAN X Graphics Card Dimensions:	-
# Height	4.556" (115.7 mm)
# Length	10.5" (266.74 mm)
# Width	2-Slot
# Thermal and Power Specs:	-
# Maximum GPU Tempurature (in C)	89 C
# Graphics Card Power (W)	250 W
# Recommended System Power (W)**	650 W
# Supplementary Power Connectors	8-pin + 8-pin
# 1 - 4k 12-bit HDR at 144Hz or 8k 12-bit HDR at 60Hz over one DisplayPort 1.4 connector (with DSC).

# 2 - DisplayPort 1.4a Ready, DSC 1.2 Ready.

# 3 - Recommendation is made based on PC configured with an Intel Core i7 3.2 GHz processor. Pre-built system may require less power depending on system configuration.

# 4 - In preparation for the emerging VirtualLink standard, Turing GPUs have implemented hardware support according to the “VirtualLink Advance Overview”. To learn more about VirtualLink, please see http://www.virtuallink.org.