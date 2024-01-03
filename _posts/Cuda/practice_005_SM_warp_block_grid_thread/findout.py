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
    print(f"워프 크기: {warp_size}") # CUDA C Programming Guide. Compute Capability, 32 for 7.0

    # 총 스레드 수 계산
    total_threads = sm_count * warp_size
    print(f"총 스레드 수: {total_threads}")




# CUDA 커널 코드 정의
mod = SourceModule("""
    __global__ void myKernel() {
        int threadId = blockIdx.x * blockDim.x + threadIdx.x;
        // 스레드 작업 수행
        // ...
        printf("블록 인덱스: %d, 스레드 인덱스: %d\\n", blockIdx.x, threadIdx.x);
    }
""")

# 블록 및 스레드 설정
block_size = 32  # 블록 내 스레드 수
grid_size = 28   # 전체 블록 수 (SM 수와 일치해야 함)

# 커널 함수 호출
my_kernel = mod.get_function("myKernel")
my_kernel(block=(block_size, 1, 1), grid=(grid_size, 1, 1))

# CUDA 커널 호출 완료 대기
cuda.Context.synchronize()






# https://jihunlee25.tistory.com/entry/Introduction-of-CUDA-Programming-1 구조