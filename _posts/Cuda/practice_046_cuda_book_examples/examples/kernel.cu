#include "kernel.cuh"


#define BLOCK_SIZE_2 16
#define BLOCK_SIZE 16 // fastest

void initializeMatrix(float* mat, int rows, int cols, float value) {
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      mat[i * cols + j] = value; // 모든 원소를 value 값으로 설정
    }
  }
}

void printMatrix(const float* matrix, int rows, int cols) {
  std::cout << "Matrix:" << std::endl;
  for (int row = 0; row < rows; row++) {
    for (int col = 0; col < cols; col++) {
      int index = row * cols + col;
      std::cout << matrix[index] << " ";
    }
    std::cout << std::endl;
  }
}

void printCudaDevicesInfo()
{
  int deviceCount;
  cudaGetDeviceCount(&deviceCount);
  for (int deviceIdx = 0; deviceIdx < deviceCount; ++deviceIdx) {
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, deviceIdx);
    std::cout << "Device " << deviceIdx << ": " << deviceProp.name << std::endl; // char[255]
    std::cout << "  Compute Capability: " << deviceProp.major << "." << deviceProp.minor << std::endl; // compute capability 주 버전. 마이너 버전. 둘 다 int
    std::cout << "  Multiprocessors: " << deviceProp.multiProcessorCount << std::endl; // SM 갯수
    std::cout << "  CUDA Cores per Multiprocessor: " << deviceProp.warpSize << std::endl;
    std::cout << "  CUDA cores: " << _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor * deviceProp.multiProcessorCount);
    std::cout << "  Global Memory: " << deviceProp.totalGlobalMem / (1024 * 1024) << " MB" << std::endl; // GPU의 global(device) 메모리 크기. 단위 byte
    std::cout << "  Shared Memory per Block: " << deviceProp.sharedMemPerBlock / 1024 << " KB" << std::endl;
    std::cout << "  Registers per Block: " << deviceProp.regsPerBlock;
    if (deviceProp.regsPerBlock >= 32768) {
      std::cout << "  (64-bit)" << std::endl;
    }
    else {
      std::cout << "  (32-bit)" << std::endl;
    }
    std::cout << "  Max Threads per Block: " << deviceProp.maxThreadsPerBlock << std::endl;
    int registersPerThreadDouble = deviceProp.regsPerBlock / deviceProp.maxThreadsPerBlock;
    std::cout << "    register per thread: " << registersPerThreadDouble << " if maxThreadPerBlock used, variable type is independent" << std::endl;
    std::cout << "  Max Threads per Dimension: (" << deviceProp.maxThreadsDim[0] << ", " << deviceProp.maxThreadsDim[1] << ", " << deviceProp.maxThreadsDim[2] << ")" << std::endl;
    std::cout << "  Max Grid Size: (" << deviceProp.maxGridSize[0] << ", " << deviceProp.maxGridSize[1] << ", " << deviceProp.maxGridSize[2] << ")" << std::endl;
    std::cout << std::endl;

    // occupancy
    int maxThreadsPerSM = deviceProp.maxThreadsPerMultiProcessor;
    int warpSize = deviceProp.warpSize;
    int maxWarpsPerSM = maxThreadsPerSM / warpSize;
    std::cout << "Occupancy = ActiveWarps / MaxWarpsPerSM " << std::endl;
    std::cout << "  MaxWarpsPerSM = maxThreadsPerMultiProcessor / warpSize = " << deviceProp.maxThreadsPerMultiProcessor << " / " << deviceProp.warpSize << " = " << maxWarpsPerSM << std::endl;
    // actually used
    int dim3_blockDimx = 32;
    int dim3_blockDimy = 32;
    int dim3_blockDimz = 1;
    int totalThreadsPerBlock = dim3_blockDimx * dim3_blockDimy * dim3_blockDimz;
    int activeWarp = (totalThreadsPerBlock + warpSize - 1) / warpSize;
    int totalActiveWarps = activeWarp * dim3_blockDimx * dim3_blockDimy * dim3_blockDimz;
    std::cout << "  totalActiveWarp = " << totalActiveWarps << std::endl;
    //Device 0: NVIDIA GeForce RTX 3060
    //Compute Capability : 8.6
    //Multiprocessors : 28
    //CUDA Cores per Multiprocessor : 32
    //MapSMtoCores for SM 8.168 is undefined.Default to use 128 Cores / SM
    //CUDA cores : 128  Global Memory : 12287 MB
    //Shared Memory per Block : 48 KB
    //Max Threads per Block : 1024
    //Max Threads per Dimension : (1024, 1024, 64)
    //Max Grid Size : (2147483647, 65535, 65535)

    // memory bandwidth
    int memoryClock = 15; // 15 Gbps
    int memoryInterface = 192; // 192 bit
    int memoryBandwidth = memoryClock * memoryInterface / 8; // bit to byte
    std::cout << "memory bandwidth = " << memoryBandwidth << " Gbyte/sec" << std::endl;
  }
}


//
//// 5.3.1 2차원 그리드, 2차원 블록 레이아웃
//cudaError_t MatAdd_G2D_B2D_Helper(int* c, const int* a, const int* b, unsigned int COL_SIZE, unsigned int ROW_SIZE)
//{
//  dim3 blockDim(32, 32);
//  dim3 gridDim(ceil((float) COL_SIZE / blockDim.x), ceil((float) ROW_SIZE / blockDim.y));
//  MatAdd_G2D_B2D << <gridDim, blockDim >> > (A, B, C, ROW_SIZE, COL_SIZE);
//}
//
//__global__ void MatAdd_G2D_B2D(float* MatA, float* MatB, float* MatC, int ROW_SIZE, int COL_SIZE)
//{
//  unsigned int col = threadIdx.x + blockIdx.x + blockDim.x;
//  unsigned int row = threadIdx.y + blockIdx.y + blockDim.y;
//  unsigned int index = row * COL_SIZE + col;
//
//  if (col < COL_SIZE && row < ROW_SIZE)
//  {
//    MatC[index] = MatA[index] + MatB[index];
//  }
//}
//
//// 5.3.2 1차원 그리드, 1차원 블록 레이아웃
//cudaError_t MatAdd_G1D_B1D_Helper(int* c, const int* a, const int* b, unsigned int COL_SIZE, unsigned int ROW_SIZE)
//{
//  dim3 blockDim(32);
//  dim3 gridDim(ceil((float)COL_SIZE / blockDim.x));
//  MatAdd_G1D_B1D << < gridDim, blockDim >> > (a, b, c, ROW_SIZE, COL_SIZE);
//
//}
//
//__global__ void MatAdd_G1D_B1D(float* MatA, float* MatB, float* MatC, int ROW_SIZE, int COL_SIZE)
//{
//  unsigned int col = threadIdx.x + blockIdx.x * blockDim.x;
//  if (col < COL_SIZE) {
//    for (int row = 0; row < ROW_SIZE; row++)
//    {
//      int index = row * COL_SIZE + col;
//      MatC[index] = MatA[index] + MatB[index];
//    }
//  }
//}
//
//// 5.3.3 2차원 그리드, 1차원 블록 레이아웃
//cudaError_t MatAdd_G2D_B1D_Helper(int* c, const int* a, const int* b, unsigned int COL_SIZE, unsigned int ROW_SIZE)
//{
//  dim3 blockDim(32);
//  dim3 gridDim(ceil((float)COL_SIZE / blockDim.x), ROW_SIZE);
//  MatAdd_G2D_B1D << <gridDim, blockDim >> > (A, B, C, ROW_SIZE, COL_SIZE);
//}
//
//__global__ void MatAdd_G2D_B1D(float* MatA, float* MatB, float* MatC, int ROW_SIZE, int COL_SIZE)
//{
//  unsigned int col = threadIdx.x + blockIdx.x * blockDim.x;
//  unsigned int row = blockIdx.y;
//  unsigned int index = row * COL_SIZE + col;
//
//  if (col < COL_SIZE && row < ROW_SIZE)
//  {
//    MatC[index] = MatA[index] + MatB[index];
//  }
//}


//// 7.3.1 행렬 C의 크기가 블록 최대 크기(1024)보다 작은 경우
__global__ void MatMulUnder1024(float* deviceMatAPtr, float* deviceMatBPtr, float* deviceMatCPtr, int m, int n, int k)
{
  // A(m,k), B(k, n), C(m, n)
  int row = threadIdx.x;
  int col = threadIdx.y;
  int index = row * n + col; // C 기준

  if ((row < m) && (col < n))
  {
    deviceMatCPtr[index] = 0.0;
    for (int offset = 0; offset < k; offset++)
    {
      deviceMatCPtr[index] += deviceMatAPtr[row * k + offset] * deviceMatBPtr[col + offset * n];
    }
  }
}



// 10.2.2 행렬 C의 크기가 블록의 최대 크기(1024)보다 큰 경우, shared memory 사용
// 먼저, 행렬을 스레드의 크기에 맞게 분할한다. 1/3?
// local block(sharedmemory에 올릴)의 크기는 스레드의 크기이다. 즉, 스레드로 공유메모리에 올리는 작업을 먼저 진행한다.
// BLOCK_SIZE 16은 shared memory가 비슷하거나 약간 빠르고, 4, 8에서는 그냥이 두 배 빠르고, 32, 64는 그냥이 더 빠르고 1.2~1.5배, 128은 메모리 부족으로 컴파일이 되지 않는다.
// 첫 실행은 둘 다 엄청 느리다.
//Elapsed time for Matrix multiply over 1024: 99.537 ms
//Elapsed time for Matrix multiply over 1024, shared memory : 200.066 ms
//Elapsed time for Matrix multiply over 1024 : 0.0058 ms
//Elapsed time for Matrix multiply over 1024, shared memory : 0.0043 ms
//Elapsed time for Matrix multiply over 1024 : 0.0053 ms
//Elapsed time for Matrix multiply over 1024, shared memory : 0.0048 ms
//Elapsed time for Matrix multiply over 1024 : 0.0032 ms
//Elapsed time for Matrix multiply over 1024, shared memory : 0.0033 ms
//Elapsed time for Matrix multiply over 1024 : 0.0025 ms
//Elapsed time for Matrix multiply over 1024, shared memory : 0.0034 ms

__global__ void MatMulOver1024_10_2_sharedMemory(float* deviceMatAPtr, float* deviceMatBPtr, float* deviceMatCPtr, int m, int n, int k)
{
  int row = blockDim.x * blockIdx.x + threadIdx.x;
  int col = blockDim.y * blockIdx.y + threadIdx.y;

  int val = 0;
  __shared__ float subA[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ float subB[BLOCK_SIZE][BLOCK_SIZE];
  int localRow = threadIdx.x;
  int localCol = threadIdx.y;

  for (int bID = 0; bID < ceil((float)k / BLOCK_SIZE); bID++)
  {
    int stride = bID * BLOCK_SIZE;
    // load subA and subB
    if (row >= m || stride + localCol >= k)
    {
      subA[localRow][localCol] = 0;
    }
    else
    {
      subA[localRow][localCol] = deviceMatAPtr[row * k + stride + localCol];
    }
    if (col >= n || stride + localRow >= k)
    {
      subB[localRow][localCol] = 0;
    }
    else
    {
      subB[localRow][localCol] = deviceMatBPtr[(stride + localRow) * n + col];
    }
    __syncthreads();
    // compute C
    for (int i = 0; i < BLOCK_SIZE; i++)
    {
      val += subA[localRow][i] * subB[i][localCol];
    }
    __syncthreads();
  }
  if (row >= m || col >= n)
  {
    return;
  }
  deviceMatCPtr[row * n + col] = val;

  //int row = threadIdx.x + blockIdx.x * blockDim.x;
  //int col = threadIdx.y + blockIdx.y * blockDim.y;
  //int index = row * n + col;
  //deviceMatCPtr[index] = 0;
  //float sum;
  //if ((row < m) && (col < n))
  //{
  //  for (int offset = 0; offset < k; offset++)
  //  {
  //    sum += deviceMatAPtr[row * k + offset] * deviceMatBPtr[col + offset * n];
  //  }
  //  deviceMatCPtr[index] = sum;
  //}
}



__global__ void MatMulOver1024_11_5_sharedMemory_NoBankConflict(float* deviceMatAPtr, float* deviceMatBPtr, float* deviceMatCPtr, int m, int n, int k)
{
  int row = blockDim.x * blockIdx.x + threadIdx.x;
  int col = blockDim.y * blockIdx.y + threadIdx.y;
  int val = 0;
  __shared__ float subA[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ float subB[BLOCK_SIZE][BLOCK_SIZE];
  int localRow = threadIdx.x;
  int localCol = threadIdx.y;
  for (int bID = 0; bID < ceil((float)k / BLOCK_SIZE); bID++)
  {
    int stride = bID * BLOCK_SIZE;
    // load subA and subB
    if (row >= m || stride + localCol >= k)
    {
      subA[localCol][localRow] = 0;
    }
    else
    {
      subA[localCol][localRow] = deviceMatAPtr[row * k + stride + localCol];
    }
    if (col >= n || stride + localRow >= k)
    {
      subB[localRow][localCol] = 0;
    }
    else
    {
      subB[localRow][localCol] = deviceMatBPtr[(stride + localRow) * n + col];
    }
    __syncthreads();
    // compute C
    for (int i = 0; i < BLOCK_SIZE; i++)
    {
      val += subA[i][localRow] * subB[i][localCol];
    }
    __syncthreads();
  }
  if (row >= m || col >= n)
  {
    return;
  }
  deviceMatCPtr[row * n + col] = val;
}


void cuBLAS_MatMul()
{
  int rowsA = 2048, colsA = 1024, rowsB = 1024, colsB = 1024;
  // 호스트 메모리 할당 및 초기화
  float* A = new float[rowsA * colsA];
  float* B = new float[rowsB * colsB];
  float* C = new float[rowsA * colsB];
  for (int i = 0; i < rowsA * colsA; ++i) A[i] = 1.0f; // 예제를 위한 초기화
  for (int i = 0; i < rowsB * colsB; ++i) B[i] = 2.0f; // 예제를 위한 초기화

  // 디바이스 메모리 할당
  float* d_A, * d_B, * d_C;
  cudaMalloc(&d_A, rowsA * colsA * sizeof(float));
  cudaMalloc(&d_B, rowsB * colsB * sizeof(float));
  cudaMalloc(&d_C, rowsA * colsB * sizeof(float));

  // cuBLAS 핸들 생성
  cublasHandle_t handle;
  cublasCreate(&handle);

  // 데이터를 디바이스로 복사
  cublasSetMatrix(rowsA, colsA, sizeof(float), A, rowsA, d_A, rowsA);
  cublasSetMatrix(rowsB, colsB, sizeof(float), B, rowsB, d_B, rowsB);

  // 행렬 곱셈 수행
  const float alpha = 1.0f;
  const float beta = 0.0f;
  utils::Timer timer;
  timer.on("CuBLAS");
  cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, colsB, rowsA, colsA, &alpha, d_A, colsA, d_B, colsB, &beta, d_C, colsB);
  timer.elapsed();
  // 결과를 호스트로 복사
  cublasGetMatrix(rowsA, colsB, sizeof(float), d_C, rowsA, C, rowsA);

  // 결과 출력 (예제 코드에서는 처음 몇 개의 원소만 출력)
  //for (int i = 0; i < 10; ++i) {
  //  std::cout << C[i] << " ";
  //}
  //std::cout << std::endl;

  // 리소스 해제
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  cublasDestroy(handle);

  delete[] A;
  delete[] B;
  delete[] C;

}


void MatMulOver1024_10_2_SharedMemory_Helper()
{
  int m = 2048;
  int n = 1024;
  int k = 1024;

  int sizeA = m * k;
  int sizeB = k * n;
  int sizeC = m * n;

  float* HostAMatrixPtr = new float[sizeA];
  float* HostBMatrixPtr = new float[sizeB];
  float* HostCMatrixPtr = new float[sizeC];

  initializeMatrix(HostAMatrixPtr, m, k, 1);
  initializeMatrix(HostBMatrixPtr, k, n, 2);
  initializeMatrix(HostCMatrixPtr, m, n, 0);

  //printMatrix(HostAMatrixPtr, m, k); // 1
  //printMatrix(HostBMatrixPtr, k, n); // 2

  float* deviceAMatrixPtr, * deviceBMatrixPtr, * deviceCMatrixPtr;

  cudaError_t status;

  status = cudaMalloc(&deviceAMatrixPtr, sizeA * sizeof(float));
  status = cudaMalloc(&deviceBMatrixPtr, sizeB * sizeof(float));
  status = cudaMalloc(&deviceCMatrixPtr, sizeC * sizeof(float));
  status = cudaMemset(deviceAMatrixPtr, 0, sizeA * sizeof(float));
  status = cudaMemset(deviceBMatrixPtr, 0, sizeB * sizeof(float));
  status = cudaMemset(deviceCMatrixPtr, 0, sizeC * sizeof(float));
  status = cudaMemcpy(deviceAMatrixPtr, HostAMatrixPtr, sizeA * sizeof(float), cudaMemcpyHostToDevice);
  status = cudaMemcpy(deviceBMatrixPtr, HostBMatrixPtr, sizeB * sizeof(float), cudaMemcpyHostToDevice);

  int blockSize_x = 32;
  int blockSize_y = 32;
  int gridSize_x = ceil(float(m) / blockSize_x);
  int gridSize_y = ceil(float(n) / blockSize_y);
  dim3 gridDim(gridSize_x, gridSize_y, 1);
  dim3 blockDim(blockSize_x, blockSize_y, 1);

  utils::Timer timer;
  timer.on("Matrix multiply over 1024, shared memory");
  MatMulOver1024_10_2_sharedMemory << <gridDim, blockDim >> > (deviceAMatrixPtr, deviceBMatrixPtr, deviceCMatrixPtr, m, n, k);
  cudaDeviceSynchronize();
  timer.elapsed();
  status = cudaMemcpy(HostCMatrixPtr, deviceCMatrixPtr, sizeC * sizeof(float), cudaMemcpyDeviceToHost);
  //printMatrix(HostCMatrixPtr, m, n);
  //printf("%f", HostCMatrixPtr[2047]);

  status = cudaFree(deviceAMatrixPtr);
  status = cudaFree(deviceBMatrixPtr);
  status = cudaFree(deviceCMatrixPtr);

  delete[] HostAMatrixPtr;
  delete[] HostBMatrixPtr;
  delete[] HostCMatrixPtr;
}



void MatMulOver1024_11_5_SharedMemory_NoBankConflict_Helper()
{
  int m = 2048;
  int n = 1024;
  int k = 1024;

  int sizeA = m * k;
  int sizeB = k * n;
  int sizeC = m * n;

  float* HostAMatrixPtr = new float[sizeA];
  float* HostBMatrixPtr = new float[sizeB];
  float* HostCMatrixPtr = new float[sizeC];

  initializeMatrix(HostAMatrixPtr, m, k, 1);
  initializeMatrix(HostBMatrixPtr, k, n, 2);
  initializeMatrix(HostCMatrixPtr, m, n, 0);

  //printMatrix(HostAMatrixPtr, m, k); // 1
  //printMatrix(HostBMatrixPtr, k, n); // 2

  float* deviceAMatrixPtr, * deviceBMatrixPtr, * deviceCMatrixPtr;

  cudaError_t status;

  status = cudaMalloc(&deviceAMatrixPtr, sizeA * sizeof(float));
  status = cudaMalloc(&deviceBMatrixPtr, sizeB * sizeof(float));
  status = cudaMalloc(&deviceCMatrixPtr, sizeC * sizeof(float));
  status = cudaMemset(deviceAMatrixPtr, 0, sizeA * sizeof(float));
  status = cudaMemset(deviceBMatrixPtr, 0, sizeB * sizeof(float));
  status = cudaMemset(deviceCMatrixPtr, 0, sizeC * sizeof(float));
  status = cudaMemcpy(deviceAMatrixPtr, HostAMatrixPtr, sizeA * sizeof(float), cudaMemcpyHostToDevice);
  status = cudaMemcpy(deviceBMatrixPtr, HostBMatrixPtr, sizeB * sizeof(float), cudaMemcpyHostToDevice);

  int blockSize_x = 32;
  int blockSize_y = 32;
  int gridSize_x = ceil(float(m) / blockSize_x);
  int gridSize_y = ceil(float(n) / blockSize_y);
  dim3 gridDim(gridSize_x, gridSize_y, 1);
  dim3 blockDim(blockSize_x, blockSize_y, 1);

  utils::Timer timer;
  timer.on("Matrix multiply over 1024, shared memory, no bank conflict");
  MatMulOver1024_11_5_sharedMemory_NoBankConflict << <gridDim, blockDim >> > (deviceAMatrixPtr, deviceBMatrixPtr, deviceCMatrixPtr, m, n, k);
  cudaDeviceSynchronize();
  timer.elapsed();
  status = cudaMemcpy(HostCMatrixPtr, deviceCMatrixPtr, sizeC * sizeof(float), cudaMemcpyDeviceToHost);
  //printMatrix(HostCMatrixPtr, m, n);
  //printf("%f", HostCMatrixPtr[2047]);

  status = cudaFree(deviceAMatrixPtr);
  status = cudaFree(deviceBMatrixPtr);
  status = cudaFree(deviceCMatrixPtr);

  delete[] HostAMatrixPtr;
  delete[] HostBMatrixPtr;
  delete[] HostCMatrixPtr;
}




// 7.3.2 행렬 C의 크기가 블록의 최대 크기(1024)보다 큰 경우
__global__ void MatMulOver1024(float* deviceMatAPtr, float* deviceMatBPtr, float* deviceMatCPtr, int m, int n, int k)
{
  // A(m,k), B(k, n), C(m, n)
  int row = threadIdx.x + blockIdx.x * blockDim.x;
  int col = threadIdx.y + blockIdx.y * blockDim.y;
  int index = row * n + col; // C 기준
  deviceMatCPtr[index] = 0;
  float sum;
  if ((row < m) && (col < n))
  {
    for (int offset = 0; offset < k; offset++)
    {
      sum += deviceMatAPtr[row * k + offset] * deviceMatBPtr[col + offset * n];
    }
    deviceMatCPtr[index] = sum;
  }
}


// 11.1.2 행렬 C의 크기가 블록의 최대 크기(1024)보다 큰 경우, xy swap (메모리 접근 최적화)
__global__ void MatMulOver1024_xyswap(float* deviceMatAPtr, float* deviceMatBPtr, float* deviceMatCPtr, int m, int n, int k)
{
  // A(m,k), B(k, n), C(m, n)
  int row = threadIdx.y + blockIdx.y * blockDim.y;
  int col = threadIdx.x + blockIdx.x * blockDim.x;
  int index = row * n + col; // C 기준
  deviceMatCPtr[index] = 0;
  float sum;
  if ((row < m) && (col < n))
  {
    for (int offset = 0; offset < k; offset++)
    {
      sum += deviceMatAPtr[row * k + offset] * deviceMatBPtr[col + offset * n];
    }
    deviceMatCPtr[index] = sum;
  }
}



void MatMulOver1024_Helper()
{
  int m = 2048;
  int n = 1024;
  int k = 1024;

  int sizeA = m * k;
  int sizeB = k * n;
  int sizeC = m * n;

  float* HostAMatrixPtr = new float[sizeA];
  float* HostBMatrixPtr = new float[sizeB];
  float* HostCMatrixPtr = new float[sizeC];

  initializeMatrix(HostAMatrixPtr, m, k, 1);
  initializeMatrix(HostBMatrixPtr, k, n, 2);
  initializeMatrix(HostCMatrixPtr, m, n, 0);

  //printMatrix(HostAMatrixPtr, m, k); // 1
  //printMatrix(HostBMatrixPtr, k, n); // 2

  float* deviceAMatrixPtr, * deviceBMatrixPtr, * deviceCMatrixPtr;

  cudaError_t status;

  status = cudaMalloc(&deviceAMatrixPtr, sizeA * sizeof(float));
  status = cudaMalloc(&deviceBMatrixPtr, sizeB * sizeof(float));
  status = cudaMalloc(&deviceCMatrixPtr, sizeC * sizeof(float));
  status = cudaMemset(deviceAMatrixPtr, 0, sizeA * sizeof(float));
  status = cudaMemset(deviceBMatrixPtr, 0, sizeB * sizeof(float));
  status = cudaMemset(deviceCMatrixPtr, 0, sizeC * sizeof(float));
  status = cudaMemcpy(deviceAMatrixPtr, HostAMatrixPtr, sizeA * sizeof(float), cudaMemcpyHostToDevice);
  status = cudaMemcpy(deviceBMatrixPtr, HostBMatrixPtr, sizeB * sizeof(float), cudaMemcpyHostToDevice);

  int blockSize_x = 32;
  int blockSize_y = 32;
  int gridSize_x = ceil(float(m) / blockSize_x);
  int gridSize_y = ceil(float(n) / blockSize_y);
  dim3 gridDim(gridSize_x, gridSize_y, 1);
  dim3 blockDim(blockSize_x, blockSize_y, 1);

  utils::Timer timer;
  timer.on("Matrix multiply over 1024");
  MatMulOver1024 << <gridDim, blockDim >> > (deviceAMatrixPtr, deviceBMatrixPtr, deviceCMatrixPtr, m, n, k);
  cudaDeviceSynchronize();
  timer.elapsed();
  status = cudaMemcpy(HostCMatrixPtr, deviceCMatrixPtr, sizeC * sizeof(float), cudaMemcpyDeviceToHost);
  //printMatrix(HostCMatrixPtr, m, n);
  //printf("%f", HostCMatrixPtr[2047]);

  status = cudaFree(deviceAMatrixPtr);
  status = cudaFree(deviceBMatrixPtr);
  status = cudaFree(deviceCMatrixPtr);

  delete[] HostAMatrixPtr;
  delete[] HostBMatrixPtr;
  delete[] HostCMatrixPtr;
}




void MatMulOver1024_xyswap_Helper()
{
  int m = 2048;
  int n = 1024;
  int k = 1024;

  int sizeA = m * k;
  int sizeB = k * n;
  int sizeC = m * n;

  float* HostAMatrixPtr = new float[sizeA];
  float* HostBMatrixPtr = new float[sizeB];
  float* HostCMatrixPtr = new float[sizeC];

  initializeMatrix(HostAMatrixPtr, m, k, 1);
  initializeMatrix(HostBMatrixPtr, k, n, 2);
  initializeMatrix(HostCMatrixPtr, m, n, 0);

  //printMatrix(HostAMatrixPtr, m, k); // 1
  //printMatrix(HostBMatrixPtr, k, n); // 2

  float* deviceAMatrixPtr, * deviceBMatrixPtr, * deviceCMatrixPtr;

  cudaError_t status;

  status = cudaMalloc(&deviceAMatrixPtr, sizeA * sizeof(float));
  status = cudaMalloc(&deviceBMatrixPtr, sizeB * sizeof(float));
  status = cudaMalloc(&deviceCMatrixPtr, sizeC * sizeof(float));
  status = cudaMemset(deviceAMatrixPtr, 0, sizeA * sizeof(float));
  status = cudaMemset(deviceBMatrixPtr, 0, sizeB * sizeof(float));
  status = cudaMemset(deviceCMatrixPtr, 0, sizeC * sizeof(float));
  status = cudaMemcpy(deviceAMatrixPtr, HostAMatrixPtr, sizeA * sizeof(float), cudaMemcpyHostToDevice);
  status = cudaMemcpy(deviceBMatrixPtr, HostBMatrixPtr, sizeB * sizeof(float), cudaMemcpyHostToDevice);

  int blockSize_x = 32;
  int blockSize_y = 32;
  int gridSize_x = ceil(float(n) / blockSize_x);
  int gridSize_y = ceil(float(m) / blockSize_y);
  dim3 gridDim(gridSize_x, gridSize_y, 1);
  dim3 blockDim(blockSize_x, blockSize_y, 1);

  utils::Timer timer;
  timer.on("Matrix multiply over 1024, xy swap");
  MatMulOver1024_xyswap << <gridDim, blockDim >> > (deviceAMatrixPtr, deviceBMatrixPtr, deviceCMatrixPtr, m, n, k);
  cudaDeviceSynchronize();
  timer.elapsed();
  status = cudaMemcpy(HostCMatrixPtr, deviceCMatrixPtr, sizeC * sizeof(float), cudaMemcpyDeviceToHost);
  //printMatrix(HostCMatrixPtr, m, n);
  //printf("%f", HostCMatrixPtr[2047]);

  status = cudaFree(deviceAMatrixPtr);
  status = cudaFree(deviceBMatrixPtr);
  status = cudaFree(deviceCMatrixPtr);

  delete[] HostAMatrixPtr;
  delete[] HostBMatrixPtr;
  delete[] HostCMatrixPtr;
}


// 7.4.1
//#define BLOCK_SIZE 16
//void MatMulUnder1024_Helper(char* argv[])
void MatMulUnder1024_Helper()
{
  int m, n, k;
  m = 2;
  n = 3;
  k = 4;
  //m = atoi(argv[1]);
  //n = atoi(argv[2]);
  //k = atoi(argv[3]);


  int sizeA = m * k;
  int sizeB = k * n;
  int sizeC = m * n;

  // cpu

  //float AA[8] = { 1, 1, 1, 1, 1, 1, 1, 1 };
  //float BB[12] = { 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2 };
  //float CCgpu[6] = { 0, 0, 0, 0, 0, 0 };
  //int index, row, col, offset;
  //for (row = 0; row < m; row++)
  //{
  //  for (col = 0; col < n; col++)
  //  {
  //    for (offset = 0; offset < k; offset++)
  //    {
  //      index = row * n + col;
  //      CCgpu[index] += (AA[row * k + offset] * BB[offset * n + col]);
  //    }
  //  }
  //}
  //printMatrix(AA, m, k);
  //printMatrix(BB, k, n);
  //printMatrix(CCgpu, m, n);

  float* A = new float[sizeA];
  float* B = new float[sizeB];
  float* Cgpu = new float[sizeC];

  initializeMatrix(A, m, k, 1);
  initializeMatrix(B, k, n, 2);
  initializeMatrix(Cgpu, m, n, 0);

  printMatrix(A, m, k);
  printMatrix(B, k, n);

  float* dA, * dB, * dC;

  cudaError_t status;

  ////// 1. Allocate device memory for dA, dB, dC
  status = cudaMalloc(&dA, sizeA * sizeof(float)); // &dA에 할당된 메모리의 주소를 전달하는 것.
  status = cudaMemset(dA, 0, sizeA * sizeof(float));
  status = cudaMalloc(&dB, sizeB * sizeof(float));
  status = cudaMemset(dB, 0, sizeB * sizeof(float));
  status = cudaMalloc(&dC, sizeC * sizeof(float));
  status = cudaMemset(dC, 0, sizeC * sizeof(float));

  ////// 2. Send(Copy) the input matrices to GPU (A -> dA, B -> dB)
  status = cudaMemcpy(dA, A, sizeA * sizeof(float), cudaMemcpyHostToDevice);
  status = cudaMemcpy(dB, B, sizeB * sizeof(float), cudaMemcpyHostToDevice);

  ////// 3. Set the thread layout
  dim3 gridDim(ceil((float)m / BLOCK_SIZE), ceil((float)n / BLOCK_SIZE), 1);
  dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE, 1);

  ////// 4. Kernel call
  MatMulUnder1024 << < gridDim, blockDim >>> (dA, dB, dC, m, n, k);

  ////// 5. Get(copy) the result from GPU to host memory (dC -> Cgpu)
  cudaDeviceSynchronize();
  status = cudaMemcpy(Cgpu, dC, sizeC * sizeof(float), cudaMemcpyDeviceToHost);


  printMatrix(Cgpu, m, n);

  ////// 6. Release device memory space (dA, dB, dC)
  status = cudaFree(dA);
  status = cudaFree(dB);
  status = cudaFree(dC);

  delete[] A;
  delete[] B;
  delete[] Cgpu;

  //return status;
}





//// 8.2.2 블록 수준 메모리
//// filtering, convolution, histogram, resize, sliding window
//// 정적 할당:
////  컴파일시 크기가 결정된다
////  같은 블록 내 모든 스레드가 공유한다. 각 블록 마다 선언된다.
////  용량: Compute capability 마다 16~96KB, RTX 3060은 46KB
////  속도: 1~4 GPU cycle
//__global__ void sharedMemory_Static_Kernel(void)
//{
//  __shared__ int sharedMemory[512];
//}
//
//
//// 동적 할당:
//extern __shared__ int sharedPool[];
//int* sIntArray = sharedPool;
//float* sFloatArray = (float*)&sharedPool[sizeIntArr];
//
//__global__ void sharedMemory_Dynamic_Kernel(void)
//{
//  sIntArray[threadIdx.x] = 0;
//  sFloatArray[threadIdx.x] = 0.0f;
//}



//// 9.1 공유메모리 사용전, 행렬 C의 크기가 블록 최대 크기(1024)보다 작은 경우
// 1024x768 혹은 800x600
#define ROW_SIZE 64
#define COL_SIZE 32
#define K_SIZE 48
__global__ void MatMulUnder1024_9_1_sharedMemory(float* deviceMatAPtr, float* deviceMatBPtr, float* deviceMatCPtr)
{
  // A(m,k), B(k, n), C(m, n)
  int row = threadIdx.x;
  int col = threadIdx.y;
  int index = row * COL_SIZE + col; // C 기준

  __shared__ float sA[ROW_SIZE][K_SIZE]; // 64 x 48 x 4 byte = 12,288 byte
  __shared__ float sB[K_SIZE][COL_SIZE]; // 48 x 32 x 4 byte = 6,144 byte

  if (row == 0)
  {
    for (int k = 0; k < K_SIZE; k++)
    {
      sB[k][col] = deviceMatBPtr[col + k * COL_SIZE];
    }
  }
  if (col == 0)
  {
    for (int k = 0; k < K_SIZE; k++)
    {
      sA[row][k] = deviceMatAPtr[row * K_SIZE + k];
    }
  }

  __syncthreads(); // 없을 때 94, 있을 때 96

  float result = 0;
  for (int k = 0; k < K_SIZE; k++)
  {
    result += sA[row][k] * sB[k][col];
    deviceMatCPtr[index] = result;
  }  
}



__global__ void MatMulUnder1024_9_1_globalMemory(float* deviceMatAPtr, float* deviceMatBPtr, float* deviceMatCPtr)
{
  // A(m,k), B(k, n), C(m, n)
  int row = threadIdx.x;
  int col = threadIdx.y;
  int index = row * COL_SIZE + col; // C 기준

  float result = 0;
  if ((row < ROW_SIZE) && (col < COL_SIZE))
  {
    {
      deviceMatCPtr[index] = 0.0;
      for (int offset = 0; offset < K_SIZE; offset++)
      {
        deviceMatCPtr[index] += deviceMatAPtr[row * K_SIZE + offset] * deviceMatBPtr[col + offset * COL_SIZE];
      }
    }
  }
}


void MatMulUnder1024_9_1_sharedMemory_Helper()
{
  int m, n, k;
  m = ROW_SIZE;
  n = COL_SIZE;
  k = K_SIZE;
  //m = atoi(argv[1]);
  //n = atoi(argv[2]);
  //k = atoi(argv[3]);


  int sizeA = m * k;
  int sizeB = k * n;
  int sizeC = m * n;

  float* A = new float[sizeA];
  float* B = new float[sizeB];
  float* Cgpu = new float[sizeC];

  initializeMatrix(A, m, k, 1);
  initializeMatrix(B, k, n, 2);
  initializeMatrix(Cgpu, m, n, 0);

  printMatrix(A, m, k);
  printMatrix(B, k, n);

  float* dA, * dB, * dC;

  cudaError_t status;

  ////// 1. Allocate device memory for dA, dB, dC
  status = cudaMalloc(&dA, sizeA * sizeof(float)); // &dA에 할당된 메모리의 주소를 전달하는 것.
  status = cudaMemset(dA, 0, sizeA * sizeof(float));
  status = cudaMalloc(&dB, sizeB * sizeof(float));
  status = cudaMemset(dB, 0, sizeB * sizeof(float));
  status = cudaMalloc(&dC, sizeC * sizeof(float));
  status = cudaMemset(dC, 0, sizeC * sizeof(float));

  ////// 2. Send(Copy) the input matrices to GPU (A -> dA, B -> dB)
  status = cudaMemcpy(dA, A, sizeA * sizeof(float), cudaMemcpyHostToDevice);
  status = cudaMemcpy(dB, B, sizeB * sizeof(float), cudaMemcpyHostToDevice);

  ////// 3. Set the thread layout
//  dim3 gridDim(ceil((float)m / BLOCK_SIZE_2), ceil((float)n / BLOCK_SIZE_2), 1);

  dim3 blockDim(64, 16, 1); // dimx * dimy * dimz <= 1024 (32, 32, 1)
  dim3 gridDim(1, 4, 1); // total data / threadperblock

  ////// 4. Kernel call
  utils::Timer timer;
  timer.on("shared memory time");
  MatMulUnder1024_9_1_sharedMemory << < gridDim, blockDim >> > (dA, dB, dC);
  timer.elapsed();


  timer.on("global memory time");
  MatMulUnder1024_9_1_globalMemory << < gridDim, blockDim >> > (dA, dB, dC);
  timer.elapsed();

  ////// 5. Get(copy) the result from GPU to host memory (dC -> Cgpu)
  status = cudaMemcpy(Cgpu, dC, sizeC * sizeof(float), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

  printMatrix(Cgpu, m, n);

  ////// 6. Release device memory space (dA, dB, dC)
  status = cudaFree(dA);
  status = cudaFree(dB);
  status = cudaFree(dC);

  delete[] A;
  delete[] B;
  delete[] Cgpu;

  //return status;
}

#define BLOCK_SIZE 64
__global__ void syncWarpKernel()
{
  int tID = threadIdx.x;
  int warpID = (int)(tID / 32);
  __shared__ int masterID[BLOCK_SIZE / 32];

  if (threadIdx.x % 32 == 0)
  {
    masterID[warpID] = tID;
  }

  __syncwarp();
  printf("[T%d] The master of our warp is %d \n", tID, masterID[warpID]);
}

void syncWarp_helper()
{
  syncWarpKernel <<<1, BLOCK_SIZE >>>();
  cudaThreadSynchronize();
}