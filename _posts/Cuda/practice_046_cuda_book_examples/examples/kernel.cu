#include "kernel.cuh"

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
    std::cout << "  Max Threads per Block: " << deviceProp.maxThreadsPerBlock << std::endl;
    std::cout << "  Max Threads per Dimension: (" << deviceProp.maxThreadsDim[0] << ", " << deviceProp.maxThreadsDim[1] << ", " << deviceProp.maxThreadsDim[2] << ")" << std::endl;
    std::cout << "  Max Grid Size: (" << deviceProp.maxGridSize[0] << ", " << deviceProp.maxGridSize[1] << ", " << deviceProp.maxGridSize[2] << ")" << std::endl;
    std::cout << std::endl;
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
  }
}

// 7.3.1 행렬 C의 크기가 블록 최대 크기(1024)보다 작은 경우
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

// 7.3.2 행렬 C의 크기가 블록의 최대 크기(1024)보다 큰 경우
__global__ void MatMulOver1024(float* deviceMatAPtr, float* deviceMatBPtr, float* deviceMatCPtr, int m, int n, int k)
{
  // A(m,k), B(k, n), C(m, n)
  int row = threadIdx.x + blockIdx.x * blockDim.x;
  int col = threadIdx.y + blockIdx.y * blockDim.y;
  int index = row * n + col; // C 기준

  deviceMatCPtr[index] = 0;
  for (int offset = 0; offset < k; offset++)
  {
    deviceMatCPtr[index] += deviceMatAPtr[row * k + offset] * deviceMatBPtr[col + offset * n];
  }
}



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

// 7.4.1
#define BLOCK_SIZE 16
void MatMulUnder1024_Helper(char* argv[])
{
  int m, n, k;
  m = atoi(argv[1]);
  n = atoi(argv[2]);
  k = atoi(argv[3]);

  int sizeA = m * k;
  int sizeB = k * n;
  int sizeC = m * n;

  // cpu
  //m = 2;
  //n = 3;
  //k = 4;
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