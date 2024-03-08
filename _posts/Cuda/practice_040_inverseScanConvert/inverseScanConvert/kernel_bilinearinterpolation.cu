#include "kernel_bilinearinterpolation.cuh"

__global__ void generateIndexWiseRangeAngleMeshVecKernel
(
  float* deviceIndexWiseRangeMeshVecPtr,
  float* deviceIndexWiseAngleMeshVecPtr,
  double* deviceXPtr,
  double* deviceYPtr,
  int cols,
  int rows,
  float firstSrcRangeValue,
  float lastSrcRangeValue,
  int numRanges,
  float firstSrcAngleValue,
  float lastSrcAngleValue,
  int numAngles
)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < cols * rows)
  {
    int row = tid % rows;
    int col = tid / rows;
    int index = row + rows * col;
    float rMm = std::sqrt(std::pow((float)deviceXPtr[col], 2) + std::pow((float)deviceYPtr[row], 2));
    float thetaDeg = 90.0 - std::atan2((float)deviceYPtr[row], (float)deviceXPtr[col]) * 180.0 / PI; // atan2: angle to x axis.
    float IndexWiseRangeValue = (rMm - firstSrcRangeValue) * ((float)numRanges - 1.0f) / (lastSrcRangeValue - firstSrcRangeValue);
    if (IndexWiseRangeValue < 0 || IndexWiseRangeValue > ((float)numRanges - 1.0f))
    {
      deviceIndexWiseRangeMeshVecPtr[index] = -1.0;
    }
    else
    {
      deviceIndexWiseRangeMeshVecPtr[index] = IndexWiseRangeValue;
    }
    float IndexWiseAngleValue = (thetaDeg - firstSrcAngleValue) * ((float)numAngles - 1.0f) / (lastSrcAngleValue - firstSrcAngleValue);
    if (IndexWiseAngleValue < 0 || IndexWiseAngleValue > ((float)numAngles - 1.0f))
    {
      deviceIndexWiseAngleMeshVecPtr[index] = -1.0;
    }
    else
    {
      deviceIndexWiseAngleMeshVecPtr[index] = IndexWiseAngleValue;
    }
  }
}

__global__ void inverseScanConversionKernels
( unsigned char** deviceOutputVecVol,
  unsigned char** deviceVecVol,
  int srcRows,
  int srcCols,
  int numVectors,
  int dstRows,
  int dstCols,
  double* deviceIndexX,
  double* deviceIndexY
)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= dstRows * dstCols)
    return;

  int row = (int)deviceIndexY[idx];
  int col = (int)deviceIndexX[idx];
  if (row < 0 || col < 0 || row >= (srcRows - 1) || col >= (srcCols - 1))
  {
    for (int vec = 0; vec < numVectors; ++vec)
    {
      deviceOutputVecVol[vec][idx] = 0;
    }
  }
  else
  {
    double row_frac = deviceIndexY[idx] - (double)row;
    double col_frac = deviceIndexX[idx] - (double)col;
    for (int vec = 0; vec < numVectors; ++vec)
    {
      double intensity_topleft = (double)deviceVecVol[vec][row * srcCols + col];
      double intensity_topright = (double)deviceVecVol[vec][row * srcCols + col + 1];
      double intensity_bottomleft = (double)deviceVecVol[vec][(row + 1) * srcCols + col];
      double intensity_bottomright = (double)deviceVecVol[vec][(row + 1) * srcCols + col + 1];
      // Bilinear interpolation
      deviceOutputVecVol[vec][idx] = (int)(((1.0f - row_frac) * ((1.0f - col_frac) * intensity_topleft + col_frac * intensity_topright)
        + row_frac * ((1.0f - col_frac) * intensity_bottomleft + col_frac * intensity_bottomright)) + 0.5f);
    }
  }
}

std::pair<std::vector<float>, std::vector<float>> generateIndexWiseRangeAngleMeshVecWithCuda
( std::vector<double> X,
  std::vector<double> Y,
  std::vector<double> srcRangeA,
  std::vector<double> srcAngleA
)
{
  int cols = X.size();
  int rows = Y.size();
  int numSamplesPerFrame = cols * rows;
  int numRanges = srcRangeA.size();
  int numAngles = srcAngleA.size();
  float firstSrcRangeValue = (float) srcRangeA[0];
  float lastSrcRangeValue = (float) srcRangeA[srcRangeA.size() - 1];
  float firstSrcAngleValue = (float) srcAngleA[0];
  float lastSrcAngleValue = (float) srcAngleA[srcAngleA.size() - 1];
  std::vector<float> hostIndexWiseRangeMeshVec(X.size() * Y.size());
  std::vector<float> hostIndexWiseAngleMeshVec(X.size() * Y.size());
  double* deviceXPtr, * deviceYPtr;// , * deviceXMeshVecPtr, * deviceYMeshVecPtr, * deviceRangeMeshVecPtr, * deviceAngleMeshVecPtr;
  float* deviceIndexWiseRangeMeshVecPtr, * deviceIndexWiseAngleMeshVecPtr;
  cudaMalloc(&deviceXPtr, cols * sizeof(double));
  cudaMalloc(&deviceYPtr, rows * sizeof(double));
  cudaMalloc(&deviceIndexWiseRangeMeshVecPtr, numSamplesPerFrame * sizeof(float));
  cudaMalloc(&deviceIndexWiseAngleMeshVecPtr, numSamplesPerFrame * sizeof(float));
  cudaMemcpy(deviceXPtr, X.data(), cols * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(deviceYPtr, Y.data(), rows * sizeof(double), cudaMemcpyHostToDevice);
  int numThreads = 256; // 256 from 32 - 1024
  auto start = std::chrono::high_resolution_clock::now();
  int threadPerBlock = numThreads;
  dim3 blockDim(threadPerBlock, 1, 1);
  dim3 blockPerGrid = (numSamplesPerFrame + threadPerBlock - 1) / threadPerBlock;
  generateIndexWiseRangeAngleMeshVecKernel << <blockPerGrid, blockDim >> > (deviceIndexWiseRangeMeshVecPtr, deviceIndexWiseAngleMeshVecPtr, deviceXPtr, deviceYPtr, cols, rows, firstSrcRangeValue, lastSrcRangeValue, numRanges, firstSrcAngleValue, lastSrcAngleValue, numAngles);
  cudaDeviceSynchronize();
  cudaMemcpy(hostIndexWiseRangeMeshVec.data(), deviceIndexWiseRangeMeshVecPtr, numSamplesPerFrame * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(hostIndexWiseAngleMeshVec.data(), deviceIndexWiseAngleMeshVecPtr, numSamplesPerFrame * sizeof(float), cudaMemcpyDeviceToHost);
  cudaFree(deviceXPtr);
  cudaFree(deviceYPtr);
  cudaFree(deviceIndexWiseRangeMeshVecPtr);
  cudaFree(deviceIndexWiseAngleMeshVecPtr);
  return std::make_pair(hostIndexWiseRangeMeshVec, hostIndexWiseAngleMeshVec);
}

cudaError_t inverseScanConversionWithCuda
( std::vector<unsigned char>& outputVecVol,
  std::vector<unsigned char>& nrrdData,
  unsigned int srcRows,
  unsigned int srcColumns,
  unsigned int numberOfFrames,
  unsigned int dstRows,
  unsigned int dstColumns,
  std::vector<double> srcIndexWiseXVec,
  std::vector<double> srcIndexWiseYVec
)
{
  cudaError_t cudaStatus;
  double* deviceIndexX;
  double* deviceIndexY;
  std::vector<unsigned char*> deviceVecEachSlice(numberOfFrames, nullptr);
  unsigned char** deviceVecVol;
  unsigned char** deviceOutVol;
  unsigned char** hostVecEachSlicePtr = new unsigned char* [numberOfFrames];

  // Select GPU
  cudaStatus = cudaSetDevice(0);
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaSetDevice failed! Do you have a CUDA-capable GPU installed?");
    goto Error;
  }

  // malloc for deviceOutVol
  cudaStatus = cudaMalloc(&deviceOutVol, sizeof(unsigned char*) * numberOfFrames);
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaMalloc failed for deviceOutVol!");
    goto Error;
  }
  for (int i = 0; i < numberOfFrames; ++i)
  {
    std::vector<unsigned char> zeroVecSlice((size_t)dstRows * (size_t)dstColumns, 0);
    cudaStatus = cudaMalloc(&deviceVecEachSlice[i], zeroVecSlice.size() * sizeof(unsigned char));
    if (cudaStatus != cudaSuccess) {
      fprintf(stderr, "cudaMalloc failed for deviceVecEachSlice[%d]!", i);
      goto Error;
    }
    cudaStatus = cudaMemcpy(deviceVecEachSlice[i], zeroVecSlice.data(), zeroVecSlice.size() * sizeof(unsigned char), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
      fprintf(stderr, "cudaMemcpy failed at host to device for slice %d!", i);
      goto Error;
    }
  }
  cudaStatus = cudaMemcpy(deviceOutVol, deviceVecEachSlice.data(), sizeof(unsigned char*) * numberOfFrames, cudaMemcpyHostToDevice);
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaMemcpy failed for deviceOutVol to device!");
    goto Error;
  }


  // malloc and memcory for srcIndexWiseXVec and srcIndexWiseYVec
  cudaStatus = cudaMalloc(&deviceIndexX, srcIndexWiseXVec.size() * sizeof(double));
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaMalloc failed for deviceIndexX!");
    goto Error;
  }
  cudaStatus = cudaMalloc(&deviceIndexY, srcIndexWiseYVec.size() * sizeof(double));
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaMalloc failed for deviceIndexX!");
    goto Error;
  }
  cudaStatus = cudaMemcpy(deviceIndexX, srcIndexWiseXVec.data(), srcIndexWiseXVec.size() * sizeof(double), cudaMemcpyHostToDevice);
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaMemcpy failed for srcIndexWiseXVec to device!");
    goto Error;
  }
  cudaStatus = cudaMemcpy(deviceIndexY, srcIndexWiseYVec.data(), srcIndexWiseYVec.size() * sizeof(double), cudaMemcpyHostToDevice);
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaMemcpy failed for srcIndexWiseYVec to device!");
    goto Error;
  }

  // malloc and memcory for nrrddata
  cudaStatus = cudaMalloc(&deviceVecVol, sizeof(unsigned char*) * numberOfFrames);
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaMalloc failed for deviceVecVol!");
    goto Error;
  }
  for (int i = 0; i < numberOfFrames; ++i)
  {
    std::vector<unsigned char> vecVolSlice(nrrdData.begin() + (size_t)i * (size_t)srcRows * (size_t)srcColumns, nrrdData.begin() + ((size_t)i + 1) * (size_t)srcRows * (size_t)srcColumns);
    cudaStatus = cudaMalloc(&deviceVecEachSlice[i], vecVolSlice.size() * sizeof(unsigned char));
    if (cudaStatus != cudaSuccess) {
      fprintf(stderr, "cudaMalloc failed for deviceVecEachSlice[%d]!", i);
      goto Error;
    }
    cudaStatus = cudaMemcpy(deviceVecEachSlice[i], vecVolSlice.data(), vecVolSlice.size() * sizeof(unsigned char), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
      fprintf(stderr, "cudaMemcpy failed at host to device for slice %d!", i);
      goto Error;
    }
  }
  cudaStatus = cudaMemcpy(deviceVecVol, deviceVecEachSlice.data(), sizeof(unsigned char*) * numberOfFrames, cudaMemcpyHostToDevice);
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaMemcpy failed for deviceVecVol to device!");
    goto Error;
  }

  // Launch kernel
  dim3 threadsPerBlock(256);
  dim3 numBlocks((outputVecVol.size() + threadsPerBlock.x - 1) / threadsPerBlock.x);
  inverseScanConversionKernels <<< numBlocks, threadsPerBlock >>> (deviceOutVol, deviceVecVol, srcRows, srcColumns, numberOfFrames, dstRows, dstColumns, deviceIndexX, deviceIndexY);

  // Check for kernel launch errors
  cudaStatus = cudaGetLastError();
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "inverseScanConversionKernels launch failed: %s\n", cudaGetErrorString(cudaStatus));
    goto Error;
  }

  // Synchronize kernel and check for errors
  cudaStatus = cudaDeviceSynchronize();
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching averageVectorNKernels!\n", cudaStatus);
    goto Error;
  }

  // Copy output vector from GPU buffer to host memory
  outputVecVol.resize((size_t)dstRows * (size_t)dstColumns * (size_t)numberOfFrames);
  cudaStatus = cudaMemcpy(hostVecEachSlicePtr, deviceOutVol, sizeof(unsigned char*) * numberOfFrames, cudaMemcpyDeviceToHost);
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaMemcpy failed for deviceVecVol to host!");
    goto Error;
  }
  for (int i = 0; i < numberOfFrames; ++i)
  {
    size_t sliceSize = dstRows * dstColumns;
    cudaStatus = cudaMemcpy(outputVecVol.data() + i * sliceSize, hostVecEachSlicePtr[i], sliceSize * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
      fprintf(stderr, "cudaMemcpy failed at device to host for slice %d!", i);
      goto Error;
    }
  }


Error:
  cudaFree(deviceIndexX);
  cudaFree(deviceIndexY);
  for (int i = 0; i < numberOfFrames; ++i) {
    cudaFree(deviceVecEachSlice[i]);
  }
  cudaFree(deviceVecVol);
  cudaFree(deviceOutVol);
  delete[] hostVecEachSlicePtr;
  return cudaStatus;
}

__global__ void bilinearInterpolationKernels
( unsigned char** deviceOutputVecVol,
  unsigned char** deviceVecVol,
  int srcRows,
  int srcCols,
  int numVectors,
  int dstRows,
  int dstCols,
  float* deviceIndexX,
  float* deviceIndexY,
  unsigned char* deviceMaskMeshVecPtr
)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= dstRows * dstCols)
    return;

  int row = (int)deviceIndexY[idx];
  int col = (int)deviceIndexX[idx];
  unsigned char mask = (deviceMaskMeshVecPtr != nullptr) ? deviceMaskMeshVecPtr[idx] : 1;

  if (row < 0 || col < 0 || row >= (srcRows - 1) || col >= (srcCols - 1) || mask == 0)
  {
    for (int vec = 0; vec < numVectors; ++vec)
    {
      deviceOutputVecVol[vec][idx] = 0;
    }
  }
  else
  {
    float row_frac = deviceIndexY[idx] - (float)row;
    float col_frac = deviceIndexX[idx] - (float)col;
    for (int vec = 0; vec < numVectors; ++vec)
    {
      float intensity_topleft = (float)deviceVecVol[vec][row * srcCols + col];
      float intensity_topright = (float)deviceVecVol[vec][row * srcCols + col + 1];
      float intensity_bottomleft = (float)deviceVecVol[vec][(row + 1) * srcCols + col];
      float intensity_bottomright = (float)deviceVecVol[vec][(row + 1) * srcCols + col + 1];
      // Bilinear interpolation
      deviceOutputVecVol[vec][idx] = (int)(((1.0f - row_frac) * ((1.0f - col_frac) * intensity_topleft + col_frac * intensity_topright)
        + row_frac * ((1.0f - col_frac) * intensity_bottomleft + col_frac * intensity_bottomright)) + 0.5f);
    }
  }
}

cudaError_t bilinearInterpolationWithCuda
( std::vector<unsigned char>& outputVecVol,
  std::vector<unsigned char>& nrrdData,
  unsigned int srcRows,
  unsigned int srcColumns,
  unsigned int numberOfFrames,
  unsigned int dstRows,
  unsigned int dstColumns,
  std::vector<float> srcIndexWiseXVec,
  std::vector<float> srcIndexWiseYVec,
  std::vector<unsigned char> hostMaskMeshVec
)
{
  cudaError_t cudaStatus;
  float* deviceIndexX;
  float* deviceIndexY;
  unsigned char* deviceMaskMeshVecPtr = nullptr;
  std::vector<unsigned char*> deviceVecEachSlice(numberOfFrames, nullptr);
  unsigned char** deviceVecVol;
  unsigned char** deviceOutVol;
  unsigned char** hostVecEachSlicePtr = new unsigned char* [numberOfFrames];

  // Select GPU
  cudaStatus = cudaSetDevice(0);

  // malloc for deviceOutVol
  cudaMalloc(&deviceOutVol, sizeof(unsigned char*) * numberOfFrames);
  for (int i = 0; i < numberOfFrames; ++i)
  {
    std::vector<unsigned char> zeroVecSlice((size_t)dstRows * (size_t)dstColumns, 0);
    cudaMalloc(&deviceVecEachSlice[i], zeroVecSlice.size() * sizeof(unsigned char));
    cudaMemcpy(deviceVecEachSlice[i], zeroVecSlice.data(), zeroVecSlice.size() * sizeof(unsigned char), cudaMemcpyHostToDevice);
  }
  cudaMemcpy(deviceOutVol, deviceVecEachSlice.data(), sizeof(unsigned char*) * numberOfFrames, cudaMemcpyHostToDevice);
  // malloc and memcory for srcIndexWiseXVec and srcIndexWiseYVec
  cudaMalloc(&deviceIndexX, srcIndexWiseXVec.size() * sizeof(float));
  cudaMalloc(&deviceIndexY, srcIndexWiseYVec.size() * sizeof(float));

  if (!hostMaskMeshVec.empty())
  {
    cudaMalloc(&deviceMaskMeshVecPtr, hostMaskMeshVec.size() * sizeof(unsigned char));
    cudaMemcpy(deviceMaskMeshVecPtr, hostMaskMeshVec.data(), hostMaskMeshVec.size() * sizeof(unsigned char), cudaMemcpyHostToDevice);
  }
  cudaMemcpy(deviceIndexX, srcIndexWiseXVec.data(), srcIndexWiseXVec.size() * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(deviceIndexY, srcIndexWiseYVec.data(), srcIndexWiseYVec.size() * sizeof(float), cudaMemcpyHostToDevice);

  // malloc and memcory for nrrddata
  cudaMalloc(&deviceVecVol, sizeof(unsigned char*) * numberOfFrames);
  for (int i = 0; i < numberOfFrames; ++i)
  {
    std::vector<unsigned char> vecVolSlice(nrrdData.begin() + (size_t)i * (size_t)srcRows * (size_t)srcColumns, nrrdData.begin() + ((size_t)i + 1) * (size_t)srcRows * (size_t)srcColumns);
    cudaMalloc(&deviceVecEachSlice[i], vecVolSlice.size() * sizeof(unsigned char));
    cudaMemcpy(deviceVecEachSlice[i], vecVolSlice.data(), vecVolSlice.size() * sizeof(unsigned char), cudaMemcpyHostToDevice);
  }
  cudaMemcpy(deviceVecVol, deviceVecEachSlice.data(), sizeof(unsigned char*) * numberOfFrames, cudaMemcpyHostToDevice);

  // Launch kernel
  // std::vector<int> numTreadVec = arangeInt(32, 1024, 32); // pick 832 from 32 - 1024
  int numThreads = 832;
  //auto start_bilinintp = std::chrono::high_resolution_clock::now();
  dim3 threadsPerBlock(numThreads);
  dim3 numBlocks((outputVecVol.size() + threadsPerBlock.x - 1) / threadsPerBlock.x);
  bilinearInterpolationKernels << < numBlocks, threadsPerBlock >> > (deviceOutVol, deviceVecVol, srcRows, srcColumns, numberOfFrames, dstRows, dstColumns, deviceIndexX, deviceIndexY, deviceMaskMeshVecPtr);
  cudaDeviceSynchronize();
  //auto end_bilinintp = std::chrono::high_resolution_clock::now();
  //std::chrono::duration<double, std::milli> duration_bilinintp = end_bilinintp - start_bilinintp;
  //std::cout << "Elapse time for the mask bilinear interpolation(in helper, only kernel), gpu: " << duration_bilinintp.count() << " msec " << " numThreads: "<< numThreads  <<"\n";

  // Copy output vector from GPU buffer to host memory
  outputVecVol.resize((size_t)dstRows * (size_t)dstColumns * (size_t)numberOfFrames);
  cudaMemcpy(hostVecEachSlicePtr, deviceOutVol, sizeof(unsigned char*) * numberOfFrames, cudaMemcpyDeviceToHost);
  for (int i = 0; i < numberOfFrames; ++i)
  {
    size_t sliceSize = dstRows * dstColumns;
    cudaMemcpy(outputVecVol.data() + i * sliceSize, hostVecEachSlicePtr[i], sliceSize * sizeof(unsigned char), cudaMemcpyDeviceToHost);
  }

  cudaFree(deviceIndexX);
  cudaFree(deviceIndexY);
  if (deviceMaskMeshVecPtr != nullptr)
  {
    cudaFree(deviceMaskMeshVecPtr);
  }
  for (int i = 0; i < numberOfFrames; ++i)
  {
    cudaFree(deviceVecEachSlice[i]);
  }
  cudaFree(deviceVecVol);
  cudaFree(deviceOutVol);
  delete[] hostVecEachSlicePtr;
  return cudaStatus;
}

