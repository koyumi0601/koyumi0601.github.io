#include "kernel_bilinearinterpolation.cuh"

/**
 * @brief CUDA kernel: Generates index-wise range and angle meshes based on given ranges and angles.
 *
 * This function calculates the index-wise interpolation position based on the provided range and angle values,
 * and the specified grid dimensions. It is used to quickly determine the interpolation position within a mesh
 * represented by a 2D array.
 * 
 * @param deviceIndexWiseRangeMeshVecPtr: Device pointer to store the range mesh.
 * @param deviceIndexWiseAngleMeshVecPtr: Device pointer to store the angle mesh.
 * @param deviceXPtr: Device pointer to store X coordinates.
 * @param deviceYPtr: Device pointer to store Y coordinates.
 * @param cols: Number of columns in the mesh.
 * @param rows: Number of rows in the mesh.
 * @param firstSrcRangeValue: First source range value.
 * @param lastSrcRangeValue: Last source range value.
 * @param numRanges: Number of ranges.
 * @param firstSrcAngleValue: First source angle value.
 * @param lastSrcAngleValue: Last source angle value.
 * @param numAngles: Number of angles.
 *
 * @return void
 */
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


/**
 * @brief Performs inverse scan conversion to generate output vectors from input vector volumes.
 *
 * This function performs inverse scan conversion to generate output vectors from input vector volumes.
 * It uses bilinear interpolation to calculate the output vector values based on the input vector volumes
 * and the specified interpolation indices.
 *
 * @param deviceOutputVecVol Pointer to the device memory containing the output vector volumes.
 * @param deviceVecVol Pointer to the device memory containing the input vector volumes.
 * @param srcRows Number of rows in the input vector volumes.
 * @param srcCols Number of columns in the input vector volumes.
 * @param numVectors Number of vectors in the input and output volumes.
 * @param dstRows Number of rows in the output vector volumes.
 * @param dstCols Number of columns in the output vector volumes.
 * @param deviceIndexX Pointer to the device memory containing the X interpolation indices.
 * @param deviceIndexY Pointer to the device memory containing the Y interpolation indices.
 */
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

/**
 * @brief Generates index-wise range and angle mesh vectors using CUDA.
 *
 * This function generates index-wise range and angle mesh vectors using CUDA parallel processing.
 * It interpolates the given X and Y coordinates to calculate the range and angle values for each index
 * in the mesh grid defined by X and Y.
 *
 * @param X Vector containing X coordinates.
 * @param Y Vector containing Y coordinates.
 * @param srcRangeA Vector containing source range values.
 * @param srcAngleA Vector containing source angle values.
 * @return A pair of vectors containing index-wise range and angle mesh values.
 * - The first vector contains index-wise range mesh values.
 * - The second vector contains index-wise angle mesh values.
 */
std::pair<std::vector<float>, std::vector<float>> generateIndexWiseRangeAngleMeshVecWithCuda
( std::vector<double> X,
  std::vector<double> Y,
  std::vector<double> srcRangeA,
  std::vector<double> srcAngleA
)
{
  // Initialize variables
  int cols = X.size();
  int rows = Y.size();
  int numSamplesPerFrame = cols * rows;
  int numRanges = srcRangeA.size();
  int numAngles = srcAngleA.size();
  float firstSrcRangeValue = (float) srcRangeA[0];
  float lastSrcRangeValue = (float) srcRangeA[srcRangeA.size() - 1];
  float firstSrcAngleValue = (float) srcAngleA[0];
  float lastSrcAngleValue = (float) srcAngleA[srcAngleA.size() - 1];
  // Allocate memory on the device
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
  // Configure CUDA kernel parameters
  int numThreads = 256; // 256 from 32 - 1024
  auto start = std::chrono::high_resolution_clock::now();
  int threadPerBlock = numThreads;
  dim3 blockDim(threadPerBlock, 1, 1);
  dim3 blockPerGrid = (numSamplesPerFrame + threadPerBlock - 1) / threadPerBlock;
  // Execute CUDA kernel
  generateIndexWiseRangeAngleMeshVecKernel << <blockPerGrid, blockDim >> > (deviceIndexWiseRangeMeshVecPtr, deviceIndexWiseAngleMeshVecPtr, deviceXPtr, deviceYPtr, cols, rows, firstSrcRangeValue, lastSrcRangeValue, numRanges, firstSrcAngleValue, lastSrcAngleValue, numAngles);
  cudaDeviceSynchronize();
  // Copy results back to host memory
  cudaMemcpy(hostIndexWiseRangeMeshVec.data(), deviceIndexWiseRangeMeshVecPtr, numSamplesPerFrame * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(hostIndexWiseAngleMeshVec.data(), deviceIndexWiseAngleMeshVecPtr, numSamplesPerFrame * sizeof(float), cudaMemcpyDeviceToHost);
  // Free device memory
  cudaFree(deviceXPtr);
  cudaFree(deviceYPtr);
  cudaFree(deviceIndexWiseRangeMeshVecPtr);
  cudaFree(deviceIndexWiseAngleMeshVecPtr);
  // Return the generated index-wise range and angle mesh vectors
  return std::make_pair(hostIndexWiseRangeMeshVec, hostIndexWiseAngleMeshVec);
}

/**
 * @brief Performs inverse scan conversion using CUDA.
 *
 * This function performs inverse scan conversion using CUDA parallel processing.
 * It converts a given NRRD data into output volume data based on the provided index-wise X and Y vectors.
 *
 * @param outputVecVol Output vector volume where the result will be stored.
 * @param nrrdData Input NRRD data.
 * @param srcRows Number of rows in the source data.
 * @param srcColumns Number of columns in the source data.
 * @param numberOfFrames Number of frames in the source data.
 * @param dstRows Number of rows in the destination volume.
 * @param dstColumns Number of columns in the destination volume.
 * @param srcIndexWiseXVec Vector containing index-wise X coordinates.
 * @param srcIndexWiseYVec Vector containing index-wise Y coordinates.
 * @return CUDA error status.
 */
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

/**
 * @brief Performs bilinear interpolation using CUDA.
 *
 * This kernel function performs bilinear interpolation on the device.
 * It interpolates values from the input volume to the output volume using bilinear interpolation method.
 *
 * @param deviceOutputVecVol Output vector volume where the result will be stored.
 * @param deviceVecVol Input vector volume.
 * @param srcRows Number of rows in the source data.
 * @param srcCols Number of columns in the source data.
 * @param numVectors Number of vectors in the input volume.
 * @param dstRows Number of rows in the destination volume.
 * @param dstCols Number of columns in the destination volume.
 * @param deviceIndexX Vector containing index-wise X coordinates on the device.
 * @param deviceIndexY Vector containing index-wise Y coordinates on the device.
 * @param deviceMaskMeshVecPtr Pointer to the mask mesh vector.
 */
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

/**
 * @brief Performs bilinear interpolation on the GPU using CUDA.
 *
 * This function performs bilinear interpolation on the GPU using CUDA.
 * It interpolates values from the input volume to the output volume using bilinear interpolation method.
 *
 * @param outputVecVol Output vector volume where the interpolated result will be stored.
 * @param nrrdData Input vector volume.
 * @param srcRows Number of rows in the source data.
 * @param srcColumns Number of columns in the source data.
 * @param numberOfFrames Number of frames in the input volume.
 * @param dstRows Number of rows in the destination volume.
 * @param dstColumns Number of columns in the destination volume.
 * @param srcIndexWiseXVec Vector containing index-wise X coordinates on the host.
 * @param srcIndexWiseYVec Vector containing index-wise Y coordinates on the host.
 * @param hostMaskMeshVec Mask mesh vector on the host.
 * @return cudaError_t Returns the status of CUDA operations.
 */
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
  cudaMalloc(&deviceIndexX, srcIndexWiseXVec.size() * sizeof(float));
  cudaMalloc(&deviceIndexY, srcIndexWiseYVec.size() * sizeof(float));

  if (!hostMaskMeshVec.empty())
  {
    cudaMalloc(&deviceMaskMeshVecPtr, hostMaskMeshVec.size() * sizeof(unsigned char));
    cudaMemcpy(deviceMaskMeshVecPtr, hostMaskMeshVec.data(), hostMaskMeshVec.size() * sizeof(unsigned char), cudaMemcpyHostToDevice);
  }
  cudaMemcpy(deviceIndexX, srcIndexWiseXVec.data(), srcIndexWiseXVec.size() * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(deviceIndexY, srcIndexWiseYVec.data(), srcIndexWiseYVec.size() * sizeof(float), cudaMemcpyHostToDevice);
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