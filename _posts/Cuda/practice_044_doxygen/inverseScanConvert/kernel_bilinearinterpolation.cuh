// kernel.cuh
#ifndef KERNEL_CUH
#define KERNEL_CUH

#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <vector>
#include <chrono>
#include <iostream>
#define PI 3.14159265358979323846

__global__ void inverseScanConversionKernels
( unsigned char** deviceOutputVecVol,
  unsigned char** deviceVecVolSlices,
  int numRows,
  int numCols,
  int numVectors,
  int dstRows,
  int dstColumns,
  double* deviceIndexX,
  double* deviceIndexY
);

cudaError_t inverseScanConversionWithCuda
( std::vector<unsigned char>& outputVecVol,
  std::vector<unsigned char>& nrrdData,
  unsigned int rows,
  unsigned int columns,
  unsigned int numberOfFrames,
  unsigned int dstRows,
  unsigned int dstColumns,
  std::vector<double> srcIndexWiseXVec,
  std::vector<double> srcIndexWiseYVec
);

__global__ void bilinearInterpolationKernels
( unsigned char** deviceOutputVecVol,
  unsigned char** deviceVecVolSlices,
  int numRows,
  int numCols,
  int numVectors,
  int dstRows,
  int dstColumns,
  float* deviceIndexX,
  float* deviceIndexY,
  unsigned char* deviceMaskMeshVecPtr
);

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
  std::vector<unsigned char> hostMaskMeshVec = std::vector<unsigned char>()
);

__global__ void generateIndexWiseRangeAngleMeshVecKernel
( float* deviceIndexWiseRangeMeshVecPtr,
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
);

std::pair<std::vector<float>, std::vector<float>> generateIndexWiseRangeAngleMeshVecWithCuda
( std::vector<double> X,
  std::vector<double> Y,
  std::vector<double> srcRangeA,
  std::vector<double> srcAngleA
);

#endif // KERNEL_CUH

