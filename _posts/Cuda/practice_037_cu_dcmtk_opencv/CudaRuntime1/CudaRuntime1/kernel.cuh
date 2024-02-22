#ifndef KERNEL_CUH_
#define KERNEL_CUH_

#include <vector>
#include <cuda_runtime.h>

__global__ void averageVectorForKernels(unsigned char* deviceOutputPlane, unsigned char** deviceVecVolSlices, int numElements, int numVectors);

// Wrapper function declaration
cudaError_t averageVectorForWithCuda(std::vector<unsigned char>& outputPlane, std::vector<unsigned char>& vecVol, unsigned int dim1Size, unsigned int dim2Size, unsigned int dim3Size);

#endif  // KERNEL_CUH_