#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <vector>
#include <iostream>
#include "dcmtk/config/osconfig.h"
#include "dcmtk/dcmdata/dctk.h"
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>

cudaError_t averageVectorWithCuda(std::vector<unsigned char>& outputPlane, std::vector<unsigned char>& vecVol, unsigned int dim1Size, unsigned int dim2Size, unsigned int dim3Size);

__global__ void averageVector15Kernel(unsigned char* deviceOutputPlane,
  unsigned char* deviceVecVol01,
  unsigned char* deviceVecVol02,
  unsigned char* deviceVecVol03,
  unsigned char* deviceVecVol04,
  unsigned char* deviceVecVol05,
  unsigned char* deviceVecVol06,
  unsigned char* deviceVecVol07,
  unsigned char* deviceVecVol08,
  unsigned char* deviceVecVol09,
  unsigned char* deviceVecVol10,
  unsigned char* deviceVecVol11,
  unsigned char* deviceVecVol12,
  unsigned char* deviceVecVol13,
  unsigned char* deviceVecVol14,
  unsigned char* deviceVecVol15,
  int numElements)
{
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx >= numElements)
    return;
  int sum = deviceVecVol01[idx] +
    deviceVecVol02[idx] +
    deviceVecVol03[idx] +
    deviceVecVol04[idx] +
    deviceVecVol05[idx] +
    deviceVecVol06[idx] +
    deviceVecVol07[idx] +
    deviceVecVol08[idx] +
    deviceVecVol09[idx] +
    deviceVecVol10[idx] +
    deviceVecVol11[idx] +
    deviceVecVol12[idx] +
    deviceVecVol13[idx] +
    deviceVecVol14[idx] +
    deviceVecVol15[idx];
  deviceOutputPlane[idx] = static_cast<unsigned char>(sum / 15);
}

__global__ void averageVector27Kernel(unsigned char* deviceOutputPlane,
  unsigned char* deviceVecVol01,
  unsigned char* deviceVecVol02,
  unsigned char* deviceVecVol03,
  unsigned char* deviceVecVol04,
  unsigned char* deviceVecVol05,
  unsigned char* deviceVecVol06,
  unsigned char* deviceVecVol07,
  unsigned char* deviceVecVol08,
  unsigned char* deviceVecVol09,
  unsigned char* deviceVecVol10,
  unsigned char* deviceVecVol11,
  unsigned char* deviceVecVol12,
  unsigned char* deviceVecVol13,
  unsigned char* deviceVecVol14,
  unsigned char* deviceVecVol15,
  unsigned char* deviceVecVol16,
  unsigned char* deviceVecVol17,
  unsigned char* deviceVecVol18,
  unsigned char* deviceVecVol19,
  unsigned char* deviceVecVol20,
  unsigned char* deviceVecVol21,
  unsigned char* deviceVecVol22,
  unsigned char* deviceVecVol23,
  unsigned char* deviceVecVol24,
  unsigned char* deviceVecVol25,
  unsigned char* deviceVecVol26,
  unsigned char* deviceVecVol27,
  int numElements)
{
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx >= numElements)
    return;
  int sum = deviceVecVol01[idx] +
    deviceVecVol02[idx] +
    deviceVecVol03[idx] +
    deviceVecVol04[idx] +
    deviceVecVol05[idx] +
    deviceVecVol06[idx] +
    deviceVecVol07[idx] +
    deviceVecVol08[idx] +
    deviceVecVol09[idx] +
    deviceVecVol10[idx] +
    deviceVecVol11[idx] +
    deviceVecVol12[idx] +
    deviceVecVol13[idx] +
    deviceVecVol14[idx] +
    deviceVecVol15[idx] +
    deviceVecVol16[idx] +
    deviceVecVol17[idx] +
    deviceVecVol18[idx] +
    deviceVecVol19[idx] +
    deviceVecVol20[idx] +
    deviceVecVol21[idx] +
    deviceVecVol22[idx] +
    deviceVecVol23[idx] +
    deviceVecVol24[idx] +
    deviceVecVol25[idx] +
    deviceVecVol26[idx] +
    deviceVecVol27[idx];
  deviceOutputPlane[idx] = static_cast<unsigned char>(sum / 27);
}


DcmDataset* OpenDICOMAndGetDataset(DcmFileFormat& fileformat, std::string& filePath)
{
  OFCondition status = fileformat.loadFile(filePath.c_str());
  if (!status.good())
  {
    std::cout << "Load Dimcom File Error: " << status.text() << std::endl;
    return nullptr;
  }
  DcmDataset* dataset = fileformat.getDataset();
  return dataset;
}


bool ReadPatientName(DcmFileFormat& fileformat, std::string& filePath)
{
  OFCondition status = fileformat.loadFile(filePath.c_str());
  if (!status.good())
  {
    std::cout << "Load Dimcom File Error: " << status.text() << std::endl;
    return false;
  }

  OFString PatientName;
  status = fileformat.getDataset()->findAndGetOFString(DCM_PatientName, PatientName);
  if (status.good())
  {
    std::cout << "Get PatientName:" << PatientName << std::endl;
  }
  else
  {
    std::cout << "Get PatientName Error:" << status.text() << std::endl;
    return false;
  }
  return true;
}


bool SavePatientName(DcmFileFormat& fileformat, std::string& filePath, const std::string& info)
{
  OFCondition status = fileformat.getDataset()->putAndInsertString(DCM_PatientName, info.c_str());
  if (status.good())
  {
    std::cout << "Save PatientName:" << info.c_str() << std::endl;
  }
  else
  {
    std::cout << "Save PatientName Error: " << status.text() << std::endl;
    return false;
  }

  status = fileformat.saveFile(filePath.c_str());
  if (!status.good())
  {
    std::cout << "Save Dimcom File Error: " << status.text() << std::endl;
    return false;
  }
  return true;
}



int main()
{
  // Load DICOM file
  //std::string dicomFile = "D:/02.project_bitbucket/test_cuda/test_cuda/resource/1.2.276.0.7230010.3.1.4.301774566.4480.1683581601.503";
  std::string dicomFile = "D:/Github_Blog/koyumi0601.github.io/_posts/Cuda/practice_037_cu_dcmtk_opencv/CudaRuntime1/x64/Debug/resource/1.2.276.0.7230010.3.1.4.301774566.4480.1683581601.503";
  //std::string dicomFile = "D:/02.project_bitbucket/test_cuda/test_cuda/resource/1.2.276.0.7230010.3.1.4.301774566.4480.1683583137.517";

  // Initialize DCMTK and open the DICOM
  DcmFileFormat dicomFileformat;
  DcmDataset* dicomDataset = OpenDICOMAndGetDataset(dicomFileformat, dicomFile);

  Uint16 rows;
  Uint16 columns;
  Sint32 numberOfFrames;
  Float64 sliceSpacing;
  Float64 rowSpacing, colSpacing;

  bool IsAvailableToReadRowColFrame = (dicomDataset->findAndGetUint16(DcmTagKey(0x0028, 0x0010), rows).good() &&
    dicomDataset->findAndGetUint16(DcmTagKey(0x0028, 0x0011), columns).good() &&
    dicomDataset->findAndGetSint32(DcmTagKey(0x0028, 0x0008), numberOfFrames).good());
  bool IsAvailableToReadPixelSpacing = (dicomDataset->findAndGetFloat64(DcmTagKey(0x0028, 0x0030), rowSpacing, 0).good() &&
    dicomDataset->findAndGetFloat64(DcmTagKey(0x0028, 0x0030), colSpacing, 1).good() &&
    dicomDataset->findAndGetFloat64(DcmTagKey(0x0018, 0x0088), sliceSpacing).good());

  const Uint8* pixelData = NULL;
  unsigned long pixelDataLength = 0;
  if (IsAvailableToReadRowColFrame && IsAvailableToReadPixelSpacing)
  {
    if (dicomDataset->findAndGetUint8Array(DCM_PixelData, pixelData, &pixelDataLength).good())
    {
      //for (int frameIndex = 0; frameIndex < numberOfFrames; ++frameIndex)
      //{
      //    // Calculate the starting index of the pixel data for the current frame
      //    const Uint8* framePixelData = pixelData + frameIndex * rows * columns * sizeof(Uint8);
      //    // Create cv::Mat from DICOM pixel data for the current frame
      //    cv::Mat dicomFrame(rows, columns, CV_8U, (void*)framePixelData);
      //    cv::imshow("DICOM Frame", dicomFrame);
      //    cv::waitKey(30);
      //}
      //cv::destroyWindow("DICOM Frame");
    }
    else {
      std::cerr << "Failed to get PixelData" << std::endl;
      return 1;
    }
  }
  std::cout << "rows=" << rows << ", columns=" << columns << ", numberOfFrames=" << numberOfFrames << std::endl;
  std::cout << "rowSpacing=" << rowSpacing << ", colSpacing=" << colSpacing << ", sliceSpacing=" << sliceSpacing << "\n" << std::endl;

  float targetSlabThicknessMm = 2.0f;
  int desiredNumCoronalSlice = (int)(targetSlabThicknessMm / (float)rowSpacing + 0.5f);
  float columnLengthMm = (float)columns * colSpacing;
  float sliceLengthMm = (float)numberOfFrames * sliceSpacing;
  int resizedColumnSize, resizedSliceSize;
  if (colSpacing < sliceSpacing)
  {
    resizedColumnSize = (int)((columnLengthMm / colSpacing) + 0.5f);
    resizedSliceSize = (int)((sliceLengthMm / colSpacing) + 0.5f);
  }
  else
  {
    resizedColumnSize = (int)((columnLengthMm / sliceSpacing) + 0.5f);
    resizedSliceSize = (int)((sliceLengthMm / sliceSpacing) + 0.5f);
  }

  std::cout << desiredNumCoronalSlice << std::endl;
  std::cout << "column Length=" << columnLengthMm << ", slice Length=" << sliceLengthMm << std::endl;
  std::cout << "resizedColumnSize=" << resizedColumnSize << ", resizedSliceSize=" << resizedSliceSize << "\n" << std::endl;


  // GPU spec
  cudaDeviceProp prop;
  int device;
  cudaGetDevice(&device);
  cudaGetDeviceProperties(&prop, device);
  printf("Max Threads per Block: %d\n", prop.maxThreadsPerBlock);
  printf("Max Threads per Multiprocessor: %d\n", prop.maxThreadsPerMultiProcessor);

  // coronal mean by CUDA
  const int numRows = (int)rows;
  const int numCols = (int)columns;
  const int numSlices = (int)numberOfFrames;

  // Resize the vector to hold the pixel data
  std::vector<unsigned char> vecVol;
  vecVol.resize(pixelDataLength);
  std::memcpy(&vecVol[0], pixelData, pixelDataLength);

  // review transverse view generation
  int targetSliceIdx = 0;
  cv::Mat tSlice((int)rows, (int)columns, CV_8U);
  std::memcpy(tSlice.data, &vecVol[targetSliceIdx * rows * columns * sizeof(Uint8)], (int)rows * columns);
  cv::imshow("DICOM transverse view", tSlice);
  cv::waitKey(0);
  // coronal view data generation
  std::vector<unsigned char> vecCoronal;
  vecCoronal.resize(numCols * numSlices);
  int targetCoronalIndex = 2;
  for (int col = 0; col < numCols; ++col)
  {
    for (int slc = 0; slc < numSlices; ++slc)
    {
      int index = slc * (numRows * numCols) + targetCoronalIndex * (numCols)+col;
      int coronalIndex = slc * (numCols)+col;
      vecCoronal[coronalIndex] = vecVol[index];
    }
  }
  cv::Mat cSlice(numSlices, numCols, CV_8U);
  std::memcpy(cSlice.data, &vecCoronal[0], numCols * numSlices);
  cv::imshow("DICOM coronal view", cSlice);
  cv::waitKey(0);
  // sagittal view data generation
  std::vector<unsigned char> vecSagittal;
  vecSagittal.resize(numRows * numSlices);
  int targetSagittalIndex = 440;
  for (int row = 0; row < numRows; ++row)
  {
    for (int slc = 0; slc < numSlices; ++slc)
    {
      int index = slc * (numRows * numCols) + row * (numCols)+targetSagittalIndex;
      int sagittalIndex = slc * (numRows)+row;

      vecSagittal[sagittalIndex] = vecVol[index];
    }
  }
  cv::Mat sSlice(numSlices, numRows, CV_8U);
  std::memcpy(sSlice.data, &vecSagittal[0], numRows * numSlices);
  cv::imshow("DICOM sagittal view", sSlice);
  cv::waitKey(0);

  //// test data
  //int rowModVal = 15;
  //std::vector<unsigned char> vecVol;
  //vecVol.resize((size_t) numRows * numCols * numSlices);
  //for (int row = 0; row < numRows; ++row)
  //{
  //    for (int col = 0; col < numCols; ++col)
  //    {
  //        for (int slc = 0; slc < numSlices; ++slc)
  //        {
  //            size_t index = (size_t)slc + (size_t)numSlices * ((size_t)col + (size_t)row * numCols);
  //            vecVol[index] = (unsigned char)row % rowModVal;
  //        }
  //    }
  //}
  //printf("vecVol[%d] = %d\n", 0, vecVol[0]);

  std::vector<unsigned char> tmpVecVol;
  tmpVecVol.resize(numCols * numSlices * desiredNumCoronalSlice);
  std::vector<unsigned char> outputPlane;
  outputPlane.resize((size_t)numCols * numSlices);
  cv::Mat averagedCSlice(numSlices, numCols, CV_8U);
  cv::Mat resizedCSlice(resizedSliceSize, resizedColumnSize, CV_8U);
  for (int slabIndex = 0; slabIndex < (int)(numRows / desiredNumCoronalSlice); ++slabIndex)
  {
    for (int row = slabIndex * desiredNumCoronalSlice; row < (slabIndex + 1) * desiredNumCoronalSlice; ++row)
    {
      for (int col = 0; col < numCols; ++col)
      {
        for (int slc = 0; slc < numSlices; ++slc)
        {
          int index = slc * (numRows * numCols) + row * (numCols)+col;
          int newIndex = (row - (slabIndex * desiredNumCoronalSlice)) * (numCols * numSlices) + slc * (numCols)+col;
          tmpVecVol[newIndex] = vecVol[index];
        }
      }
    }
    //std::memcpy(testSlice.data, &tmpVecVol[0], numCols * numSlices * sizeof(unsigned char));
    //cv::imshow("DICOM test view", testSlice);
    //cv::waitKey(0);

    // do making slab
    //cudaError_t cudaStatus = averageVectorForWithCuda(outputPlane, vecVol, numRows, numCols, numSlices);
    //if (cudaStatus != cudaSuccess) {
    //    fprintf(stderr, "\naddVectorWithCuda failed!");
    //    return 1;
    //}
    cudaError_t cudastatus = averageVectorWithCuda(outputPlane, tmpVecVol, numCols, numSlices, desiredNumCoronalSlice);
    if (cudastatus != cudaSuccess) {
      fprintf(stderr, "\naverageVectorWithCuda failed!");
      return 1;
    }
    std::memcpy(averagedCSlice.data, &outputPlane[0], numCols * numSlices);
    std::cout << averagedCSlice.rows << ", " << averagedCSlice.cols << std::endl;
    std::cout << resizedCSlice.rows << ", " << resizedCSlice.cols << std::endl;
    //cv::imshow("DICOM filtered coronal view", averagedCSlice);
    //cv::resize(averagedCSlice, resizedCSlice, cv::Size(resizedSliceSize, resizedColumnSize), 0, 0, cv::INTER_LINEAR);
    cv::resize(averagedCSlice, resizedCSlice, resizedCSlice.size(), 0, 0, cv::INTER_LINEAR);
    cv::imshow("DICOM filtered coronal view", resizedCSlice);
    cv::waitKey(0);
  }
  return 0;
}



//// wrapper function for using CUDA to add vectors in parallel
//cudaError_t averageVectorForWithCuda(std::vector<unsigned char>& outputPlane, std::vector<unsigned char>& vecVol, unsigned int dim1Size, unsigned int dim2Size, unsigned int dim3Size)
//{
//    // init variables
//    cudaError_t cudaStatus;
//    unsigned char* deviceOutputPlane;
//    unsigned char* deviceVecVol;
//
//    // Choose which GPU to run on, change this on a multi-GPU system.
//    cudaStatus = cudaSetDevice(0);
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
//        cudaFree(deviceOutputPlane);
//        cudaFree(deviceVecVol);
//        return cudaStatus;
//    }
//
//    // Allocate GPU buffers.
//    cudaStatus = cudaMalloc(&deviceOutputPlane, (size_t)outputPlane.size() * sizeof(unsigned char));
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMalloc failed for deviceOutputPlane!");
//        cudaFree(deviceOutputPlane);
//        cudaFree(deviceVecVol);
//        return cudaStatus;
//    }
//    cudaStatus = cudaMalloc(&deviceVecVol, (size_t)vecVol.size() * sizeof(unsigned char));
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMalloc failed for deviceVecVol!");
//        cudaFree(deviceOutputPlane);
//        cudaFree(deviceVecVol);
//        return cudaStatus;
//    }
//
//    // Copy input vectors from host memory to GPU buffers.
//    cudaStatus = cudaMemcpy(deviceVecVol, vecVol.data(), (size_t)vecVol.size() * sizeof(unsigned char), cudaMemcpyHostToDevice);
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMemcpy failed at cudaMemcpyHostToDevice for deviceVecVol!");
//        cudaFree(deviceOutputPlane);
//        cudaFree(deviceVecVol);
//        return cudaStatus;
//    }
//
//    // Launch a kernel on the GPU with one thread for each element.
//    dim3 threadsPerBlock(256);
//    dim3 numBlocks((outputPlane.size() + threadsPerBlock.x - 1) / threadsPerBlock.x);
//    averageVector15ForKernel <<< numBlocks, threadsPerBlock >>> (deviceOutputPlane, deviceVecVol, dim1Size * dim2Size);
//
//    // Check for any errors launching the kernel
//    cudaStatus = cudaGetLastError();
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "averageKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
//        cudaFree(deviceOutputPlane);
//        cudaFree(deviceVecVol);
//        return cudaStatus;
//    }
//
//    // cudaDeviceSynchronize waits for the kernel to finish, and returns any errors encountered during the launch.
//    cudaStatus = cudaDeviceSynchronize();
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching averageKernel!\n", cudaStatus);
//        cudaFree(deviceOutputPlane);
//        cudaFree(deviceVecVol);
//        return cudaStatus;
//    }
//
//    // Copy output vector from GPU buffer to host memory.
//    outputPlane.resize(dim1Size * dim2Size);
//    cudaStatus = cudaMemcpy(outputPlane.data(), deviceOutputPlane, dim1Size * dim2Size * sizeof(unsigned char), cudaMemcpyDeviceToHost);
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMemcpy failed at cudaMemcpyDeviceToHost!");
//        cudaFree(deviceOutputPlane);
//        cudaFree(deviceVecVol);
//        return cudaStatus;
//    }
//
//    // finish
//    cudaFree(deviceOutputPlane);
//    cudaFree(deviceVecVol);
//    return cudaStatus;
//}


// wrapper function for using CUDA to add vectors in parallel
cudaError_t averageVectorWithCuda(std::vector<unsigned char>& outputPlane, std::vector<unsigned char>& vecVol, unsigned int dim1Size, unsigned int dim2Size, unsigned int dim3Size)
{
  unsigned char* deviceOutputPlane;
  unsigned char* deviceVecVolSlice01;
  unsigned char* deviceVecVolSlice02;
  unsigned char* deviceVecVolSlice03;
  unsigned char* deviceVecVolSlice04;
  unsigned char* deviceVecVolSlice05;
  unsigned char* deviceVecVolSlice06;
  unsigned char* deviceVecVolSlice07;
  unsigned char* deviceVecVolSlice08;
  unsigned char* deviceVecVolSlice09;
  unsigned char* deviceVecVolSlice10;
  unsigned char* deviceVecVolSlice11;
  unsigned char* deviceVecVolSlice12;
  unsigned char* deviceVecVolSlice13;
  unsigned char* deviceVecVolSlice14;
  unsigned char* deviceVecVolSlice15;
  unsigned char* deviceVecVolSlice16;
  unsigned char* deviceVecVolSlice17;
  unsigned char* deviceVecVolSlice18;
  unsigned char* deviceVecVolSlice19;
  unsigned char* deviceVecVolSlice20;
  unsigned char* deviceVecVolSlice21;
  unsigned char* deviceVecVolSlice22;
  unsigned char* deviceVecVolSlice23;
  unsigned char* deviceVecVolSlice24;
  unsigned char* deviceVecVolSlice25;
  unsigned char* deviceVecVolSlice26;
  unsigned char* deviceVecVolSlice27;

  cudaError_t cudaStatus;

  std::vector<unsigned char> vecVolSlice01(vecVol.begin() + 0 * dim1Size * dim2Size, vecVol.begin() + 1 * dim1Size * dim2Size);
  std::vector<unsigned char> vecVolSlice02(vecVol.begin() + 1 * dim1Size * dim2Size, vecVol.begin() + 2 * dim1Size * dim2Size);
  std::vector<unsigned char> vecVolSlice03(vecVol.begin() + 2 * dim1Size * dim2Size, vecVol.begin() + 3 * dim1Size * dim2Size);
  std::vector<unsigned char> vecVolSlice04(vecVol.begin() + 3 * dim1Size * dim2Size, vecVol.begin() + 4 * dim1Size * dim2Size);
  std::vector<unsigned char> vecVolSlice05(vecVol.begin() + 4 * dim1Size * dim2Size, vecVol.begin() + 5 * dim1Size * dim2Size);
  std::vector<unsigned char> vecVolSlice06(vecVol.begin() + 5 * dim1Size * dim2Size, vecVol.begin() + 6 * dim1Size * dim2Size);
  std::vector<unsigned char> vecVolSlice07(vecVol.begin() + 6 * dim1Size * dim2Size, vecVol.begin() + 7 * dim1Size * dim2Size);
  std::vector<unsigned char> vecVolSlice08(vecVol.begin() + 7 * dim1Size * dim2Size, vecVol.begin() + 8 * dim1Size * dim2Size);
  std::vector<unsigned char> vecVolSlice09(vecVol.begin() + 8 * dim1Size * dim2Size, vecVol.begin() + 9 * dim1Size * dim2Size);
  std::vector<unsigned char> vecVolSlice10(vecVol.begin() + 9 * dim1Size * dim2Size, vecVol.begin() + 10 * dim1Size * dim2Size);
  std::vector<unsigned char> vecVolSlice11(vecVol.begin() + 10 * dim1Size * dim2Size, vecVol.begin() + 11 * dim1Size * dim2Size);
  std::vector<unsigned char> vecVolSlice12(vecVol.begin() + 11 * dim1Size * dim2Size, vecVol.begin() + 12 * dim1Size * dim2Size);
  std::vector<unsigned char> vecVolSlice13(vecVol.begin() + 12 * dim1Size * dim2Size, vecVol.begin() + 13 * dim1Size * dim2Size);
  std::vector<unsigned char> vecVolSlice14(vecVol.begin() + 13 * dim1Size * dim2Size, vecVol.begin() + 14 * dim1Size * dim2Size);
  std::vector<unsigned char> vecVolSlice15(vecVol.begin() + 14 * dim1Size * dim2Size, vecVol.begin() + 15 * dim1Size * dim2Size);
  std::vector<unsigned char> vecVolSlice16(vecVol.begin() + 15 * dim1Size * dim2Size, vecVol.begin() + 16 * dim1Size * dim2Size);
  std::vector<unsigned char> vecVolSlice17(vecVol.begin() + 16 * dim1Size * dim2Size, vecVol.begin() + 17 * dim1Size * dim2Size);
  std::vector<unsigned char> vecVolSlice18(vecVol.begin() + 17 * dim1Size * dim2Size, vecVol.begin() + 18 * dim1Size * dim2Size);
  std::vector<unsigned char> vecVolSlice19(vecVol.begin() + 18 * dim1Size * dim2Size, vecVol.begin() + 19 * dim1Size * dim2Size);
  std::vector<unsigned char> vecVolSlice20(vecVol.begin() + 19 * dim1Size * dim2Size, vecVol.begin() + 20 * dim1Size * dim2Size);
  std::vector<unsigned char> vecVolSlice21(vecVol.begin() + 20 * dim1Size * dim2Size, vecVol.begin() + 21 * dim1Size * dim2Size);
  std::vector<unsigned char> vecVolSlice22(vecVol.begin() + 21 * dim1Size * dim2Size, vecVol.begin() + 22 * dim1Size * dim2Size);
  std::vector<unsigned char> vecVolSlice23(vecVol.begin() + 22 * dim1Size * dim2Size, vecVol.begin() + 23 * dim1Size * dim2Size);
  std::vector<unsigned char> vecVolSlice24(vecVol.begin() + 23 * dim1Size * dim2Size, vecVol.begin() + 24 * dim1Size * dim2Size);
  std::vector<unsigned char> vecVolSlice25(vecVol.begin() + 24 * dim1Size * dim2Size, vecVol.begin() + 25 * dim1Size * dim2Size);
  std::vector<unsigned char> vecVolSlice26(vecVol.begin() + 25 * dim1Size * dim2Size, vecVol.begin() + 26 * dim1Size * dim2Size);
  std::vector<unsigned char> vecVolSlice27(vecVol.begin() + 26 * dim1Size * dim2Size, vecVol.begin() + 27 * dim1Size * dim2Size);


  // Choose which GPU to run on, change this on a multi-GPU system.
  cudaStatus = cudaSetDevice(0);
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
    goto Error;
  }

  // Allocate GPU buffers.
  cudaStatus = cudaMalloc(&deviceOutputPlane, (size_t)outputPlane.size() * sizeof(unsigned char));
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaMalloc failed!");
    goto Error;
  }

  cudaStatus = cudaMalloc(&deviceVecVolSlice01, (size_t)vecVolSlice01.size() * sizeof(unsigned char));
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaMalloc failed!");
    goto Error;
  }
  cudaStatus = cudaMalloc(&deviceVecVolSlice02, (size_t)vecVolSlice02.size() * sizeof(unsigned char));
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaMalloc failed!");
    goto Error;
  }
  cudaStatus = cudaMalloc(&deviceVecVolSlice03, (size_t)vecVolSlice03.size() * sizeof(unsigned char));
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaMalloc failed!");
    goto Error;
  }
  cudaStatus = cudaMalloc(&deviceVecVolSlice04, (size_t)vecVolSlice04.size() * sizeof(unsigned char));
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaMalloc failed!");
    goto Error;
  }
  cudaStatus = cudaMalloc(&deviceVecVolSlice05, (size_t)vecVolSlice05.size() * sizeof(unsigned char));
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaMalloc failed!");
    goto Error;
  }
  cudaStatus = cudaMalloc(&deviceVecVolSlice06, (size_t)vecVolSlice06.size() * sizeof(unsigned char));
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaMalloc failed!");
    goto Error;
  }
  cudaStatus = cudaMalloc(&deviceVecVolSlice07, (size_t)vecVolSlice07.size() * sizeof(unsigned char));
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaMalloc failed!");
    goto Error;
  }
  cudaStatus = cudaMalloc(&deviceVecVolSlice08, (size_t)vecVolSlice08.size() * sizeof(unsigned char));
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaMalloc failed!");
    goto Error;
  }
  cudaStatus = cudaMalloc(&deviceVecVolSlice09, (size_t)vecVolSlice09.size() * sizeof(unsigned char));
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaMalloc failed!");
    goto Error;
  }
  cudaStatus = cudaMalloc(&deviceVecVolSlice10, (size_t)vecVolSlice10.size() * sizeof(unsigned char));
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaMalloc failed!");
    goto Error;
  }
  cudaStatus = cudaMalloc(&deviceVecVolSlice11, (size_t)vecVolSlice11.size() * sizeof(unsigned char));
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaMalloc failed!");
    goto Error;
  }
  cudaStatus = cudaMalloc(&deviceVecVolSlice12, (size_t)vecVolSlice12.size() * sizeof(unsigned char));
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaMalloc failed!");
    goto Error;
  }
  cudaStatus = cudaMalloc(&deviceVecVolSlice13, (size_t)vecVolSlice13.size() * sizeof(unsigned char));
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaMalloc failed!");
    goto Error;
  }
  cudaStatus = cudaMalloc(&deviceVecVolSlice14, (size_t)vecVolSlice14.size() * sizeof(unsigned char));
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaMalloc failed!");
    goto Error;
  }
  cudaStatus = cudaMalloc(&deviceVecVolSlice15, (size_t)vecVolSlice15.size() * sizeof(unsigned char));
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaMalloc failed!");
    goto Error;
  }
  cudaStatus = cudaMalloc(&deviceVecVolSlice16, (size_t)vecVolSlice16.size() * sizeof(unsigned char));
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaMalloc failed!");
    goto Error;
  }
  cudaStatus = cudaMalloc(&deviceVecVolSlice17, (size_t)vecVolSlice17.size() * sizeof(unsigned char));
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaMalloc failed!");
    goto Error;
  }
  cudaStatus = cudaMalloc(&deviceVecVolSlice18, (size_t)vecVolSlice18.size() * sizeof(unsigned char));
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaMalloc failed!");
    goto Error;
  }
  cudaStatus = cudaMalloc(&deviceVecVolSlice19, (size_t)vecVolSlice19.size() * sizeof(unsigned char));
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaMalloc failed!");
    goto Error;
  }
  cudaStatus = cudaMalloc(&deviceVecVolSlice20, (size_t)vecVolSlice20.size() * sizeof(unsigned char));
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaMalloc failed!");
    goto Error;
  }
  cudaStatus = cudaMalloc(&deviceVecVolSlice21, (size_t)vecVolSlice21.size() * sizeof(unsigned char));
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaMalloc failed!");
    goto Error;
  }
  cudaStatus = cudaMalloc(&deviceVecVolSlice22, (size_t)vecVolSlice22.size() * sizeof(unsigned char));
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaMalloc failed!");
    goto Error;
  }
  cudaStatus = cudaMalloc(&deviceVecVolSlice23, (size_t)vecVolSlice23.size() * sizeof(unsigned char));
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaMalloc failed!");
    goto Error;
  }
  cudaStatus = cudaMalloc(&deviceVecVolSlice24, (size_t)vecVolSlice24.size() * sizeof(unsigned char));
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaMalloc failed!");
    goto Error;
  }
  cudaStatus = cudaMalloc(&deviceVecVolSlice25, (size_t)vecVolSlice25.size() * sizeof(unsigned char));
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaMalloc failed!");
    goto Error;
  }
  cudaStatus = cudaMalloc(&deviceVecVolSlice26, (size_t)vecVolSlice26.size() * sizeof(unsigned char));
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaMalloc failed!");
    goto Error;
  }
  cudaStatus = cudaMalloc(&deviceVecVolSlice27, (size_t)vecVolSlice27.size() * sizeof(unsigned char));
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaMalloc failed!");
    goto Error;
  }

  // Copy input vectors from host memory to GPU buffers.
  cudaStatus = cudaMemcpy(deviceVecVolSlice01, vecVolSlice01.data(), (size_t)vecVolSlice01.size() * sizeof(unsigned char), cudaMemcpyHostToDevice);
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaMemcpy failed at cudaMemcpyHostToDevice!");
    goto Error;
  }
  cudaStatus = cudaMemcpy(deviceVecVolSlice02, vecVolSlice02.data(), (size_t)vecVolSlice02.size() * sizeof(unsigned char), cudaMemcpyHostToDevice);
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaMemcpy failed at cudaMemcpyHostToDevice!");
    goto Error;
  }
  cudaStatus = cudaMemcpy(deviceVecVolSlice03, vecVolSlice03.data(), (size_t)vecVolSlice03.size() * sizeof(unsigned char), cudaMemcpyHostToDevice);
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaMemcpy failed at cudaMemcpyHostToDevice!");
    goto Error;
  }
  cudaStatus = cudaMemcpy(deviceVecVolSlice04, vecVolSlice04.data(), (size_t)vecVolSlice04.size() * sizeof(unsigned char), cudaMemcpyHostToDevice);
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaMemcpy failed at cudaMemcpyHostToDevice!");
    goto Error;
  }
  cudaStatus = cudaMemcpy(deviceVecVolSlice05, vecVolSlice05.data(), (size_t)vecVolSlice05.size() * sizeof(unsigned char), cudaMemcpyHostToDevice);
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaMemcpy failed at cudaMemcpyHostToDevice!");
    goto Error;
  }
  cudaStatus = cudaMemcpy(deviceVecVolSlice06, vecVolSlice06.data(), (size_t)vecVolSlice06.size() * sizeof(unsigned char), cudaMemcpyHostToDevice);
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaMemcpy failed at cudaMemcpyHostToDevice!");
    goto Error;
  }
  cudaStatus = cudaMemcpy(deviceVecVolSlice07, vecVolSlice07.data(), (size_t)vecVolSlice07.size() * sizeof(unsigned char), cudaMemcpyHostToDevice);
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaMemcpy failed at cudaMemcpyHostToDevice!");
    goto Error;
  }
  cudaStatus = cudaMemcpy(deviceVecVolSlice08, vecVolSlice08.data(), (size_t)vecVolSlice08.size() * sizeof(unsigned char), cudaMemcpyHostToDevice);
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaMemcpy failed at cudaMemcpyHostToDevice!");
    goto Error;
  }
  cudaStatus = cudaMemcpy(deviceVecVolSlice09, vecVolSlice09.data(), (size_t)vecVolSlice09.size() * sizeof(unsigned char), cudaMemcpyHostToDevice);
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaMemcpy failed at cudaMemcpyHostToDevice!");
    goto Error;
  }
  cudaStatus = cudaMemcpy(deviceVecVolSlice10, vecVolSlice10.data(), (size_t)vecVolSlice10.size() * sizeof(unsigned char), cudaMemcpyHostToDevice);
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaMemcpy failed at cudaMemcpyHostToDevice!");
    goto Error;
  }
  cudaStatus = cudaMemcpy(deviceVecVolSlice11, vecVolSlice11.data(), (size_t)vecVolSlice11.size() * sizeof(unsigned char), cudaMemcpyHostToDevice);
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaMemcpy failed at cudaMemcpyHostToDevice!");
    goto Error;
  }
  cudaStatus = cudaMemcpy(deviceVecVolSlice12, vecVolSlice12.data(), (size_t)vecVolSlice12.size() * sizeof(unsigned char), cudaMemcpyHostToDevice);
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaMemcpy failed at cudaMemcpyHostToDevice!");
    goto Error;
  }
  cudaStatus = cudaMemcpy(deviceVecVolSlice13, vecVolSlice13.data(), (size_t)vecVolSlice13.size() * sizeof(unsigned char), cudaMemcpyHostToDevice);
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaMemcpy failed at cudaMemcpyHostToDevice!");
    goto Error;
  }
  cudaStatus = cudaMemcpy(deviceVecVolSlice14, vecVolSlice14.data(), (size_t)vecVolSlice14.size() * sizeof(unsigned char), cudaMemcpyHostToDevice);
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaMemcpy failed at cudaMemcpyHostToDevice!");
    goto Error;
  }
  cudaStatus = cudaMemcpy(deviceVecVolSlice15, vecVolSlice15.data(), (size_t)vecVolSlice15.size() * sizeof(unsigned char), cudaMemcpyHostToDevice);
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaMemcpy failed at cudaMemcpyHostToDevice!");
    goto Error;
  }
  cudaStatus = cudaMemcpy(deviceVecVolSlice16, vecVolSlice16.data(), (size_t)vecVolSlice16.size() * sizeof(unsigned char), cudaMemcpyHostToDevice);
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaMemcpy failed at cudaMemcpyHostToDevice!");
    goto Error;
  }
  cudaStatus = cudaMemcpy(deviceVecVolSlice17, vecVolSlice17.data(), (size_t)vecVolSlice17.size() * sizeof(unsigned char), cudaMemcpyHostToDevice);
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaMemcpy failed at cudaMemcpyHostToDevice!");
    goto Error;
  }
  cudaStatus = cudaMemcpy(deviceVecVolSlice18, vecVolSlice18.data(), (size_t)vecVolSlice18.size() * sizeof(unsigned char), cudaMemcpyHostToDevice);
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaMemcpy failed at cudaMemcpyHostToDevice!");
    goto Error;
  }
  cudaStatus = cudaMemcpy(deviceVecVolSlice19, vecVolSlice19.data(), (size_t)vecVolSlice19.size() * sizeof(unsigned char), cudaMemcpyHostToDevice);
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaMemcpy failed at cudaMemcpyHostToDevice!");
    goto Error;
  }
  cudaStatus = cudaMemcpy(deviceVecVolSlice20, vecVolSlice20.data(), (size_t)vecVolSlice20.size() * sizeof(unsigned char), cudaMemcpyHostToDevice);
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaMemcpy failed at cudaMemcpyHostToDevice!");
    goto Error;
  }
  cudaStatus = cudaMemcpy(deviceVecVolSlice21, vecVolSlice21.data(), (size_t)vecVolSlice21.size() * sizeof(unsigned char), cudaMemcpyHostToDevice);
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaMemcpy failed at cudaMemcpyHostToDevice!");
    goto Error;
  }
  cudaStatus = cudaMemcpy(deviceVecVolSlice22, vecVolSlice22.data(), (size_t)vecVolSlice22.size() * sizeof(unsigned char), cudaMemcpyHostToDevice);
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaMemcpy failed at cudaMemcpyHostToDevice!");
    goto Error;
  }
  cudaStatus = cudaMemcpy(deviceVecVolSlice23, vecVolSlice23.data(), (size_t)vecVolSlice23.size() * sizeof(unsigned char), cudaMemcpyHostToDevice);
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaMemcpy failed at cudaMemcpyHostToDevice!");
    goto Error;
  }
  cudaStatus = cudaMemcpy(deviceVecVolSlice24, vecVolSlice24.data(), (size_t)vecVolSlice24.size() * sizeof(unsigned char), cudaMemcpyHostToDevice);
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaMemcpy failed at cudaMemcpyHostToDevice!");
    goto Error;
  }
  cudaStatus = cudaMemcpy(deviceVecVolSlice25, vecVolSlice25.data(), (size_t)vecVolSlice25.size() * sizeof(unsigned char), cudaMemcpyHostToDevice);
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaMemcpy failed at cudaMemcpyHostToDevice!");
    goto Error;
  }
  cudaStatus = cudaMemcpy(deviceVecVolSlice26, vecVolSlice26.data(), (size_t)vecVolSlice26.size() * sizeof(unsigned char), cudaMemcpyHostToDevice);
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaMemcpy failed at cudaMemcpyHostToDevice!");
    goto Error;
  }
  cudaStatus = cudaMemcpy(deviceVecVolSlice27, vecVolSlice27.data(), (size_t)vecVolSlice27.size() * sizeof(unsigned char), cudaMemcpyHostToDevice);
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaMemcpy failed at cudaMemcpyHostToDevice!");
    goto Error;
  }

  // Launch a kernel on the GPU with one thread for each element.
  dim3 threadsPerBlock(256);
  dim3 numBlocks((outputPlane.size() + threadsPerBlock.x - 1) / threadsPerBlock.x);
  averageVector27Kernel << < numBlocks, threadsPerBlock >> > (deviceOutputPlane, deviceVecVolSlice01
    , deviceVecVolSlice02
    , deviceVecVolSlice03
    , deviceVecVolSlice04
    , deviceVecVolSlice05
    , deviceVecVolSlice06
    , deviceVecVolSlice07
    , deviceVecVolSlice08
    , deviceVecVolSlice09
    , deviceVecVolSlice10
    , deviceVecVolSlice11
    , deviceVecVolSlice12
    , deviceVecVolSlice13
    , deviceVecVolSlice14
    , deviceVecVolSlice15
    , deviceVecVolSlice16
    , deviceVecVolSlice17
    , deviceVecVolSlice18
    , deviceVecVolSlice19
    , deviceVecVolSlice20
    , deviceVecVolSlice21
    , deviceVecVolSlice22
    , deviceVecVolSlice23
    , deviceVecVolSlice24
    , deviceVecVolSlice25
    , deviceVecVolSlice26
    , deviceVecVolSlice27
    , dim1Size * dim2Size);

  // Check for any errors launching the kernel
  cudaStatus = cudaGetLastError();
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "averageKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
    goto Error;
  }

  // cudaDeviceSynchronize waits for the kernel to finish, and returns
  // any errors encountered during the launch.
  cudaStatus = cudaDeviceSynchronize();
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching averageKernel!\n", cudaStatus);
    goto Error;
  }

  // Copy output vector from GPU buffer to host memory.
  outputPlane.resize(dim1Size * dim2Size);
  cudaStatus = cudaMemcpy(outputPlane.data(), deviceOutputPlane, dim1Size * dim2Size * sizeof(unsigned char), cudaMemcpyDeviceToHost);
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaMemcpy failed at cudaMemcpyDeviceToHost!");
    goto Error;
  }

Error:
  cudaFree(deviceOutputPlane);
  cudaFree(deviceVecVolSlice01);
  cudaFree(deviceVecVolSlice02);
  cudaFree(deviceVecVolSlice03);
  cudaFree(deviceVecVolSlice04);
  cudaFree(deviceVecVolSlice05);
  cudaFree(deviceVecVolSlice06);
  cudaFree(deviceVecVolSlice07);
  cudaFree(deviceVecVolSlice08);
  cudaFree(deviceVecVolSlice09);
  cudaFree(deviceVecVolSlice10);
  cudaFree(deviceVecVolSlice11);
  cudaFree(deviceVecVolSlice12);
  cudaFree(deviceVecVolSlice13);
  cudaFree(deviceVecVolSlice14);
  cudaFree(deviceVecVolSlice15);
  cudaFree(deviceVecVolSlice16);
  cudaFree(deviceVecVolSlice17);
  cudaFree(deviceVecVolSlice18);
  cudaFree(deviceVecVolSlice19);
  cudaFree(deviceVecVolSlice20);
  cudaFree(deviceVecVolSlice21);
  cudaFree(deviceVecVolSlice22);
  cudaFree(deviceVecVolSlice23);
  cudaFree(deviceVecVolSlice24);
  cudaFree(deviceVecVolSlice25);
  cudaFree(deviceVecVolSlice26);
  cudaFree(deviceVecVolSlice27);
  return cudaStatus;
}
