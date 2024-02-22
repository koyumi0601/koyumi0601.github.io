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
#include "utils.h"
#include "kernel.cuh"
#include <windows.h>
#include <direct.h>


int main(int argc, char* argv[])
{
  // GPU spec
  print_GPU_properties();

  // prepare dir, path, folder
  std::tuple<OFList<OFString>, std::string, std::string> result = prepare_filesystem(argc, argv);
  OFList<OFString> fileList = std::get<0>(result);
  std::string imageOutputPath = std::get<1>(result);
  std::string stdFolderPath = std::get<2>(result);

  // init params for dcmk
  std::string fullPathName;
  std::string deductedPathName;
  DcmFileFormat dicomFileformat;
  Uint16 rows;
  Uint16 columns;
  Sint32 numberOfFrames;
  Float64 sliceSpacing;
  Float64 rowSpacing, colSpacing;
  OFString viewName;

  bool IsAvailableToReadRowColFrame = false;
  bool IsAvailableToReadPixelSpacing = false;
  bool IsAvailableToReadViewName = false;
  const Uint8* pixelData = NULL;
  unsigned long pixelDataLength = 0;
  float targetSlabThicknessMm = 2.0f;
  int desiredNumAverageCoronalSlice;
  float columnLengthMm;
  float sliceLengthMm;
  int resizedColumnSize, resizedSliceSize;
  // init params for CUDA
  int numRows;
  int numCols;
  int numSlices;
  int targetCroppingSize = 768;

  
  for (const auto& fileName : fileList) {
    fullPathName = (std::string)fileName.c_str();
    deductedPathName = getDeductedPathName(fullPathName, stdFolderPath);
    if (((deductedPathName.find("\\US\\") != std::string::npos)) && (!deductedPathName.empty()))
    {
        std::string dataName = getDataName(deductedPathName);
        DcmDataset* dicomDataset = OpenDICOMAndGetDataset(dicomFileformat, fullPathName);
        extractDICOMMetadata(dicomDataset, rows, columns, numberOfFrames, sliceSpacing, rowSpacing, colSpacing, viewName, pixelData, pixelDataLength);
        calculateDICOMImageSpec(targetSlabThicknessMm, rowSpacing, colSpacing, sliceSpacing, desiredNumAverageCoronalSlice, columnLengthMm, sliceLengthMm, resizedColumnSize, resizedSliceSize, numRows, numCols, numSlices, rows, columns, numberOfFrames);
        
        // Resize the vector to hold the pixel data
        std::vector<unsigned char> vecVol;
        vecVol.resize(pixelDataLength);
        std::memcpy(&vecVol[0], pixelData, pixelDataLength);
        std::vector<unsigned char> tmpVecVol, outputPlane;
        tmpVecVol.resize(numCols * numSlices * desiredNumAverageCoronalSlice);
        outputPlane.resize((size_t)numCols * numSlices);
        cv::Mat averagedCSlice(numSlices, numCols, CV_8U);
        cv::Mat resizedCSlice(resizedSliceSize, resizedColumnSize, CV_8U);

        // process and save images
        for (int slabIndex = 0; slabIndex < (int)(numRows / desiredNumAverageCoronalSlice); ++slabIndex)
        {
          transpose(tmpVecVol, vecVol, numRows, numCols, numSlices, slabIndex, desiredNumAverageCoronalSlice);
          cudaError_t cudastatus = averageVectorForWithCuda(outputPlane, tmpVecVol, numCols, numSlices, desiredNumAverageCoronalSlice);
          cv::Mat croppedCSlice = processCroppedSlice(cudastatus, outputPlane, numCols, numSlices, resizedSliceSize, resizedColumnSize, targetCroppingSize);
          saveImage(imageOutputPath, dataName, viewName.c_str(), croppedCSlice, slabIndex);
        }
    }
  }
  return 0;
}

