#ifndef DICOM_UTILS_H
#define DICOM_UTILS_H

#include "dcmtk/dcmdata/dctk.h"
#include <string>
#include <iostream>
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <regex>
#include <cuda_runtime.h>
#include <direct.h>


void print_GPU_properties();
std::tuple<OFList<OFString>, std::string, std::string> prepare_filesystem(int argc, char* argv[]);
DcmDataset* OpenDICOMAndGetDataset(DcmFileFormat& fileformat, const std::string& filePath);
void extractDICOMMetadata(DcmDataset* dicomDataset,
  Uint16& rows, Uint16& columns, Sint32& numberOfFrames,
  Float64& sliceSpacing, Float64& rowSpacing, Float64& colSpacing,
  OFString& viewName, const Uint8*& pixelData, unsigned long& pixelDataLength);
void calculateDICOMImageSpec(float targetSlabThicknessMm, Float64 rowSpacing, Float64 colSpacing, Float64 sliceSpacing,
  int& desiredNumAverageCoronalSlice, float& columnLengthMm, float& sliceLengthMm,
  int& resizedColumnSize, int& resizedSliceSize,
  int& numRows, int& numCols, int& numSlices,
  Uint16 rows, Uint16 columns, Sint32 numberOfFrames);
bool ReadPatientName(DcmFileFormat& fileformat, std::string& filePath);
bool SavePatientName(DcmFileFormat& fileformat, std::string& filePath, const std::string& info);
void SaveMatToPNG(const cv::Mat& image, const std::string& filename);
std::vector<std::string> splitString(const std::string& s, const std::string& delimiter);
std::string getDeductedPathName(const std::string& fullPathName, const std::string& stdFolderPath);
std::string getDataName(const std::string& deductedPathName);
cv::Mat cropCenter(cv::Mat& input, int cropWidth, int cropHeight);
void transpose(std::vector<unsigned char>& tmpVecVol, const std::vector<unsigned char>& vecVol, int numRows, int numCols, int numSlices, int slabIndex, int desiredNumAverageCoronalSlice);
cv::Mat processCroppedSlice(const cudaError_t cudastatus, const std::vector<unsigned char>& outputPlane, const int numCols, const int numSlices, const int resizedSliceSize, const int resizedColumnSize, const int targetCroppingSize);
void saveImage(const std::string& imageOutputPath, const std::string& dataName, const std::string& viewName, const cv::Mat& croppedCSlice, int slabIndex);
void MakeTestData();


#endif // DICOM_UTILS_H