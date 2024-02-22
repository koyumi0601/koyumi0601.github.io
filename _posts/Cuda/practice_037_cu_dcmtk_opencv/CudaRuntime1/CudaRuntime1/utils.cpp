#include "utils.h"
#include <direct.h>


void print_GPU_properties() {
  cudaDeviceProp prop;
  int device;
  cudaGetDevice(&device);
  cudaGetDeviceProperties(&prop, device);
  std::cout << "Max Threads per Block: " << prop.maxThreadsPerBlock << std::endl;
  std::cout << "Max Threads per Multiprocessor: " << prop.maxThreadsPerMultiProcessor << std::endl;
}


std::tuple<OFList<OFString>, std::string, std::string> prepare_filesystem(int argc, char* argv[]) {
  OFList<OFString> fileList;
  OFString folderPathStr;
  std::string stdFolderPath, imageOutputPath, rootOutputPath, labelOutputPath;
  const char* folderPath;
  if (argc != 2) {
    std::cerr << "Usage: " << argv[0] << " <folder_path>" << std::endl;
    return std::make_tuple(fileList, stdFolderPath, imageOutputPath);
  }
  folderPath = argv[1];
  folderPathStr = folderPath;
  stdFolderPath = (std::string)folderPath;
  OFStandard::searchDirectoryRecursively(folderPathStr, fileList);
  rootOutputPath = "./TrainDataSet";
  imageOutputPath = "./TrainDataSet/imgs-coronal";
  labelOutputPath = "./TrainDataSet/masks-coronal";
  _mkdir(rootOutputPath.c_str());
  _mkdir(imageOutputPath.c_str());
  _mkdir(labelOutputPath.c_str());
  return std::make_tuple(fileList, imageOutputPath, stdFolderPath);
}


DcmDataset* OpenDICOMAndGetDataset(DcmFileFormat& fileformat, const std::string& filePath)
{
  OFCondition status = fileformat.loadFile(filePath.c_str());
  if (!status.good())
  {
    std::cout << "Load DICOM File Error: " << status.text() << std::endl;
    return nullptr;
  }
  DcmDataset* dataset = fileformat.getDataset();
  return dataset;
}


void extractDICOMMetadata(DcmDataset* dicomDataset,
  Uint16& rows, Uint16& columns, Sint32& numberOfFrames,
  Float64& sliceSpacing, Float64& rowSpacing, Float64& colSpacing,
  OFString& viewName, const Uint8*& pixelData, unsigned long& pixelDataLength)
{
  // Row, Column, and Number of Frames
  bool IsAvailableToReadRowColFrame =
    dicomDataset->findAndGetUint16(DcmTagKey(0x0028, 0x0010), rows).good() &&
    dicomDataset->findAndGetUint16(DcmTagKey(0x0028, 0x0011), columns).good() &&
    dicomDataset->findAndGetSint32(DcmTagKey(0x0028, 0x0008), numberOfFrames).good();
  // Pixel Spacing and Slice Spacing
  bool IsAvailableToReadPixelSpacing =
    dicomDataset->findAndGetFloat64(DcmTagKey(0x0028, 0x0030), rowSpacing, 0).good() &&
    dicomDataset->findAndGetFloat64(DcmTagKey(0x0028, 0x0030), colSpacing, 1).good() &&
    dicomDataset->findAndGetFloat64(DcmTagKey(0x0018, 0x0088), sliceSpacing).good();
  // View Name
  bool IsAvailableToReadViewName = dicomDataset->findAndGetOFString(DcmTagKey(0x0008, 0x2127), viewName).good();
  // Pixel Data
  bool IsPixelDataAvailable = dicomDataset->findAndGetUint8Array(DCM_PixelData, pixelData, &pixelDataLength).good();
  // Logging results
  if (!IsAvailableToReadRowColFrame || !IsAvailableToReadPixelSpacing || !IsAvailableToReadViewName || !IsPixelDataAvailable)
  {
    std::cerr << "Failed to extract all required DICOM metadata" << std::endl;
    if (!IsAvailableToReadRowColFrame)
    {
      std::cerr << "Missing Row, Column, or Number of Frames information." << std::endl;
    }
    if (!IsAvailableToReadPixelSpacing)
    {
      std::cerr << "Missing Pixel or Slice Spacing information." << std::endl;
    }
    if (!IsAvailableToReadViewName)
    {
      std::cerr << "Missing View Name information." << std::endl;
    }
    if (!IsPixelDataAvailable)
    {
      std::cerr << "Failed to get PixelData" << std::endl;
    }
  }
  else
  {
    std::cout << "\nDICOM Metadata:" << std::endl;
    std::cout << "\tRows: " << rows << std::endl;
    std::cout << "\tColumns: " << columns << std::endl;
    std::cout << "\tNumber of Frames: " << numberOfFrames << std::endl;
    std::cout << "\tSlice Spacing: " << sliceSpacing << std::endl;
    std::cout << "\tRow Spacing: " << rowSpacing << std::endl;
    std::cout << "\tColumn Spacing: " << colSpacing << std::endl;
    std::cout << "\tView Name: " << viewName << std::endl;
    std::cout << std::endl;
  }
}


void calculateDICOMImageSpec(float targetSlabThicknessMm, Float64 rowSpacing, Float64 colSpacing, Float64 sliceSpacing,
  int& desiredNumAverageCoronalSlice, float& columnLengthMm, float& sliceLengthMm,
  int& resizedColumnSize, int& resizedSliceSize,
  int& numRows, int& numCols, int& numSlices,
  Uint16 rows, Uint16 columns, Sint32 numberOfFrames)
{
  desiredNumAverageCoronalSlice = (int)(targetSlabThicknessMm / static_cast<float>(rowSpacing) + 0.5f);
  columnLengthMm = static_cast<float>(columns) * colSpacing;
  sliceLengthMm = static_cast<float>(numberOfFrames) * sliceSpacing;
  resizedColumnSize = (colSpacing < sliceSpacing) ? static_cast<int>((columnLengthMm / colSpacing) + 0.5f) : static_cast<int>((columnLengthMm / sliceSpacing) + 0.5f);
  resizedSliceSize = (colSpacing < sliceSpacing) ? static_cast<int>((sliceLengthMm / colSpacing) + 0.5f) : static_cast<int>((sliceLengthMm / sliceSpacing) + 0.5f);
  numRows = static_cast<int>(rows);
  numCols = static_cast<int>(columns);
  numSlices = static_cast<int>(numberOfFrames);
  std::cout << "desiredVolumeInfo:" << std::endl;
  std::cout << "\tdesiredNumAverageCoronalSlice: " << desiredNumAverageCoronalSlice << std::endl;
  std::cout << "\tcolumnLengthMm: " << columnLengthMm << std::endl;
  std::cout << "\tsliceLengthMm: " << sliceLengthMm << std::endl;
  std::cout << "\tresizedColumnSize: " << resizedColumnSize << std::endl;
  std::cout << "\tresizedSliceSize: " << resizedSliceSize << std::endl;
  std::cout << "\tnumRows: " << numRows << std::endl;
  std::cout << "\tnumCols: " << numCols << std::endl;
  std::cout << "\tnumSlices: " << numSlices << std::endl;
}


bool ReadPatientName(DcmFileFormat& fileformat, std::string& filePath)
{
  OFCondition status = fileformat.loadFile(filePath.c_str());
  if (!status.good())
  {
    std::cout << "Load DICOM File Error: " << status.text() << std::endl;
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
    std::cout << "Save DICOM File Error: " << status.text() << std::endl;
    return false;
  }
  return true;
}


void SaveMatToPNG(const cv::Mat& image, const std::string& filename)
{
  std::vector<int> compression_params;
  compression_params.push_back(cv::IMWRITE_PNG_COMPRESSION);
  compression_params.push_back(0); // no compression
  try
  {
    cv::imwrite(filename, image, compression_params);
  }
  catch (const std::runtime_error& ex)
  {
    std::cerr << "Exception converting image to PNG format: " << ex.what() << std::endl;
  }
}


std::vector<std::string> splitString(const std::string& s, const std::string& delimiter)
{
  std::regex regexDelimiter(delimiter);
  std::vector<std::string> tokens(std::sregex_token_iterator(s.begin(), s.end(), regexDelimiter, -1), std::sregex_token_iterator());
  return tokens;
}


std::string getDeductedPathName(const std::string& fullPathName, const std::string& stdFolderPath)
{
  size_t pos = fullPathName.find(stdFolderPath);
  if (pos != std::string::npos)
  {
    return fullPathName.substr(pos + stdFolderPath.length());
  }
  else
  {
    std::cout << "Path not found: " << fullPathName << std::endl;
    return "";
  }
}


std::string getDataName(const std::string& deductedPathName)
{
  std::regex delimiter("\\\\");
  std::vector<std::string> parts(std::sregex_token_iterator(deductedPathName.begin(), deductedPathName.end(), delimiter, -1), std::sregex_token_iterator());
  std::string dataName = (deductedPathName[0] == '\\') ? parts[2] : parts[1];
  dataName.erase(std::remove_if(dataName.begin(), dataName.end(), [](char c) { return c == '-'; }), dataName.end());
  return dataName;
}


cv::Mat cropCenter(cv::Mat& input, int cropWidth, int cropHeight)
{
  int inputWidth = input.cols;
  int inputHeight = input.rows;
  if (inputWidth < cropWidth || inputHeight < cropHeight)
  {
    int paddedWidth = (inputWidth > cropWidth) ? inputWidth : cropWidth;
    int paddedHeight = (inputHeight > cropHeight) ? inputHeight : cropHeight;
    int startX = (paddedWidth - inputWidth) / 2;
    int startY = (paddedHeight - inputHeight) / 2;
    cv::Mat paddedImage(paddedHeight, paddedWidth, input.type(), cv::Scalar(0));
    input.copyTo(paddedImage(cv::Rect(startX, startY, inputWidth, inputHeight)));
    int centerX = paddedWidth / 2;
    int centerY = paddedHeight / 2;
    int startXCrop = centerX - (cropWidth / 2);
    int startYCrop = centerY - (cropHeight / 2);
    return paddedImage(cv::Rect(startXCrop, startYCrop, cropWidth, cropHeight)).clone();
  }
  else
  {
    int centerX = inputWidth / 2;
    int centerY = inputHeight / 2;
    int startX = centerX - (cropWidth / 2);
    int startY = centerY - (cropHeight / 2);
    return input(cv::Rect(startX, startY, cropWidth, cropHeight)).clone();
  }
}


void transpose(std::vector<unsigned char>& tmpVecVol, const std::vector<unsigned char>& vecVol, int numRows, int numCols, int numSlices, int slabIndex, int desiredNumAverageCoronalSlice)
{
  for (int row = slabIndex * desiredNumAverageCoronalSlice; row < (slabIndex + 1) * desiredNumAverageCoronalSlice; ++row) {
    for (int col = 0; col < numCols; ++col) {
      for (int slc = 0; slc < numSlices; ++slc) {
        int index = slc * (numRows * numCols) + row * (numCols)+col;
        int newIndex = (row - (slabIndex * desiredNumAverageCoronalSlice)) * (numCols * numSlices) + slc * (numCols)+col;
        tmpVecVol[newIndex] = vecVol[index];
      }
    }
  }
}


cv::Mat processCroppedSlice(const cudaError_t cudastatus, const std::vector<unsigned char>& outputPlane, const int numCols, const int numSlices, const int resizedSliceSize, const int resizedColumnSize, const int targetCroppingSize)
{
  // Check CUDA status
  if (cudastatus != cudaSuccess) {
    fprintf(stderr, "\naverageVectorWithCuda failed!");
    return cv::Mat(); // Return an empty Mat if CUDA operation fails
  }
  // Create Mat for averaged coronal slice
  cv::Mat averagedCSlice(numSlices, numCols, CV_8U);
  // Copy data from outputPlane to averagedCSlice
  std::memcpy(averagedCSlice.data, &outputPlane[0], numCols * numSlices);
  // Resize averagedCSlice
  cv::Mat resizedCSlice;
  cv::resize(averagedCSlice, resizedCSlice, cv::Size((size_t)resizedSliceSize, (size_t)resizedColumnSize), 0, 0, cv::INTER_LINEAR);
  // Crop resizedCSlice
  cv::Mat croppedCSlice = cropCenter(resizedCSlice, targetCroppingSize, targetCroppingSize);
  // Flip croppedCSlice
  cv::flip(croppedCSlice, croppedCSlice, 0);
  return croppedCSlice; // Return the processed cropped slice
}


void saveImage(const std::string& imageOutputPath, const std::string& dataName, const std::string& viewName, const cv::Mat& croppedCSlice, int slabIndex)
{
  // Create slab name
  char slabName[10];
  sprintf(slabName, "SlabIdx%02d", slabIndex);
  // Construct image filename
  std::stringstream ssImg;
  ssImg << imageOutputPath << "\\" << dataName << "_" << viewName << "_" << slabName << ".png";
  std::string imgFilename = ssImg.str();
  // Save image
  SaveMatToPNG(croppedCSlice, imgFilename);
  std::cout << "Saved image: " << imgFilename << std::endl;
}


void MakeTestData()
{
  int numRows = 100;
  int numCols = 100;
  int numSlices = 100;
  int rowModVal = 15;
  std::vector<unsigned char> vecVol;
  vecVol.resize((size_t)numRows * numCols * numSlices);
  for (int row = 0; row < numRows; ++row)
  {
    for (int col = 0; col < numCols; ++col)
    {
      for (int slc = 0; slc < numSlices; ++slc)
      {
        size_t index = (size_t)slc + (size_t)numSlices * ((size_t)col + (size_t)row * numCols);
        vecVol[index] = (unsigned char)row % rowModVal;
      }
    }
  }
  printf("vecVol[%d] = %d\n", 0, vecVol[0]);
}


