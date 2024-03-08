#include "utils1.h"


std::tuple <std::string, std::string, std::string> GetDataNameViewName(const std::string& pathName)
{
  std::regex delimiter("\\\\");
  std::vector<std::string> parts(std::sregex_token_iterator(pathName.begin(), pathName.end(), delimiter, -1), std::sregex_token_iterator());
  std::string fileName = parts[parts.size() - 1];
  std::regex fileNameDelimiter("_|\\.");
  std::vector<std::string> fileNameParts(std::sregex_token_iterator(fileName.begin(), fileName.end(), fileNameDelimiter, -1), std::sregex_token_iterator());
  std::string dataName = fileNameParts[0];
  std::string viewName = fileNameParts[1];
  std::tuple <std::string, std::string, std::string> outTuple;
  outTuple = std::make_tuple(dataName, viewName, fileName);
  return outTuple;
}


std::string trimString(std::string str)
{
  str.erase(0, str.find_first_not_of(" \t\r\n"));
  str.erase(str.find_last_not_of(" \t\r\n") + 1);
  return str;
}


std::tuple <std::vector<std::string>, std::map<std::string, std::string>, std::vector<unsigned char>> readNrrd(std::string fullPathName)
{
  std::tuple <std::vector<std::string>, std::map<std::string, std::string>, std::vector<unsigned char>> outTuple;
  // Open the NRRD file in binary mode
  std::ifstream file(fullPathName, std::ios::binary);
  if (!file)
  {
    std::cerr << "Failed to open file: " << fullPathName << std::endl;
    return outTuple;
  }
  // read header and data
  if (file.is_open())
  {
    const char colonCString[] = ": ";
    const char colonEqualCString[] = ":=";
    std::string starterKeyName = "magicAndComment";
    std::vector<std::string> keyOrder;
    keyOrder.push_back(starterKeyName);
    std::map<std::string, std::string> header;
    std::string key;
    std::string value;
    std::string line;
    size_t colonPos;
    size_t colonEqualPos;

    // read the first line, it's always NRRD00X
    std::getline(file, line);
    std::string magicAndComment = line;
    while (std::getline(file, line))
    {
      //std::cout << "line= " << line << std::endl;
      if ((line.find(colonCString) == std::string::npos) && (line.find(colonEqualCString) == std::string::npos))
      {
        magicAndComment += ("\n" + line);
      }
      else
      {
        colonPos = line.find(colonCString);
        colonEqualPos = line.find(colonEqualCString);
        // fields
        if (colonPos != std::string::npos)
        {
          key = line.substr(0, colonPos);
          value = line.substr(colonPos + strlen(colonCString));
        }
        // keys
        else if (colonEqualPos != std::string::npos)
        {
          key = line.substr(0, colonEqualPos);
          value = line.substr(colonEqualPos + strlen(colonEqualCString));
        }
        keyOrder.push_back(trimString(key));
        header[trimString(key)] = trimString(value);
      }
      if (line.empty()) // header is finished at empty line
      {
        header[trimString(starterKeyName)] = trimString(magicAndComment);
        break;
      }
    }
    // Read the binary data
    std::vector<char> signedData((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    std::vector<unsigned char> unsignedData(signedData.begin(), signedData.end());
    outTuple = std::make_tuple(keyOrder, header, unsignedData);
  }
  // Close the file
  file.close();
  return outTuple;
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


DcmDataset* OpenDICOMAndGetDataset(DcmFileFormat& fileformat, std::string& filePath)
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


void SaveUpdatedDicom(const std::string& outputPath, DcmFileFormat& dicomFileformat, const std::vector<unsigned char>& data, DicomStruct& ds)
{
  // Use referecce DICOM file
  DcmDataset* dicomDataset;
  OFCondition writingResult;
  // Update PixelData
  dicomDataset = dicomFileformat.getDataset();
  writingResult = dicomDataset->putAndInsertUint8Array(ds.tag_PixelData, reinterpret_cast<const Uint8*>(data.data()), (size_t)data.size());
  if (writingResult.bad())
    std::cout << "tag_PixelData is not ok" << std::endl;
  // Save DICOM file
  OFCondition status = dicomFileformat.saveFile(outputPath.c_str(), EXS_LittleEndianExplicit);
  if (status.bad()) {
    std::cerr << "Error saving DICOM file: " << status.text() << std::endl;
  }
  else {
    std::cout << "DICOM file saved successfully." << std::endl;
  }
}


std::string StringReplace(const std::string& aStr, const std::string& src, const std::string& des)
{
  std::string result = aStr;
  size_t found = result.find(src);
  if (found != std::string::npos)
  {
    result.replace(found, src.length(), des);
  }
  return result;
}


void SaveNrrdAsDicom(const std::string& outputPath, const std::vector<unsigned char>& data, std::map<std::string, float>& header, DicomStruct& ds)
{
  // Create an empty DICOM dataset
  DcmFileFormat dicomFileformat;
  DcmDataset* dicomDataset = dicomFileformat.getDataset();
  OFCondition writingResult;

  writingResult = dicomDataset->putAndInsertString(ds.tag_PatientName, "John Doe");
  if (writingResult.bad())
    std::cout << "tag_PatientName is not ok" << std::endl;
  writingResult = dicomDataset->putAndInsertString(ds.tag_PatientID, "123456");
  if (writingResult.bad())
    std::cout << "tag_PatieuntID is not ok" << std::endl;
  writingResult = dicomDataset->putAndInsertUint16(ds.tag_BitsAllocated, (Uint16)8);
  if (writingResult.bad())
    std::cout << "tag_BitsAllocated is not ok" << std::endl;
  writingResult = dicomDataset->putAndInsertUint16(ds.tag_BitsStored, (Uint16)8);
  if (writingResult.bad())
    std::cout << "tag_BitsStored is not ok" << std::endl;
  writingResult = dicomDataset->putAndInsertUint16(ds.tag_HighBit, (Uint16)7);
  if (writingResult.bad())
    std::cout << "tag_HighBit is not ok" << std::endl;
  writingResult = dicomDataset->putAndInsertUint16(ds.tag_PixelRepresentation, (Uint16)0);
  if (writingResult.bad())
    std::cout << "tag_PixelRepresentation is not ok" << std::endl;
  writingResult = dicomDataset->putAndInsertUint16(ds.tag_SamplesPerPixel, (Uint16)1);
  if (writingResult.bad())
    std::cout << "tag_SamplesPerPixel is not ok" << std::endl;
  writingResult = dicomDataset->putAndInsertFloat64(ds.tag_SliceThickness, (Float64)header["SpacingBetweenSlices"]);
  if (writingResult.bad())
    std::cout << "tag_SliceThickness is not ok" << std::endl;
  writingResult = dicomDataset->putAndInsertFloat64(ds.tag_SpacingBetweenSlices, (Float64)header["SpacingBetweenSlices"]);
  if (writingResult.bad())
    std::cout << "tag_SpacingBetweenSlices is not ok" << std::endl;
  writingResult = dicomDataset->putAndInsertOFStringArray(ds.tag_PixelSpacing, OFString((std::to_string(header["PixelSpacingRow"]) + "\\" + std::to_string(header["PixelSpacingCol"])).c_str()));
  if (writingResult.bad())
    std::cout << "tag_PixelSpacing is not ok" << std::endl;
  writingResult = dicomDataset->putAndInsertUint16(ds.tag_SamplesPerPixel, (Uint16)1);
  if (writingResult.bad())
    std::cout << "tag_SamplesPerPixel is not ok" << std::endl;
  writingResult = dicomDataset->putAndInsertString(ds.tag_NumberOfFrames, std::to_string((int)header["NumberOfFrames"]).c_str());
  if (writingResult.bad())
    std::cout << "tag_NumberOfFrames is not ok" << std::endl;
  writingResult = dicomDataset->putAndInsertUint16(ds.tag_Rows, (Uint16)header["Rows"]);
  if (writingResult.bad())
    std::cout << "tag_Rows is not ok" << std::endl;
  writingResult = dicomDataset->putAndInsertUint16(ds.tag_Columns, (Uint16)header["Columns"]);
  if (writingResult.bad())
    std::cout << "tag_Columns is not ok" << std::endl;
  writingResult = dicomDataset->putAndInsertUint8Array(ds.tag_PixelData, reinterpret_cast<const Uint8*>(data.data()), (size_t)data.size());
  if (writingResult.bad())
    std::cout << "tag_PixelData is not ok" << std::endl;

  // Save DICOM file
  OFCondition status = dicomFileformat.saveFile(outputPath.c_str(), EXS_LittleEndianExplicit);
  if (status.bad()) {
    std::cerr << "Error saving DICOM file: " << status.text() << std::endl;
  }
  else {
    std::cout << "DICOM file saved successfully." << std::endl;
  }
}



std::tuple <bool, std::map<std::string, double>, std::vector<unsigned char>> OpenDICOMAndGetDatasetMap(std::string& filePath, DicomStruct& ds)
{
  std::tuple <bool, std::map<std::string, double>, std::vector<unsigned char>> outTuple;
  bool IsDicomCorrectlyRead = false;
  std::map<std::string, double> dicomDataTags;
  std::vector<unsigned char> dicomDataImage;

  DcmFileFormat dicomFileformat;
  bool IsAvailableToReadRowColFrame = false;
  bool IsAvailableToReadPixelSpacing = false;
  bool IsAvailableToReadViewName = false;
  bool IsAvailableToReadPrivateTags = false;
  Uint16 rows, columns;
  Sint32 numberOfFrames;
  Float64 sliceSpacing, rowSpacing, colSpacing, probeRadius, maxCut;
  OFString viewName;
  unsigned long pixelDataLength = 0;
  const Uint8* pixelData = NULL;

  OFCondition status = dicomFileformat.loadFile(filePath.c_str());
  if (!status.good())
  {
    std::cout << "Load DICOM File Error: " << status.text() << std::endl;
    outTuple = std::make_tuple(IsDicomCorrectlyRead, dicomDataTags, dicomDataImage);
    return outTuple;
  }
  DcmDataset* dicomDataset = dicomFileformat.getDataset();

  IsAvailableToReadRowColFrame = (dicomDataset->findAndGetUint16(ds.tag_Rows, rows).good() &&
    dicomDataset->findAndGetUint16(ds.tag_Columns, columns).good() &&
    dicomDataset->findAndGetSint32(ds.tag_NumberOfFrames, numberOfFrames).good());
  IsAvailableToReadPixelSpacing = (dicomDataset->findAndGetFloat64(ds.tag_PixelSpacing, rowSpacing, 0).good() &&
    dicomDataset->findAndGetFloat64(ds.tag_PixelSpacing, colSpacing, 1).good() &&
    dicomDataset->findAndGetFloat64(ds.tag_SpacingBetweenSlices, sliceSpacing).good());
  IsAvailableToReadViewName = dicomDataset->findAndGetOFString(ds.tag_SeriesDescription, viewName).good();
  IsAvailableToReadPrivateTags = (dicomDataset->findAndGetFloat64(ds.tag_ProbeRadius, probeRadius).good() &&
    dicomDataset->findAndGetFloat64(ds.tag_MaxCut, maxCut).good());

  if (IsAvailableToReadRowColFrame && IsAvailableToReadPixelSpacing)
  {
    if (!(dicomDataset->findAndGetUint8Array(DCM_PixelData, pixelData, &pixelDataLength).good()))
    {
      std::cerr << "Failed to get PixelData" << std::endl;
      outTuple = std::make_tuple(IsDicomCorrectlyRead, dicomDataTags, dicomDataImage);
      return outTuple;
    }
  }
  // put tags
  dicomDataTags["Rows"] = (double)rows;
  dicomDataTags["Columns"] = (double)columns;
  dicomDataTags["NumberOfFrames"] = (double)numberOfFrames;
  dicomDataTags["RowSpacing"] = (double)rowSpacing;
  dicomDataTags["ColSpacing"] = (double)colSpacing;
  dicomDataTags["SliceSpacing"] = (double)sliceSpacing;
  dicomDataTags["ProbeRadius"] = (double)probeRadius;
  dicomDataTags["MaxCut"] = (double)maxCut;
  // put image
  dicomDataImage.resize(pixelDataLength);
  std::memcpy(&dicomDataImage[0], pixelData, pixelDataLength);
  IsDicomCorrectlyRead = (IsAvailableToReadRowColFrame && IsAvailableToReadPixelSpacing && IsAvailableToReadViewName && IsAvailableToReadPrivateTags);

  std::cout << "dicomDataTags['Rows']=" << dicomDataTags["Rows"] << std::endl;
  std::cout << "dicomDataTags['Columns']=" << dicomDataTags["Columns"] << std::endl;
  std::cout << "dicomDataTags['NumberOfFrames']=" << dicomDataTags["NumberOfFrames"] << std::endl;
  std::cout << "dicomDataTags['RowSpacing']=" << dicomDataTags["RowSpacing"] << std::endl;
  std::cout << "dicomDataTags['ColSpacing']=" << dicomDataTags["ColSpacing"] << std::endl;
  std::cout << "dicomDataTags['SliceSpacing']=" << dicomDataTags["SliceSpacing"] << std::endl;
  std::cout << "dicomDataTags['ProbeRadius']=" << dicomDataTags["ProbeRadius"] << std::endl;
  std::cout << "dicomDataTags['MaxCut']=" << dicomDataTags["MaxCut"] << std::endl;

  return std::make_tuple(IsDicomCorrectlyRead, dicomDataTags, dicomDataImage);
}



std::vector<unsigned char> flipVolumeVector(std::vector<unsigned char>& imageVolume, int rows, int columns, int frames, bool flipMode)
{
  std::vector<unsigned char> flippedImageVolumeVector(imageVolume.size());
  for (int frame = 0; frame < frames; ++frame)
  {
    cv::Mat frameMat(rows, columns, CV_8U, &imageVolume[frame * rows * columns]);
    cv::Mat flippedFrameMat;
    cv::flip(frameMat, flippedFrameMat, flipMode);
    std::memcpy(&flippedImageVolumeVector[frame * rows * columns], flippedFrameMat.data, rows * columns * sizeof(unsigned char));
  }
  return flippedImageVolumeVector;
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
