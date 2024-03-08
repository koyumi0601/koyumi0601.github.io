#pragma once
#include <array>
#include <string>
#include <vector>
#include <regex>
#include <map>
#include <fstream>
#include <iostream>
#include "dcmtk/config/osconfig.h"
#include "dcmtk/dcmdata/dctk.h"
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>

struct DicomStruct
{
  DcmTagKey tag_PatientName = DcmTagKey(0x0010, 0x0010); // PatientName
  DcmTagKey tag_PatientID = DcmTagKey(0x0010, 0x0020); // PatientID
  DcmTagKey tag_BitsAllocated = DcmTagKey(0x0028, 0x0100); // BitsAllocated
  DcmTagKey tag_BitsStored = DcmTagKey(0x0028, 0x0101); // BitsStored
  DcmTagKey tag_HighBit = DcmTagKey(0x0028, 0x0102); // HighBit
  DcmTagKey tag_PixelRepresentation = DcmTagKey(0x0028, 0x0103); // PixelRepresentation
  DcmTagKey tag_SliceThickness = DcmTagKey(0x0018, 0x0050); // SliceThickness
  DcmTagKey tag_SpacingBetweenSlices = DcmTagKey(0x0018, 0x0088); // SpacingBetweenSlices
  DcmTagKey tag_SamplesPerPixel = DcmTagKey(0x0028, 0x0002); // SamplesPerPixel
  DcmTagKey tag_NumberOfFrames = DcmTagKey(0x0028, 0x0008); // NumberOfFrames
  DcmTagKey tag_Rows = DcmTagKey(0x0028, 0x0010); // Rows
  DcmTagKey tag_Columns = DcmTagKey(0x0028, 0x0011); // Columns
  DcmTagKey tag_PixelSpacing = DcmTagKey(0x0028, 0x0030); // PixelSpacing
  DcmTagKey tag_PixelData = DcmTagKey(0x7FE0, 0x0010); // PixelData
  DcmTagKey tag_SeriesDescription = DcmTagKey(0x0008, 0x2127); // SeriesDescription (ViewName)
  DcmTagKey tag_ProbeRadius = DcmTagKey(0x0021, 0x1040); // ProbeRadius
  DcmTagKey tag_MaxCut = DcmTagKey(0x0021, 0x1061); // MaxCut
};

std::tuple <std::string, std::string, std::string> GetDataNameViewName(const std::string& pathName);
std::string trimString(std::string str);
std::tuple <std::vector<std::string>, std::map<std::string, std::string>, std::vector<unsigned char>> readNrrd(std::string fullPathName);
bool SavePatientName(DcmFileFormat& fileformat, std::string& filePath, const std::string& info);
DcmDataset* OpenDICOMAndGetDataset(DcmFileFormat& fileformat, std::string& filePath);
std::string StringReplace(const std::string& aStr, const std::string& src, const std::string& des);
void SaveUpdatedDicom(const std::string& outputPath, DcmFileFormat& dicomFileformat, const std::vector<unsigned char>& data, DicomStruct& ds);
void SaveNrrdAsDicom(const std::string& outputPath, const std::vector<unsigned char>& data, std::map<std::string, float>& header, DicomStruct& ds);
std::tuple <bool, std::map<std::string, double>, std::vector<unsigned char>> OpenDICOMAndGetDatasetMap(std::string& filePath, DicomStruct& ds);
std::vector<unsigned char> flipVolumeVector(std::vector<unsigned char>& imageVolume, int rows, int columns, int frames, bool flipMode);
std::string getDeductedPathName(const std::string& fullPathName, const std::string& stdFolderPath);

