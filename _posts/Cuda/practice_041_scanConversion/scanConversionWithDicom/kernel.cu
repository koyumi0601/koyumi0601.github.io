// D:\Github_Blog\Reverse_curved_scanconversion\main_abus_KU_DICOM_convert.py
// D:\Github_Blog\Reverse_curved_scanconversion\functions.py

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include "dcmtk/config/osconfig.h"
#include "dcmtk/dcmdata/dctk.h"
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include "dcmtk/config/osconfig.h"  
#include "dcmtk/ofstd/ofstd.h"      
#include <windows.h>
#include "utils.h"


struct Params
{
  std::string seriesDescription; // dicomViewName = get_Tag_Val_By_Desc(ds, "Series Description")
  std::pair<float, float> pixelSpacing; // PixelSpacing = get_Tag_Val_By_addr(ds, pydicom.tag.Tag('0028', '0030'))
  //        rangeSpacing = float(PixelSpacing[0])
  // azimuthalSpacing = float(PixelSpacing[1])
  float sliceSpacing; // sliceSpacing = float(get_Tag_Val_By_addr(ds, pydicom.tag.Tag('0018', '0088')))
  float probeRadius; // probeRadius = float(get_Tag_Val_By_addr(ds, pydicom.tag.Tag('0021', '1040')))
  float maxCut; // maxCut = float(get_Tag_Val_By_addr(ds, pydicom.tag.Tag('0021', '1061')))
  // dicom_image = ds.pixel_array
  //dataName = f"{dataName.upper()}_{dicomViewName.upper()}"

};

DcmDataset* getDICOMDataset(const std::string& dicomFilePath)
{
  DcmFileFormat fileFormat;
  OFCondition status = fileFormat.loadFile(dicomFilePath.c_str());
  if (status.good())
  {
    // DICOM 파일이 성공적으로 열렸을 때
    return fileFormat.getDataset();
  }
  else
  {
    // DICOM 파일 열기 실패 시 NULL 포인터 반환
    return nullptr;
  }
}


Params getDICOMParams(const std::string& dicomFilePath)
{
  Params params;

  DcmFileFormat fileFormat;
  OFCondition status = fileFormat.loadFile(dicomFilePath.c_str());
  if (status.good())
  {
    // DICOM 파일이 성공적으로 열렸을 때
    DcmDataset* dataset = fileFormat.getDataset();

    // 시리즈 설명 가져오기
    OFString seriesDescription;
    if (dataset->findAndGetOFString(DCM_SeriesDescription, seriesDescription).good())
    {
      params.seriesDescription = seriesDescription.c_str();
    }

    // 픽셀 간격 가져오기
    OFString pixelSpacingX;
    OFString pixelSpacingY;
    if (dataset->findAndGetOFString(DCM_PixelSpacing, pixelSpacingX, 0).good() &&
      dataset->findAndGetOFString(DCM_PixelSpacing, pixelSpacingY, 1).good())
    {
      params.pixelSpacing.first = std::stof(pixelSpacingX.c_str());
      params.pixelSpacing.second = std::stof(pixelSpacingY.c_str());
    }

    // 슬라이스 간격 가져오기
    OFString sliceSpacing;
    if (dataset->findAndGetFloat32(DCM_SpacingBetweenSlices, params.sliceSpacing).bad())
    {
      // 해당 태그가 없으면 0으로 설정
      params.sliceSpacing = 0.0f;
    }

    // 탐침 반지름 가져오기
    OFString probeRadius;
    if (dataset->findAndGetFloat32(DCM_CurvatureRadiusProbe, params.probeRadius).bad())
    {
      // 해당 태그가 없으면 0으로 설정
      params.probeRadius = 0.0f;
    }

    // 최대 절단 값 가져오기
    OFString maxCut;
    if (dataset->findAndGetFloat32(DCM_MaxCut, params.maxCut).bad())
    {
      // 해당 태그가 없으면 0으로 설정
      params.maxCut = 0.0f;
    }
  }
  else
  {
    std::cerr << "Failed to open DICOM file: " << dicomFilePath << std::endl;
  }

  return params;
}


int main(int argc, char* argv[])
{
  // find dicom files
  std::string resourceFolderName = "Tested_DICOM";
  std::string dataFolderName = "US";
  std::pair<std::string, std::vector<std::string>> result = FindFolderAndFileListByNameRecursively(GetResourceDirectory(resourceFolderName), dataFolderName);
  std::string dataFolderPath = result.first;
  std::vector<std::string> dataFileList = result.second;

  // process each dicom file
  for (const auto& dataFile : dataFileList) {
    std::cout << dataFile << std::endl;
    std::string dicomFilePath = dataFolderPath + "/" + dataFile;
    Params params = getDicomParams(dicomFilePath);

   /* ds = pydicom.read_file(dicomUSFileName, stop_before_pixels = False)
      dicomViewName = get_Tag_Val_By_Desc(ds, "Series Description")

      # get DICOM tag
      PixelSpacing = get_Tag_Val_By_addr(ds, pydicom.tag.Tag('0028', '0030'))
      rangeSpacing = float(PixelSpacing[0])
      azimuthalSpacing = float(PixelSpacing[1])
      sliceSpacing = float(get_Tag_Val_By_addr(ds, pydicom.tag.Tag('0018', '0088')))
      probeRadius = float(get_Tag_Val_By_addr(ds, pydicom.tag.Tag('0021', '1040')))
      maxCut = float(get_Tag_Val_By_addr(ds, pydicom.tag.Tag('0021', '1061')))
      metadata = { 'PixelSpacing' : (rangeSpacing, azimuthalSpacing),
                  'SpacingBetweenSlices' : sliceSpacing,
                  'CurvatureRadiusProbe' : probeRadius,
                  'MaxCut' : maxCut }
      dicom_image = ds.pixel_array
      dataName = f"{dataName.upper()}_{dicomViewName.upper()}"*/

    DcmTagKey tagKey(0x0010, 0x0010);
    std::string tagValue = getTagValue(dicomFilePath, tagKey);
    std::cout << "File: " << dataFile << ", Tag Value: " << tagValue << std::endl;


  }

  return 0;
}
  // read dicom metadata
  // scan conversion
  // interpolation

