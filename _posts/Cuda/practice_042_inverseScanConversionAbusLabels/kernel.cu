#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <string>
#include <iostream>
#include <vector>
#include <array>
#include <regex>
#include "dcmtk/config/osconfig.h"
#include "dcmtk/dcmdata/dctk.h"
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <map>
#include "kernel.h"
#include <Eigen/Dense>
#define PI 3.14159265358979323846


__global__ void inverseScanConversionKernels(unsigned char* deviceOutputPlane, unsigned char** deviceVecVolSlices, int numRows, int numCols, int numVectors, int dstRows, int dstColumns, double* deviceIndexX, double* deviceIndexY);
cudaError_t inverseScanConversionWithCuda(std::vector<unsigned char>& outputVecVol, std::vector<unsigned char>& nrrdData, unsigned int rows, unsigned int columns, unsigned int numberOfFrames, unsigned int dstRows, unsigned int dstColumns, std::vector<double> srcIndexWiseXVec, std::vector<double> srcIndexWiseYVec);


struct DicomStruct
{
    DcmTagKey tag_PatientName          = DcmTagKey(0x0010, 0x0010); // PatientName
    DcmTagKey tag_PatientID            = DcmTagKey(0x0010, 0x0020); // PatientID
    DcmTagKey tag_BitsAllocated        = DcmTagKey(0x0028, 0x0100); // BitsAllocated
    DcmTagKey tag_BitsStored           = DcmTagKey(0x0028, 0x0101); // BitsStored
    DcmTagKey tag_HighBit              = DcmTagKey(0x0028, 0x0102); // HighBit
    DcmTagKey tag_PixelRepresentation  = DcmTagKey(0x0028, 0x0103); // PixelRepresentation
    DcmTagKey tag_SliceThickness       = DcmTagKey(0x0018, 0x0050); // SliceThickness
    DcmTagKey tag_SpacingBetweenSlices = DcmTagKey(0x0018, 0x0088); // SpacingBetweenSlices
    DcmTagKey tag_SamplesPerPixel      = DcmTagKey(0x0028, 0x0002); // SamplesPerPixel
    DcmTagKey tag_NumberOfFrames       = DcmTagKey(0x0028, 0x0008); // NumberOfFrames
    DcmTagKey tag_Rows                 = DcmTagKey(0x0028, 0x0010); // Rows
    DcmTagKey tag_Columns              = DcmTagKey(0x0028, 0x0011); // Columns
    DcmTagKey tag_PixelSpacing         = DcmTagKey(0x0028, 0x0030); // PixelSpacing
    DcmTagKey tag_PixelData            = DcmTagKey(0x7FE0, 0x0010); // PixelData
    DcmTagKey tag_SeriesDescription    = DcmTagKey(0x0008, 0x2127); // SeriesDescription (ViewName)
    DcmTagKey tag_ProbeRadius          = DcmTagKey(0x0021, 0x1040); // ProbeRadius
    DcmTagKey tag_MaxCut               = DcmTagKey(0x0021, 0x1061); // MaxCut
};


DicomStruct ds;


std::tuple <std::string, std::string, std::string> GetNrrdDataNameViewName(const std::string& pathName)
{
    std::regex delimiter("\\\\");
    std::vector<std::string> parts(std::sregex_token_iterator(pathName.begin(), pathName.end(), delimiter, -1), std::sregex_token_iterator());
    std::string fileName = parts[parts.size()-1];
    std::regex fileNameDelimiter("_|\\.");
    std::vector<std::string> fileNameParts(std::sregex_token_iterator(fileName.begin(), fileName.end(), fileNameDelimiter, -1), std::sregex_token_iterator());
    std::string groupName = fileNameParts[0];
    std::string dataName = fileNameParts[1];
    std::tuple <std::string, std::string, std::string> outTuple;
    outTuple = std::make_tuple(groupName, dataName, fileName);
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


void Save3DDataAsDicom(const std::string& outputPath, const std::vector<unsigned char>& data, std::map<std::string, float>& dicomHeader)
{
    // Create an empty DICOM dataset
    DcmFileFormat dicomFileformat;
    DcmDataset* dicomDataset = dicomFileformat.getDataset();
    //DcmFileFormat dicomFileformat;
    //DcmDataset* dicomDataset;
    //std::string thePath = "D:/02.project_bitbucket/01.inverseScanConversionAbusLabels/resource/test.dcm";
    //dicomDataset = OpenDICOMAndGetDataset(dicomFileformat, thePath);

    OFCondition writingResult = dicomDataset->putAndInsertString(ds.tag_PatientName, "John Doe");
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
    writingResult = dicomDataset->putAndInsertFloat64(ds.tag_SliceThickness, (Float64)dicomHeader["SpacingBetweenSlices"]);
    if (writingResult.bad())
        std::cout << "tag_SliceThickness is not ok" << std::endl;
    writingResult = dicomDataset->putAndInsertFloat64(ds.tag_SpacingBetweenSlices, (Float64)dicomHeader["SpacingBetweenSlices"]);
    if (writingResult.bad())
        std::cout << "tag_SpacingBetweenSlices is not ok" << std::endl;
    writingResult = dicomDataset->putAndInsertOFStringArray(ds.tag_PixelSpacing, OFString((std::to_string(dicomHeader["PixelSpacingRow"]) + "\\" + std::to_string(dicomHeader["PixelSpacingCol"])).c_str()));
    if (writingResult.bad())
        std::cout << "tag_PixelSpacing is not ok" << std::endl;
    writingResult = dicomDataset->putAndInsertUint16(ds.tag_SamplesPerPixel, (Uint16)1);
    if (writingResult.bad())
        std::cout << "tag_SamplesPerPixel is not ok" << std::endl;
    writingResult = dicomDataset->putAndInsertString(ds.tag_NumberOfFrames, std::to_string((int)dicomHeader["NumberOfFrames"]).c_str());
    if (writingResult.bad())
        std::cout << "tag_NumberOfFrames is not ok" << std::endl;
    writingResult = dicomDataset->putAndInsertUint16(ds.tag_Rows, (Uint16)dicomHeader["Rows"]);
    if (writingResult.bad())
        std::cout << "tag_Rows is not ok" << std::endl;
    writingResult = dicomDataset->putAndInsertUint16(ds.tag_Columns, (Uint16)dicomHeader["Columns"]);
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


std::tuple <bool, std::map<std::string, double>, std::vector<unsigned char>> OpenDICOMAndGetDatasetMap(std::string& filePath)
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

    IsAvailableToReadRowColFrame =  (dicomDataset->findAndGetUint16(ds.tag_Rows, rows).good() &&
                                     dicomDataset->findAndGetUint16(ds.tag_Columns, columns).good() &&
                                     dicomDataset->findAndGetSint32(ds.tag_NumberOfFrames, numberOfFrames).good());
    IsAvailableToReadPixelSpacing = (dicomDataset->findAndGetFloat64(ds.tag_PixelSpacing, rowSpacing, 0).good() &&
                                     dicomDataset->findAndGetFloat64(ds.tag_PixelSpacing, colSpacing, 1).good() &&
                                     dicomDataset->findAndGetFloat64(ds.tag_SpacingBetweenSlices, sliceSpacing).good());
    IsAvailableToReadViewName =      dicomDataset->findAndGetOFString(ds.tag_SeriesDescription, viewName).good();
    IsAvailableToReadPrivateTags =  (dicomDataset->findAndGetFloat64(ds.tag_ProbeRadius, probeRadius).good() &&
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
    return std::make_tuple(IsDicomCorrectlyRead, dicomDataTags, dicomDataImage);
}


int main(int argc, char* argv[])
{
    const double targetPrecision = 0.05; // (mm)
    const double targetResamplingUnitLength = 0.15; // (mm)

    // get DICOM filelist in subfolders
    const char* dicomFolderPath = argv[1];
    OFString dicomFolderPathStr;
    dicomFolderPathStr = dicomFolderPath;
    OFList<OFString> dicomFileList;
    OFStandard::searchDirectoryRecursively(dicomFolderPathStr, dicomFileList);

    // get NRRD filelist in subfolders
    const char* nrrdFolderPath = argv[2];
    OFString nrrdFolderPathStr;
    nrrdFolderPathStr = nrrdFolderPath;
    OFList<OFString> nrrdFileList;
    OFStandard::searchDirectoryRecursively(nrrdFolderPathStr, nrrdFileList);

    // declaration
    std::tuple <bool, std::map<std::string, double>, std::vector<unsigned char>> dicomDataTuple;
    bool IsDicomCorrectlyRead;
    std::map<std::string, double> dicomDataTags;
    std::vector<unsigned char> dicomDataImage;

    // declaration
    std::string dicomFullPathName;
    std::string nrrdFullPathName;
    std::tuple <std::string, std::string, std::string> outDataNameViewName;
    std::tuple <std::vector<std::string>, std::map<std::string, std::string>, std::vector<unsigned char>> outHeaderData;
    std::string dataName;
    std::string viewName;
    std::string nrrdDataName;
    std::vector<std::string> keyOrder;
    std::map<std::string, std::string> header;
    std::map<std::string, float> dicomHeader;
    std::vector<unsigned char> nrrdData;
    std::vector<std::string> essentialKeys = {"dimension", "sizes", "type", "encoding"};
    int dimension;

    Eigen::MatrixXd srcIndexWiseXMesh;
    Eigen::MatrixXd srcIndexWiseYMesh;


    for (const auto& dicomFileName : dicomFileList)
    {
        dicomFullPathName = (std::string)dicomFileName.c_str();
        if (dicomFullPathName.find(".dcm") != std::string::npos)
        {
            dicomDataTuple = OpenDICOMAndGetDatasetMap(dicomFullPathName);
            IsDicomCorrectlyRead = std::get<0>(dicomDataTuple);
            if (IsDicomCorrectlyRead)
            {
                dicomDataTags = std::get<1>(dicomDataTuple);
                dicomDataImage = std::get<2>(dicomDataTuple);

                std::cout << "dicomDataTags['Rows']=" << dicomDataTags["Rows"] << std::endl;
                std::cout << "dicomDataTags['Columns']=" << dicomDataTags["Columns"] << std::endl;
                std::cout << "dicomDataTags['NumberOfFrames']=" << dicomDataTags["NumberOfFrames"] << std::endl;
                std::cout << "dicomDataTags['RowSpacing']=" << dicomDataTags["RowSpacing"] << std::endl;
                std::cout << "dicomDataTags['ColSpacing']=" << dicomDataTags["ColSpacing"] << std::endl;
                std::cout << "dicomDataTags['SliceSpacing']=" << dicomDataTags["SliceSpacing"] << std::endl;
                std::cout << "dicomDataTags['ProbeRadius']=" << dicomDataTags["ProbeRadius"] << std::endl;
                std::cout << "dicomDataTags['MaxCut']=" << dicomDataTags["MaxCut"] << std::endl;

                // preprocess for inverse scanconverion
                int center_IJK = (int)dicomDataTags["Columns"] / 2;
                std::vector<double> srcRangeA, srcAngleA;
                for (int rangeIdx = 0; rangeIdx < (int)dicomDataTags["Rows"]; ++rangeIdx)
                {
                    srcRangeA.push_back(dicomDataTags["ProbeRadius"] - ((double)rangeIdx * dicomDataTags["RowSpacing"]));
                }
                for (int angleIdx = 0; angleIdx < (int)dicomDataTags["Columns"]; ++angleIdx)
                {
                    srcAngleA.push_back(((double)angleIdx - (double)center_IJK) * dicomDataTags["ColSpacing"] / dicomDataTags["ProbeRadius"] * 180.0f / PI);
                }
                Eigen::MatrixXd srcRangeMesh, srcAngleMesh;
                srcRangeMesh.resize(srcRangeA.size(), srcAngleA.size());
                srcAngleMesh.resize(srcRangeA.size(), srcAngleA.size());
                for (int i = 0; i < srcRangeA.size(); ++i)
                {
                    for (int j = 0; j < srcAngleA.size(); ++j)
                    {
                        srcRangeMesh(i, j) = srcRangeA[i];
                        srcAngleMesh(i, j) = srcAngleA[j];
                    }
                }
                Eigen::MatrixXd srcYMesh = ((srcRangeMesh.array().square()) / ((srcAngleMesh.array()/180.0f*PI).tan().square() + 1.0f)).sqrt();
                Eigen::MatrixXd srcXMesh = (srcAngleMesh.array() / 180.0f * PI).tan() * srcYMesh.array();

                double minSrcRangeA = *std::min_element(srcRangeA.begin(), srcRangeA.end());
                double maxSrcRangeA = *std::max_element(srcRangeA.begin(), srcRangeA.end());
                double minSrcAngleA = *std::min_element(srcAngleA.begin(), srcAngleA.end());
                double maxSrcAngleA = *std::max_element(srcAngleA.begin(), srcAngleA.end());
                double targetXMin = (-1.0f) * std::round(std::round(std::abs(maxSrcRangeA * std::sin(minSrcAngleA / 180.0f * PI)) / targetResamplingUnitLength) * targetResamplingUnitLength * 100.0f) / 100.0f;
                double targetXMax = std::round(std::round(maxSrcRangeA * std::sin(maxSrcAngleA / 180.0f * PI) / targetResamplingUnitLength) * targetResamplingUnitLength * 100.0f) / 100.0f;
                double targetYMin = std::round(std::round(minSrcRangeA * std::cos(minSrcAngleA / 180.0f * PI) / targetResamplingUnitLength) * targetResamplingUnitLength * 100.0f) / 100.0f;
                double targetYMax = std::round(std::round(maxSrcRangeA / targetResamplingUnitLength) * targetResamplingUnitLength * 100.0f) / 100.0f;

                std::vector<double> X;
                X.push_back(targetXMin);
                while (true)
                {
                    if (X[(size_t)(X.size()) - 1] + targetResamplingUnitLength > targetXMax)
                    {
                        break;
                    }
                    else
                    {
                        X.push_back(X[(size_t)(X.size()) - 1] + targetResamplingUnitLength);
                    }
                }
                std::vector<double> Y;
                Y.push_back(targetYMin);
                while (true)
                {
                    if (Y[(size_t)(Y.size()) - 1] + targetResamplingUnitLength > targetYMax)
                    {
                        break;
                    }
                    else
                    {
                        Y.push_back(Y[(size_t)(Y.size()) - 1] + targetResamplingUnitLength);
                    }
                }
                Eigen::MatrixXd dstXMesh, dstYMesh;
                dstXMesh.resize(X.size(), Y.size());
                dstYMesh.resize(X.size(), Y.size());
                for (int i = 0; i < X.size(); ++i)
                {
                    for (int j = 0; j < Y.size(); ++j)
                    {
                        dstXMesh(i, j) = X[i];
                        dstYMesh(i, j) = Y[j];
                    }
                }
                Eigen::MatrixXd dstRangeMesh = (dstXMesh.array().square() + dstYMesh.array().square()).sqrt();
                Eigen::MatrixXd dstAngleMesh = (dstXMesh.array() / dstYMesh.array()).atan().array() / PI * 180.0f;
                srcIndexWiseXMesh = (srcXMesh.array() - X[0]) / (X[(size_t)X.size() - 1] - X[0]) * ((double)X.size() - 1.0f);
                srcIndexWiseYMesh = (srcYMesh.array() - Y[0]) / (Y[(size_t)Y.size() - 1] - Y[0]) * ((double)Y.size() - 1.0f);

                std::vector<double> srcIndexWiseXVec, srcIndexWiseYVec;
                srcIndexWiseXVec.reserve(srcIndexWiseXMesh.rows() * srcIndexWiseXMesh.cols());
                srcIndexWiseYVec.reserve(srcIndexWiseXMesh.rows() * srcIndexWiseXMesh.cols());

                for (int i = 0; i < srcIndexWiseXMesh.rows(); ++i)
                {
                    for (int j = 0; j < srcIndexWiseXMesh.cols(); ++j)
                    {
                        srcIndexWiseXVec.push_back(srcIndexWiseXMesh(i, j));
                        srcIndexWiseYVec.push_back(srcIndexWiseYMesh(i, j));
                    }
                }





                for (const auto& nrrdFileName : nrrdFileList)
                {
                    nrrdFullPathName = (std::string)nrrdFileName.c_str();
                    if (nrrdFullPathName.find(".nrrd") != std::string::npos)
                    {
                        outDataNameViewName = GetNrrdDataNameViewName(nrrdFullPathName);
                        dataName = std::get<0>(outDataNameViewName);
                        viewName = std::get<1>(outDataNameViewName);
                        nrrdDataName = std::get<2>(outDataNameViewName);
                        outHeaderData = readNrrd(nrrdFullPathName);
                        keyOrder = std::get<0>(outHeaderData);
                        header = std::get<1>(outHeaderData);
                        nrrdData = std::get<2>(outHeaderData);

                        //// print header
                        //std::cout << "\n" << std::endl;
                        //for (const std::pair<std::string, std::string>& pair : header)
                        //{
                        //    std::cout << pair.first << "       " << pair.second << std::endl;
                        //}
                        //std::cout << "\n" << std::endl;
                        //for (const auto& keyString : essentialKeys)
                        //    std::cout << "header[" << keyString << "]= " << header[keyString] << std::endl;

                        // get info from header string
                        dimension = std::stoi(header["dimension"]);
                        std::regex whiteSpaceDelimiter = std::regex(" ");
                        std::vector<std::string> sizesStrings(std::sregex_token_iterator(header["sizes"].begin(), header["sizes"].end(), whiteSpaceDelimiter, -1), std::sregex_token_iterator());
                        std::vector<int> sizes;
                        for (int dimIdx = 0; dimIdx < dimension; ++dimIdx)
                        {
                            sizes.push_back(std::stoi(sizesStrings[dimIdx]));
                        }

                        dicomHeader["Rows"] = std::stof(sizesStrings[1]);
                        dicomHeader["Columns"] = std::stof(sizesStrings[0]);
                        dicomHeader["NumberOfFrames"] = std::stof(sizesStrings[2]);

                        std::regex braceDelimiter = std::regex("\\(|\\)");
                        std::string removedBracesString = std::regex_replace(header["space directions"], braceDelimiter, "");
                        std::vector<std::string> spaceDirectionsStrings(std::sregex_token_iterator(removedBracesString.begin(), removedBracesString.end(), whiteSpaceDelimiter, -1), std::sregex_token_iterator());
                        std::regex commaDelimiter = std::regex("\\,");
                        std::vector<std::string> subSpaceDirectionsStrings(std::sregex_token_iterator(spaceDirectionsStrings[0].begin(), spaceDirectionsStrings[0].end(), commaDelimiter, -1), std::sregex_token_iterator());
                        dicomHeader["PixelSpacingRow"] = std::stof(subSpaceDirectionsStrings[0]);
                        subSpaceDirectionsStrings = std::vector<std::string>(std::sregex_token_iterator(spaceDirectionsStrings[1].begin(), spaceDirectionsStrings[1].end(), commaDelimiter, -1), std::sregex_token_iterator());
                        dicomHeader["PixelSpacingCol"] = std::stof(subSpaceDirectionsStrings[1]);
                        subSpaceDirectionsStrings = std::vector<std::string>(std::sregex_token_iterator(spaceDirectionsStrings[2].begin(), spaceDirectionsStrings[2].end(), commaDelimiter, -1), std::sregex_token_iterator());
                        dicomHeader["SpacingBetweenSlices"] = std::stof(subSpaceDirectionsStrings[2]);

                        //// draw image
                        //cv::Mat img(dicomHeader["Rows"], dicomHeader["Columns"], CV_8U);
                        //std::memcpy(img.data, &nrrdData[((size_t)dicomHeader["Rows"] * (size_t)dicomHeader["Columns"] * (size_t)dicomHeader["NumberOfFrames"] /2)], ((size_t)dicomHeader["Rows"] * (size_t)dicomHeader["Columns"]));
                        //if (nrrdFullPathName.find(".seg.") != std::string::npos)
                        //{
                        //    img *= 255;
                        //}
                        //cv::imshow("test", img);
                        //cv::waitKey(0);

                        std::vector<unsigned char> outputVecVol;
                        outputVecVol.resize((size_t)srcIndexWiseXMesh.rows() * (size_t)srcIndexWiseXMesh.cols());
                        //outputVecVol.resize((size_t)dicomHeader["Rows"] * (size_t)dicomHeader["Columns"] * (size_t)dicomHeader["NumberOfFrames"]);


                        std::cout << "srcIndexWiseXVec.size()=" << srcIndexWiseXVec.size() << ", srcIndexWiseYVec.size()=" << srcIndexWiseYVec.size() << ", outputVecVol.size()=" << outputVecVol.size() << std::endl;

                        inverseScanConversionWithCuda(outputVecVol, nrrdData, (unsigned int)dicomHeader["Rows"], (unsigned int)dicomHeader["Columns"], (unsigned int)dicomHeader["NumberOfFrames"], (unsigned int)srcIndexWiseXMesh.rows(), (unsigned int)srcIndexWiseXMesh.cols(), srcIndexWiseXVec, srcIndexWiseYVec);

                        std::cout << "srcIndexWiseXMesh.rows()=" << srcIndexWiseXMesh.rows() << ", srcIndexWiseXMesh.cols()=" << srcIndexWiseXMesh.cols() << std::endl;
                        std::cout << "srcIndexWiseXMesh.rows()=" << srcIndexWiseXMesh.rows() << ", srcIndexWiseXMesh.cols()=" << srcIndexWiseXMesh.cols() << std::endl;

                        std::cout << "dicomHeader['Rows']=" << dicomHeader["Rows"] << ", dicomHeader['Columns']=" << dicomHeader["Columns"] << ", dicomHeader['NumberOfFrames']=" << dicomHeader["NumberOfFrames"] << std::endl;
                        std::cout << "nrrdData.size()=" << nrrdData.size() << ", Rows x Columns x NumberOfFrames=" << dicomHeader["Rows"] * dicomHeader["Columns"] * dicomHeader["NumberOfFrames"] << std::endl;
                        
                        cv::Mat img(srcIndexWiseXMesh.rows(), srcIndexWiseXMesh.cols(), CV_8U);
                        std::memcpy(img.data, &outputVecVol[0], ((size_t)srcIndexWiseXMesh.rows()* (size_t)srcIndexWiseXMesh.cols()));
                        if (nrrdFullPathName.find(".seg.") != std::string::npos)
                        {
                            img *= 255;
                        }
                        cv::imshow("test1", img);
                        cv::Mat img2(srcIndexWiseXMesh.cols(), srcIndexWiseXMesh.rows(), CV_8U);
                        std::memcpy(img2.data, &outputVecVol[0], ((size_t)srcIndexWiseXMesh.rows() * (size_t)srcIndexWiseXMesh.cols()));
                        if (nrrdFullPathName.find(".seg.") != std::string::npos)
                        {
                            img2 *= 255;
                        }
                        cv::imshow("test2", img2);
                        cv::waitKey(0);

                        // write the volume as DICOM
                        const std::string outputPath = "./" + StringReplace(nrrdDataName, ".nrrd", ".dcm");
                        //Save3DDataAsDicom(outputPath, nrrdData, dicomHeader);
                    }
                }

                //// Print out the results
                //std::cout << std::endl;
                //std::cout << "srcIndexWiseXMesh.minCoeff():" << srcIndexWiseXMesh.minCoeff() << ", " << "srcIndexWiseXMesh.maxCoeff():" << srcIndexWiseXMesh.maxCoeff() << std::endl;
                //std::cout << "srcIndexWiseYMesh.minCoeff():" << srcIndexWiseYMesh.minCoeff() << ", " << "srcIndexWiseYMesh.maxCoeff():" << srcIndexWiseYMesh.maxCoeff() << std::endl;
                //std::cout << std::endl;
                //std::cout << "srcRangeMesh(0,0):" << srcRangeMesh(0, 0) << ", " << "srcRangeMesh(-1,-1):" << srcRangeMesh((size_t)srcRangeMesh.rows() - 1, (size_t)srcRangeMesh.cols() - 1) << std::endl;
                //std::cout << "srcAngleMesh(0,0):" << srcAngleMesh(0, 0) << ", " << "srcAngleMesh(-1,-1):" << srcAngleMesh((size_t)srcAngleMesh.rows() - 1, (size_t)srcAngleMesh.cols() - 1) << std::endl;
                //std::cout << "srcRangeMesh.rows():" << srcRangeMesh.rows() << ", " << "srcRangeMesh.cols():" << srcRangeMesh.cols() << std::endl;
                //std::cout << "srcAngleMesh.rows():" << srcAngleMesh.rows() << ", " << "srcAngleMesh.cols():" << srcAngleMesh.cols() << std::endl;
                //std::cout << "dstRangeMesh(0,0):" << dstRangeMesh(0, 0) << ", " << "dstRangeMesh(-1,-1):" << dstRangeMesh((size_t)dstRangeMesh.rows() - 1, (size_t)dstRangeMesh.cols() - 1) << std::endl;
                //std::cout << "dstAngleMesh(0,0):" << dstAngleMesh(0, 0) << ", " << "dstAngleMesh(-1,-1):" << dstAngleMesh((size_t)dstAngleMesh.rows() - 1, (size_t)dstAngleMesh.cols() - 1) << std::endl;
                //std::cout << "dstRangeMesh.rows():" << dstRangeMesh.rows() << ", " << "dstRangeMesh.cols():" << dstRangeMesh.cols() << std::endl;
                //std::cout << "dstAngleMesh.rows():" << dstAngleMesh.rows() << ", " << "dstAngleMesh.cols():" << dstAngleMesh.cols() << std::endl;
                //std::cout << std::endl;
                //std::cout << "srcXMesh(0,0):" << srcXMesh(0, 0) << ", " << "srcXMesh(1,0):" << srcXMesh(1, 0) << ", " << "srcXMesh(0,1):" << srcXMesh(0, 1) << std::endl;
                //std::cout << "srcXMesh(0,0):" << srcXMesh(0, 0) << ", " << "srcXMesh(-1,-1):" << srcXMesh((size_t)srcXMesh.rows() - 1, (size_t)srcXMesh.cols() - 1) << std::endl;
                //std::cout << "srcYMesh(0,0):" << srcYMesh(0, 0) << ", " << "srcYMesh(1,0):" << srcYMesh(1, 0) << ", " << "srcYMesh(0,1):" << srcYMesh(0, 1) << std::endl;
                //std::cout << "srcYMesh(0,0):" << srcYMesh(0, 0) << ", " << "srcYMesh(-1,-1):" << srcYMesh((size_t)srcYMesh.rows() - 1, (size_t)srcYMesh.cols() - 1) << std::endl;
                //std::cout << "dstXMesh(0,0):" << dstXMesh(0, 0) << ", " << "dstXMesh(1,0):" << dstXMesh(1, 0) << ", " << "dstXMesh(0,1):" << dstXMesh(0, 1) << std::endl;
                //std::cout << "dstXMesh(0,0):" << dstXMesh(0, 0) << ", " << "dstXMesh(-1,-1):" << dstXMesh((size_t)dstXMesh.rows() - 1, (size_t)dstXMesh.cols() - 1) << std::endl;
                //std::cout << "dstYMesh(0,0):" << dstYMesh(0, 0) << ", " << "dstYMesh(1,0):" << dstYMesh(1, 0) << ", " << "dstYMesh(0,1):" << dstYMesh(0, 1) << std::endl;
                //std::cout << "dstYMesh(0,0):" << dstYMesh(0, 0) << ", " << "dstYMesh(-1,-1):" << dstYMesh((size_t)dstYMesh.rows() - 1, (size_t)dstYMesh.cols() - 1) << std::endl;

                //cv::Mat testSlice(dicomDataTags["Rows"], dicomDataTags["Columns"], CV_8U);
                //std::memcpy(testSlice.data, &dicomDataImage[0], (size_t)dicomDataTags["Rows"] * (size_t)dicomDataTags["Columns"]);
                //cv::imshow("tested image", testSlice);
                //cv::waitKey(0);
                //cv::Mat cvMat(dstRangeMesh.rows(), dstRangeMesh.cols(), CV_64F);
                //for (int i = 0; i < dstRangeMesh.rows(); ++i) {
                //    for (int j = 0; j < dstRangeMesh.cols(); ++j) {
                //        cvMat.at<double>(i, j) = dstRangeMesh(i, j);
                //    }
                //}
                //cv::imwrite("dstRangeMesh.png", cvMat);
                //for (int i = 0; i < dstAngleMesh.rows(); ++i) {
                //    for (int j = 0; j < dstAngleMesh.cols(); ++j) {
                //        cvMat.at<double>(i, j) = dstAngleMesh(i, j);
                //    }
                //}
                //cv::imwrite("dstAngleMesh.png", cvMat);
            }
        }


        //if ((nrrdFullPathName.find(".nrrd") != std::string::npos))
        //{
        //    outDataNameViewName = GetNrrdDataNameViewName(nrrdFullPathName);
        //    dataName = std::get<0>(outDataNameViewName);
        //    viewName = std::get<1>(outDataNameViewName);
        //    nrrdDataName = std::get<2>(outDataNameViewName);
        //    outHeaderData = readNrrd(nrrdFullPathName);
        //    keyOrder = std::get<0>(outHeaderData);
        //    header = std::get<1>(outHeaderData);
        //    nrrdData = std::get<2>(outHeaderData);

        //    //// print header
        //    //std::cout << "\n" << std::endl;
        //    //for (const std::pair<std::string, std::string>& pair : header)
        //    //{
        //    //    std::cout << pair.first << "       " << pair.second << std::endl;
        //    //}
        //    //std::cout << "\n" << std::endl;
        //    //for (const auto& keyString : essentialKeys)
        //    //    std::cout << "header[" << keyString << "]= " << header[keyString] << std::endl;

        //    // get info from header string
        //    dimension = std::stoi(header["dimension"]);
        //    std::regex whiteSpaceDelimiter = std::regex(" ");
        //    std::vector<std::string> sizesStrings(std::sregex_token_iterator(header["sizes"].begin(), header["sizes"].end(), whiteSpaceDelimiter, -1), std::sregex_token_iterator());
        //    std::vector<int> sizes;
        //    for (int dimIdx = 0; dimIdx < dimension; ++dimIdx)
        //    {
        //        sizes.push_back(std::stoi(sizesStrings[dimIdx]));
        //    }

        //    dicomHeader["Rows"] = std::stof(sizesStrings[1]);
        //    dicomHeader["Columns"] = std::stof(sizesStrings[0]);
        //    dicomHeader["NumberOfFrames"] = std::stof(sizesStrings[2]);

        //    std::regex braceDelimiter = std::regex("\\(|\\)");
        //    std::string removedBracesString = std::regex_replace(header["space directions"], braceDelimiter, "");
        //    std::vector<std::string> spaceDirectionsStrings(std::sregex_token_iterator(removedBracesString.begin(), removedBracesString.end(), whiteSpaceDelimiter, -1), std::sregex_token_iterator());
        //    std::regex commaDelimiter = std::regex("\\,");
        //    std::vector<std::string> subSpaceDirectionsStrings(std::sregex_token_iterator(spaceDirectionsStrings[0].begin(), spaceDirectionsStrings[0].end(), commaDelimiter, -1), std::sregex_token_iterator());
        //    dicomHeader["PixelSpacingRow"] = std::stof(subSpaceDirectionsStrings[0]);
        //    subSpaceDirectionsStrings = std::vector<std::string>(std::sregex_token_iterator(spaceDirectionsStrings[1].begin(), spaceDirectionsStrings[1].end(), commaDelimiter, -1), std::sregex_token_iterator());
        //    dicomHeader["PixelSpacingCol"] = std::stof(subSpaceDirectionsStrings[1]);
        //    subSpaceDirectionsStrings = std::vector<std::string>(std::sregex_token_iterator(spaceDirectionsStrings[2].begin(), spaceDirectionsStrings[2].end(), commaDelimiter, -1), std::sregex_token_iterator());
        //    dicomHeader["SpacingBetweenSlices"] = std::stof(subSpaceDirectionsStrings[2]);

        //    //// draw image
        //    //cv::Mat img(dicomHeader["Rows"], dicomHeader["Columns"], CV_8U);
        //    //std::memcpy(img.data, &nrrdData[((size_t)dicomHeader["Rows"] * (size_t)dicomHeader["Columns"] * (size_t)dicomHeader["NumberOfFrames"] /2)], ((size_t)dicomHeader["Rows"] * (size_t)dicomHeader["Columns"]));
        //    //if (nrrdFullPathName.find(".seg.") != std::string::npos)
        //    //{
        //    //    img *= 255;
        //    //}
        //    //cv::imshow("test", img);
        //    //cv::waitKey(0);


        //    srcIndexWiseXMesh;              // Eigen::MatrixXd 
        //    srcIndexWiseYMesh;              // Eigen::MatrixXd 
        //    nrrdData;                       // std::vector<unsigned char>
        //    dicomHeader["Rows"];            // float
        //    dicomHeader["Columns"];         // float
        //    dicomHeader["NumberOfFrames"];  // float

        //    
        //    std::cout << "srcIndexWiseXMesh.rows()=" << srcIndexWiseXMesh.rows() << ", srcIndexWiseXMesh.cols()=" << srcIndexWiseXMesh.cols() << std::endl;
        //    std::cout << "srcIndexWiseXMesh.rows()=" << srcIndexWiseXMesh.rows() << ", srcIndexWiseXMesh.cols()=" << srcIndexWiseXMesh.cols() << std::endl;
        //    std::cout << "dicomHeader['Rows']=" << dicomHeader["Rows"] << ", dicomHeader['Columns']=" << dicomHeader["Columns"] << ", dicomHeader['NumberOfFrames']=" << dicomHeader["NumberOfFrames"] << std::endl;
        //    std::cout << "nrrdData.size()=" << nrrdData.size() << ", Rows x Columns x NumberOfFrames=" << dicomHeader["Rows"]* dicomHeader["Columns"]* dicomHeader["NumberOfFrames"] << std::endl;


        //    // write the volume as DICOM
        //    const std::string outputPath = "./" + StringReplace(nrrdDataName, ".nrrd", ".dcm");
        //    //Save3DDataAsDicom(outputPath, nrrdData, dicomHeader);
        //}
    }
    return 0;
}


__global__ void inverseScanConversionKernels(unsigned char* deviceOutputPlane, unsigned char** deviceVecVol, int srcRows, int srcCols, int numVectors, int dstRows, int dstCols, double* deviceIndexX, double* deviceIndexY)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= dstRows * dstCols)
        return;
    int x = (int)deviceIndexX[idx];
    int y = (int)deviceIndexY[idx];

    if (x < 0 || y < 0 || x >= (srcRows - 1) || y >= (srcCols - 1))
        deviceOutputPlane[idx] = 0;
    deviceOutputPlane[idx] = 0;
    for (int vec = 0; vec < numVectors; ++vec)
    {
        deviceOutputPlane[idx] = deviceVecVol[vec][x * srcCols + y];
    }
}


cudaError_t inverseScanConversionWithCuda(std::vector<unsigned char>& outputVecVol, std::vector<unsigned char>& nrrdData, unsigned int srcRows, unsigned int srcColumns, unsigned int numberOfFrames, unsigned int dstRows, unsigned int dstColumns, std::vector<double> srcIndexWiseXVec, std::vector<double> srcIndexWiseYVec)
{
    cudaError_t cudaStatus;
    unsigned char* deviceOutVol;
    double* deviceIndexX;
    double* deviceIndexY;

    std::vector<unsigned char*> deviceVecEachSlice(numberOfFrames, nullptr);
    unsigned char** deviceVecVol;

    // Select GPU
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed! Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // malloc for output
    cudaStatus = cudaMalloc(&deviceOutVol, outputVecVol.size() * sizeof(unsigned char));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed for deviceOutVol!");
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
    outputVecVol.resize((size_t)dstRows * (size_t)dstColumns);
    cudaStatus = cudaMemcpy(outputVecVol.data(), deviceOutVol, (size_t)dstRows * (size_t)dstColumns * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed for deviceOutVol to host!");
        goto Error;
    }

Error:
    cudaFree(deviceOutVol);
    cudaFree(deviceIndexX);
    cudaFree(deviceIndexY);
    for (int i = 0; i < numberOfFrames; ++i) {
        cudaFree(deviceVecEachSlice[i]);
    }
    cudaFree(deviceVecVol);
    return cudaStatus;
}