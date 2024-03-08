#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <string>
#include <iostream>
#include <vector>
#include <regex>
#include "dcmtk/config/osconfig.h"
#include "dcmtk/dcmdata/dctk.h"
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <map>
#include "kernel.h"


std::tuple <std::string, std::string> getNrrdDataNameViewName(const std::string& pathName)
{
    std::regex delimiter("\\\\");
    std::vector<std::string> parts(std::sregex_token_iterator(pathName.begin(), pathName.end(), delimiter, -1), std::sregex_token_iterator());
    std::string fileName = parts[parts.size()-1];
    std::regex fileNameDelimiter("_|\\.");
    std::vector<std::string> fileNameParts(std::sregex_token_iterator(fileName.begin(), fileName.end(), fileNameDelimiter, -1), std::sregex_token_iterator());
    std::string groupName = fileNameParts[0];
    std::string dataName = fileNameParts[1];
    std::tuple <std::string, std::string> outTuple;
    outTuple = std::make_tuple(groupName, dataName);
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
    if (!file) {
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


//void save3DDataAsDicom(const std::string& outputPath, const std::vector<unsigned char>& data, int width, int height, int depth)
//{
//    // Create an empty DICOM dataset
//    DcmFileFormat fileformat;
//    DcmDataset* dataset = fileformat.getDataset();
//    // Set DICOM attributes (e.g., image pixel data)
//    // Note: You need to adjust these attributes according to your DICOM requirements
//    // For example, you need to set attributes such as SOP Class UID, Study Instance UID, etc.
//    // Refer to the DCMTK documentation for more details on DICOM attributes
//    int bitsAllocated = 8;
//    int bitsStored = 8;
//    int highBit = 7;
//    const Uint16* constHeight = static_cast<const Uint16*>(static_cast<void*>(&height));
//    const Uint16* constWidth = static_cast<const Uint16*>(static_cast<void*>(&width));
//    const Uint16* constDepth = static_cast<const Uint16*>(static_cast<void*>(&depth));
//    const Uint16* constBitsAllocated = static_cast<const Uint16*>(static_cast<void*>(&bitsAllocated));
//    const Uint16* constBitsStored = static_cast<const Uint16*>(static_cast<void*>(&bitsStored));
//    const Uint16* constHighBit = static_cast<const Uint16*>(static_cast<void*>(&highBit));
//
//    dataset->putAndInsertString(DCM_PatientName, "John Doe");
//    dataset->putAndInsertString(DCM_PatientID, "123456");
//    dataset->putAndInsertUint16Array(DCM_Rows, constHeight, 0);
//    dataset->putAndInsertUint16Array(DCM_Columns, constWidth, 0);
//    dataset->putAndInsertUint16Array(DCM_NumberOfFrames, constDepth, 0);
//    dataset->putAndInsertUint16Array(DCM_BitsAllocated, constBitsAllocated, 0);
//    dataset->putAndInsertUint16Array(DCM_BitsStored, constBitsStored, 0);
//    dataset->putAndInsertUint16Array(DCM_HighBit, constHighBit, 0);
//    dataset->putAndInsertUint16Array(DCM_PixelRepresentation, 0, 0);
//    // Convert 3D data to DICOM pixel data format
//    DcmPixelData* pixData = new DcmPixelData(DCM_PixelData);
//    pixData->putUint8Array(data.data(), data.size());
//    // Insert pixel data into dataset
//    dataset->insert(pixData, true);
//    // Save DICOM file
//    OFCondition status = fileformat.saveFile(outputPath.c_str(), EXS_LittleEndianExplicit);
//    if (status.bad()) {
//        std::cerr << "Error saving DICOM file: " << status.text() << std::endl;
//    }
//    else {
//        std::cout << "DICOM file saved successfully." << std::endl;
//    }
//}


int main(int argc, char* argv[])
{
    const char* folderPath = argv[1];
    OFString folderPathStr;
    folderPathStr = folderPath;
    std::string stdFolderPath = (std::string)folderPath;

    // get filelist in subfolders
    OFList<OFString> fileList;
    OFStandard::searchDirectoryRecursively(folderPathStr, fileList);

    // declaration
    std::string fullPathName;
    std::tuple <std::string, std::string> outDataNameViewName;
    std::tuple <std::vector<std::string>, std::map<std::string, std::string>, std::vector<unsigned char>> outHeaderData;
    std::string dataName;
    std::string viewName;
    std::vector<std::string> keyOrder;
    std::map<std::string, std::string> header;
    std::vector<unsigned char> nrrdData;
    std::vector<std::string> essentialKeys = {"dimension", "sizes", "type", "encoding"};
    int dimension;
    for (const auto& fileName : fileList)
    {
        fullPathName = (std::string)fileName.c_str();
        if (fullPathName.find(".nrrd") != std::string::npos)
        {
            outDataNameViewName = getNrrdDataNameViewName(fullPathName);
            dataName = std::get<0>(outDataNameViewName);
            viewName = std::get<1>(outDataNameViewName);
            outHeaderData = readNrrd(fullPathName);
            keyOrder = std::get<0>(outHeaderData);
            header = std::get<1>(outHeaderData);
            nrrdData = std::get<2>(outHeaderData);

            // print header
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
            std::regex delimiter(" ");
            std::vector<std::string> sizesStrings(std::sregex_token_iterator(header["sizes"].begin(), header["sizes"].end(), delimiter, -1), std::sregex_token_iterator());
            std::vector<int> sizes;
            for (int dimIdx = 0; dimIdx < dimension; ++dimIdx)
            {
                sizes.push_back(std::stoi(sizesStrings[dimIdx]));
            }

            // write the volume as DICOM




            // draw image
            cv::Mat img(sizes[1], sizes[0], CV_8U);
            std::memcpy(img.data, &nrrdData[((size_t)sizes[0] * (size_t)sizes[1] * (size_t)sizes[2]/2)], ((size_t)sizes[0] * (size_t)sizes[1]));
            if (fullPathName.find(".seg.") != std::string::npos)
            {
                img *= 255;
            }
            cv::imshow("test", img);
            cv::waitKey(0);
        }
    }
    return 0;
}
