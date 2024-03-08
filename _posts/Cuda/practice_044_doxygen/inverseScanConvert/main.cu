#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <string>
#include <iostream>
#include <vector>
#include <array>
#include <regex>
#include <fstream>
#include <map>
#include "dcmtk/config/osconfig.h"
#include "dcmtk/dcmdata/dctk.h"
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <Eigen>
#include <algorithm>
#include <windows.h>
#include <direct.h>
#include <numeric>
#include "kernel_bilinearinterpolation.cuh"
#include "file_utils.h"
#include "functions.h"
#include "verification_utils.h"

DicomStruct ds;
const double targetPrecision = 0.05; // (mm)
const double targetResamplingUnitLength = 0.15; // (mm)


int main(int argc, char* argv[])
{
   Timer timer;


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
            dicomDataTuple = OpenDICOMAndGetDatasetMap(dicomFullPathName, ds);
            IsDicomCorrectlyRead = std::get<0>(dicomDataTuple);
            if (IsDicomCorrectlyRead)
            {
                // preprocess for inverse scanconverion
                dicomDataTags = std::get<1>(dicomDataTuple);
                dicomDataImage = std::get<2>(dicomDataTuple);
                int dicomNumRows = static_cast<int>(dicomDataTags["Rows"]); // 818
                int dicomNumColumns = static_cast<int>(dicomDataTags["Columns"]); // 889
                int dicomNumFrames = static_cast<int>(dicomDataTags["NumberOfFrames"]); // 340
                int dicomNumSamplesPerFrame = static_cast<int>(dicomNumRows * dicomNumColumns * dicomNumFrames); // 247248680
                double dicomMaxCut = static_cast<double>(dicomDataTags["MaxCut"]); // 0.0686164
                double dicomProbeRadiusMm = static_cast<double>(dicomDataTags["ProbeRadius"]); // 442.11
                double dicomRowSpacingMm = static_cast<double>(dicomDataTags["RowSpacing"]); // 0.073
                double dicomColSpacingMm = static_cast<double>(dicomDataTags["ColSpacing"]); //  0.2

                // srcRangeA, srcAngleA
                std::vector<double> srcRangeA = linspace<double>(dicomProbeRadiusMm, dicomProbeRadiusMm - (dicomRowSpacingMm * (dicomNumRows - 1)), dicomNumRows);
                double fovSpanDeg = (double) dicomNumColumns * dicomColSpacingMm / dicomProbeRadiusMm * 180.0 / PI ;
                std::vector<double> srcAngleA = linspace<double>(-fovSpanDeg/2.0f, fovSpanDeg/2.0f, dicomNumColumns, false); // endpoint=false
                // XY Grid
                std::pair<std::vector<double>, std::vector<double>> XY = generateXY(srcRangeA, srcAngleA, targetResamplingUnitLength);
                std::vector<double> X = XY.first;
                std::vector<double> Y = XY.second;

                // scan conversion, mask generation, gpu
                timer.on("scan conversion, mask generation, gpu");
                std::pair<std::vector<float>, std::vector<float>> IndexWiseRangeAngleMeshVec = generateIndexWiseRangeAngleMeshVecWithCuda(X, Y, srcRangeA, srcAngleA);
                std::vector<float> hostIndexWiseRangeMeshVec = IndexWiseRangeAngleMeshVec.first;
                std::vector<float> hostIndexWiseAngleMeshVec = IndexWiseRangeAngleMeshVec.second;
                //showMeshVec(hostIndexWiseRangeMeshVec, X.size(), Y.size(), MatOrientation::twist); // shape like XY

                std::vector<unsigned char> dicomMaxCutStencilVec = generateStencilMaskVecTwist(dicomNumColumns, dicomNumRows, dicomMaxCut); 
                //showMeshVec(dicomMaxCutStencilVec, dicomNumColumns, dicomNumRows, MatOrientation::normal); // shape like dicom data
                std::vector<unsigned char> hostMaskMeshVec(X.size() * Y.size(), 0.0);
                bilinearInterpolationWithCuda(hostMaskMeshVec, dicomMaxCutStencilVec, (unsigned int)dicomNumRows, (unsigned int)dicomNumColumns, (unsigned int)1, (unsigned int)X.size(), (unsigned int)Y.size(), hostIndexWiseAngleMeshVec, hostIndexWiseRangeMeshVec);
                //showMeshVec(hostMaskMeshVec, X.size(), Y.size(), MatOrientation::twist);
                timer.elapsed();

                // scan conversion, gpu
                timer.on("scan conversion, gpu (+flip volume)");
                std::vector<unsigned char> sCOutputVecVol((size_t)X.size() * (size_t)Y.size() * (size_t)dicomNumFrames);
                // with mask
                bilinearInterpolationWithCuda(sCOutputVecVol, dicomDataImage, (unsigned int)dicomNumRows, (unsigned int)dicomNumColumns, (unsigned int)dicomNumFrames, (unsigned int)X.size(), (unsigned int)Y.size(), hostIndexWiseAngleMeshVec, hostIndexWiseRangeMeshVec, hostMaskMeshVec);
                std::vector<unsigned char> flipedsCOutputVecVol = flipVolumeVector(sCOutputVecVol, X.size(), Y.size(), dicomNumFrames, 1);
                timer.elapsed();
                //showVolumeVec(flipedsCOutputVecVol, X.size(), Y.size(), 0);


                // scan conversion, mask generation, cpu
                timer.on("scan conversion, mask generation, cpu");
                std::pair<Eigen::MatrixXd, Eigen::MatrixXd> dstXYMesh = meshgrid(X, Y);
                Eigen::MatrixXd dstXMesh = dstXYMesh.first;
                Eigen::MatrixXd dstYMesh = dstXYMesh.second;
                Eigen::MatrixXd dstRangeMesh = (dstXMesh.array().square() + dstYMesh.array().square()).sqrt();
                Eigen::MatrixXd dstAngleMesh = (dstXMesh.array() / dstYMesh.array()).atan().array() / PI * 180.0f;
                Eigen::MatrixXd dstIndexWiseRangeMesh = generateIndexWiseMesh(dstRangeMesh, srcRangeA);
                Eigen::MatrixXd dstIndexWiseAngleMesh = generateIndexWiseMesh(dstAngleMesh, srcAngleA);
                Eigen::MatrixXd dicomMaxCutStencil = generateStencilMask(dicomNumColumns, dicomNumRows, dicomMaxCut);
                Eigen::MatrixXd dstMaskMesh = generateMaskMesh(dstIndexWiseRangeMesh, dstIndexWiseAngleMesh, dicomMaxCutStencil);
                //showMatrix(dstMaskMesh);
                timer.elapsed();

                // scan conversion, cpu
                timer.on("scan conversion, cpu");
                Eigen::MatrixXd frame, frameT;
                Eigen::MatrixXd dstOutputPlane;
                Eigen::MatrixXd stencildicomFrameTransposed;
                Eigen::MatrixXd dstOutputPlaneLrflip;
                for (int sliceIdx = 0; sliceIdx < dicomNumFrames; ++sliceIdx)
                //for (int sliceIdx = 0; sliceIdx < 1; ++sliceIdx) // test, first frame
                {
                  frame = convertDicomVolumeVecToMatrix(dicomDataImage, sliceIdx, dicomNumRows, dicomNumColumns);
                  frameT = frame.transpose();
                  //showMatrix(frameT);
                  dstOutputPlane = bilinearInterpIndexWiseMesh(dstIndexWiseRangeMesh, dstIndexWiseAngleMesh, dstMaskMesh, frameT);
                  dstOutputPlaneLrflip = dstOutputPlane.rowwise().reverse(); // left right flip
                  //showMatrix(dstOutputPlaneLrflip.transpose());
                }
                //printMatrixStats(dstOutputPlaneLrflip);
                timer.elapsed();
               

                // inverse scan conversion, mask generation, cpu
                timer.on("inverse scan conversion, mask generation, cpu");
                //srcRangeMesh, srcAngleMesh
                std::pair<Eigen::MatrixXd, Eigen::MatrixXd> resultMeshes = meshgrid(srcRangeA, srcAngleA);
                Eigen::MatrixXd srcRangeMesh = resultMeshes.first;
                Eigen::MatrixXd srcAngleMesh = resultMeshes.second;
                // srcYMesh, srcXMesh
                Eigen::MatrixXd srcYMesh = ((srcRangeMesh.array().square()) / ((srcAngleMesh.array() / 180.0f * PI).tan().square() + 1.0f)).sqrt();
                Eigen::MatrixXd srcXMesh = (srcAngleMesh.array() / 180.0f * PI).tan() * srcYMesh.array();
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
                timer.elapsed();


                for (const auto& nrrdFileName : nrrdFileList)
                {
                    nrrdFullPathName = (std::string)nrrdFileName.c_str();
                    if (nrrdFullPathName.find(".nrrd") != std::string::npos)
                    {
                        
                        outDataNameViewName = GetDataNameViewName(nrrdFullPathName);
                        dataName = std::get<0>(outDataNameViewName);
                        viewName = std::get<1>(outDataNameViewName);
                        nrrdDataName = std::get<2>(outDataNameViewName);
                        outHeaderData = readNrrd(nrrdFullPathName);
                        keyOrder = std::get<0>(outHeaderData);
                        header = std::get<1>(outHeaderData);
                        nrrdData = std::get<2>(outHeaderData);

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

                        std::vector<unsigned char> outputVecVol;
                        outputVecVol.resize((size_t)dicomHeader["Rows"] * (size_t)dicomHeader["Columns"] * (size_t)dicomHeader["NumberOfFrames"]);
                        //std::cout << "srcIndexWiseXVec.size()=" << srcIndexWiseXVec.size() << ", srcIndexWiseYVec.size()=" << srcIndexWiseYVec.size() << ", outputVecVol.size()=" << outputVecVol.size() << std::endl;
                        nrrdData = flipVolumeVector(nrrdData, dicomHeader["Rows"], dicomHeader["Columns"], dicomHeader["NumberOfFrames"], 0);
                        //inverseScanConversionWithCuda(outputVecVol, nrrdData, (unsigned int)dicomHeader["Rows"], (unsigned int)dicomHeader["Columns"], (unsigned int)dicomHeader["NumberOfFrames"], (unsigned int)srcIndexWiseXMesh.rows(), (unsigned int)srcIndexWiseXMesh.cols(), srcIndexWiseXVec, srcIndexWiseYVec);
                        timer.on("inverse scan conversion, gpu(kernel only)");
                        inverseScanConversionWithCuda(outputVecVol, nrrdData, (unsigned int)dicomHeader["Rows"], (unsigned int)dicomHeader["Columns"], (unsigned int)dicomHeader["NumberOfFrames"], (unsigned int)srcIndexWiseXMesh.rows(), (unsigned int)srcIndexWiseXMesh.cols(), srcIndexWiseXVec, srcIndexWiseYVec);
                        timer.elapsed();
                        //std::cout << "srcIndexWiseXMesh.rows()=" << srcIndexWiseXMesh.rows() << ", srcIndexWiseXMesh.cols()=" << srcIndexWiseXMesh.cols() << std::endl;
                        //std::cout << "srcIndexWiseXMesh.rows()=" << srcIndexWiseXMesh.rows() << ", srcIndexWiseXMesh.cols()=" << srcIndexWiseXMesh.cols() << std::endl;
                        //std::cout << "dicomHeader['Rows']=" << dicomHeader["Rows"] << ", dicomHeader['Columns']=" << dicomHeader["Columns"] << ", dicomHeader['NumberOfFrames']=" << dicomHeader["NumberOfFrames"] << std::endl;
                        //std::cout << "nrrdData.size()=" << nrrdData.size() << ", Rows x Columns x NumberOfFrames=" << dicomHeader["Rows"] * dicomHeader["Columns"] * dicomHeader["NumberOfFrames"] << std::endl;

                        ////size_t sliceIdx = ((size_t)dicomHeader["NumberOfFrames"] - 1);
                        //size_t sliceIdx = 0;
                        //cv::Mat img(srcIndexWiseXMesh.rows(), srcIndexWiseXMesh.cols(), CV_8U);
                        //std::memcpy(img.data, &outputVecVol[(size_t)srcIndexWiseXMesh.rows() * (size_t)srcIndexWiseXMesh.cols() * sliceIdx], ((size_t)srcIndexWiseXMesh.rows()* (size_t)srcIndexWiseXMesh.cols()));
                        //if (nrrdFullPathName.find(".seg.") != std::string::npos)
                        //{
                        //    img *= 255;
                        //}
                        //cv::imshow("inverse scan conversion", img);
                        //cv::Mat img2(srcIndexWiseXMesh.rows(), srcIndexWiseXMesh.cols(), CV_8U);
                        //std::memcpy(img2.data, &dicomDataImage[(size_t)srcIndexWiseXMesh.rows() * (size_t)srcIndexWiseXMesh.cols()* sliceIdx], ((size_t)srcIndexWiseXMesh.rows() * (size_t)srcIndexWiseXMesh.cols()));
                        //if (nrrdFullPathName.find(".seg.") != std::string::npos)
                        //{
                        //    img2 *= 255;
                        //}
                        //cv::imshow("org reference", img2);
                        //cv::waitKey(0);
                        // write the volume as DICOM
                        const std::string outputPath = "./" + StringReplace(nrrdDataName, ".nrrd", ".dcm");
                        //SaveNrrdAsDicom(outputPath, nrrdData, dicomHeader);
                        
                    }
                }
            }
        }
    }
    std::cin.get();
    return 0;
}


