#include <iostream>
#include "utils.h"
#include <opencv2/opencv.hpp>

bool debugMode_parseTransactionParams = false;
bool debugMode_importData = false;
bool debugMode_dynamicRange = false;
bool debugMode_grayMap = false;



int main() {

  std::string filePath = "D:/Github_Blog/koyumi0601.github.io/_posts/Cuda/practice_039_backendProcessing_py2cu/showCineImage_py/exampledatas/Convex_6C1/BCine-0-49-Image-20220818_072712.trsc";
  std::string signalFileName = "D:/Github_Blog/koyumi0601.github.io/_posts/Cuda/practice_039_backendProcessing_py2cu/showCineImage_py/exampledatas/Convex_6C1/BCine-0-49-Image-20220818_072712.img";

  TransactionParams params = parseTransactionParams(filePath, debugMode_parseTransactionParams);

  int targetFrame = 1;
  int LogCompressStrengthDb = 96;
  int dr = 70;
  double pivotIn = 0.5;
  double pivotOut = 0.4;
  int downscale = 1;
  std::vector<int> grayMapIdx = { 0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100,105,110,115,120,125,130,135,140,145,150,155,160,165,170,175,180,185,190,195,200,205,210,215,220,225,230,235,240,245,250,255 };
  std::vector<int> grayMapValue = { 0,3,6,10,13,16,20,23,27,30,34,38,42,46,50,54,59,63,68,73,78,83,89,94,99,105,110,115,120,125,130,135,140,146,151,157,163,168,174,181,187,194,200,207,213,220,226,233,239,245,250,255 };

  auto result = importData(signalFileName, targetFrame, params, debugMode_importData);
  std::vector<std::vector<uint8_t>> acqSignalData = result.first;
  std::vector<std::vector<uint8_t>> displaySignal = result.second;
  std::vector<std::vector<uint8_t>> drOut = dynamicRange(displaySignal, pivotIn, pivotOut, dr, LogCompressStrengthDb, debugMode_dynamicRange);
  std::vector<std::vector<uint8_t>> grayMapped = grayMap(drOut, grayMapIdx, grayMapValue, debugMode_grayMap);
  //std::vector<std::vector<uint8_t>> scOut = scanConversion(grayMapped, params, downscale);
  scanConversion(grayMapped, params, downscale);


  return 0;
}