#include "utils.h"


const int OUTGRIDX = 800; // Set your desired dimensions here
const int OUTGRIDY = 600; // Set your desired dimensions here


TransactionParams parseTransactionParams(const std::string& filePath, bool debugMode)
{
  std::cout << "parsing params..." << std::endl;
  std::ifstream file(filePath);
  if (!file.is_open())
  {
    std::cerr << "Fail : open file" << std::endl;
    exit(1);
  }
  TransactionParams params;
  std::string line;
  while (std::getline(file, line))
  {
    std::istringstream iss(line);
    std::string key, value;
    std::getline(iss, key, ':');
    std::getline(iss, value);
    key.erase(std::remove_if(key.begin(), key.end(), ::isspace), key.end());
    value.erase(std::remove_if(value.begin(), value.end(), ::isspace), value.end());
    if (key == "TransactionId")
    {
      params.TransactionId = std::stoi(value);
    }
    else if (key == "VersionNum")
    {
      params.VersionNum = std::stoi(value);
    }
    else if (key == "SizeBytes")
    {
      params.SizeBytes = std::stoi(value);
    }
    else if (key == "StreamType")
    {
      params.StreamType = std::stoi(value);
    }
    else if (key == "StreamNum")
    {
      params.StreamNum = std::stoi(value);
    }
    else if (key == "StreamDataType")
    {
      params.StreamDataType = std::stoi(value);
    }
    else if (key == "BeginChunkId")
    {
      params.BeginChunkId = std::stoi(value);
    }
    else if (key == "BeginChunkIdTimestamp")
    {
      params.BeginChunkIdTimestamp = value; // too big number to read
    }
    else if (key == "EndChunkId")
    {
      params.EndChunkId = std::stoi(value);
    }
    else if (key == "EndChunkIdTimestamp")
    {
      params.EndChunkIdTimestamp = value; // too big number to read
    }
    else if (key == "MaxChunkSizeBytes")
    {
      params.MaxChunkSizeBytes = std::stoi(value);
    }
    else if (key == "MaxNumChunks")
    {
      params.MaxNumChunks = std::stoi(value);
    }
    else if (key == "steeringAngleAzimuthDeg")
    {
      params.steeringAngleAzimuthDeg = std::stod(value);
    }
    else if (key == "CineTimestamp")
    {
      params.CineTimestamp = value; // too big number to read
    }
    else if (key == "radiusOfCurvatureElevationMm")
    {
      params.radiusOfCurvatureElevationMm = std::stod(value);
    }
    else if (key == "radiusOfCurvatureAzimuthMm")
    {
      params.radiusOfCurvatureAzimuthMm = std::stod(value);
    }
    else if (key == "elevationTwistRate")
    {
      params.elevationTwistRate = std::stod(value);
    }
    else if (key == "StreamSubType")
    {
      params.StreamSubType = std::stoi(value);
    }
    else if (key == "AcquiredLateralMin")
    {
      params.AcquiredLateralMin = std::stod(value);
    }
    else if (key == "AcquiredLateralSpan")
    {
      params.AcquiredLateralSpan = std::stod(value);
    }
    else if (key == "DisplayedLateralMin")
    {
      params.DisplayedLateralMin = std::stod(value);
    }
    else if (key == "DisplayedLateralSpan")
    {
      params.DisplayedLateralSpan = std::stod(value);
    }
    else if (key == "AcquiredAxialMin")
    {
      params.AcquiredAxialMin = std::stod(value);
    }
    else if (key == "AcquiredAxialSpan")
    {
      params.AcquiredAxialSpan = std::stod(value);
    }
    else if (key == "DisplayedAxialMin")
    {
      params.DisplayedAxialMin = std::stod(value);
    }
    else if (key == "DisplayedAxialSpan")
    {
      params.DisplayedAxialSpan = std::stod(value);
    }
    else if (key == "AcquiredElevationMin")
    {
      params.AcquiredElevationMin = std::stod(value);
    }
    else if (key == "AcquiredElevationSpan")
    {
      params.AcquiredElevationSpan = std::stod(value);
    }
    else if (key == "DisplayedElevationMin")
    {
      params.DisplayedElevationMin = std::stod(value);
    }
    else if (key == "DisplayedElevationSpan")
    {
      params.DisplayedElevationSpan = std::stod(value);
    }
    else if (key == "ChunkDurationUs")
    {
      params.ChunkDurationUs = std::stod(value);
    }
    else if (key == "VectorApexElevZMm")
    {
      params.VectorApexElevZMm = std::stod(value);
    }
    else if (key == "VectorApexAzimZMm")
    {
      params.VectorApexAzimZMm = std::stod(value);
    }
    else if (key == "AzimBeamSpacing")
    {
      params.AzimBeamSpacing = value;
    }
    else if (key == "ElevBeamSpacing")
    {
      params.ElevBeamSpacing = value;
    }
    else if (key == "NumBytesPerSample")
    {
      params.NumBytesPerSample = std::stoi(value);
    }
    else if (key == "NumSamplesPerLine")
    {
      params.NumSamplesPerLine = std::stoi(value);
    }
    else if (key == "NumLinesPerSlice")
    {
      params.NumLinesPerSlice = std::stoi(value);
    }
    else if (key == "NumSlicesPerChunk")
    {
      params.NumSlicesPerChunk = std::stoi(value);
    }
    else if (key == "FovShape")
    {
      params.FovShape = value;
    }
  }
  file.close();

  if (debugMode)
  {
    std::cout << std::endl;
    std::cout << "parseTransactionParams: " << params.FovShape << std::endl;
    std::cout << "\tFovShape: " << params.FovShape << std::endl;
    std::cout << "\tradiusOfCurvatureAzimuthMm: " << params.radiusOfCurvatureAzimuthMm << std::endl;
    std::cout << "\tDisplayedLateralMin: " << params.DisplayedLateralMin << std::endl;
    std::cout << "\tDisplayedLateralSpan: " << params.DisplayedLateralSpan << std::endl;
    std::cout << "\tDisplayedAxialMin: " << params.DisplayedAxialMin << std::endl;
    std::cout << "\tDisplayedAxialSpan: " << params.DisplayedAxialSpan << std::endl;
    std::cout << "\tNumLinesPerSlice: " << params.NumLinesPerSlice << std::endl;
    std::cout << "\tNumSamplesPerLine: " << params.NumSamplesPerLine << std::endl;
    std::cout << "\tAcquiredAxialSpan: " << params.AcquiredAxialSpan << std::endl;
  }
  return params;
}

std::pair<std::vector<std::vector<uint8_t>>, std::vector<std::vector<uint8_t>>> importData(const std::string& signalFileName, int targetFrame, const TransactionParams& params, bool debugMode)
{
  std::cout << "importing data..." << std::endl;
  int numLinesPerFrame = params.NumLinesPerSlice;
  int numSamplesPerLine = params.NumSamplesPerLine;
  double acquiredAxialSpan = params.AcquiredAxialSpan;
  double displayedAxialSpan = params.DisplayedAxialSpan;
  double samplePerMm = acquiredAxialSpan / numSamplesPerLine;
  int numDisplaySample = static_cast<int>(std::floor(displayedAxialSpan / samplePerMm));
  std::ifstream inFile(signalFileName, std::ios::binary);
  if (!inFile.is_open())
  {
    std::cerr << "Failed to open file." << std::endl;
    exit(1);
  }
  std::vector<uint8_t> cineData(std::istreambuf_iterator<char>(inFile), {});
  inFile.close();
  int countPerFrame = numLinesPerFrame * numSamplesPerLine;
  int startIdx = (targetFrame - 1) * countPerFrame;
  int endIdx = startIdx + countPerFrame;
  std::vector<std::vector<uint8_t>> acqSignalData(numSamplesPerLine, std::vector<uint8_t>(numLinesPerFrame));
  for (int i = 0; i < numSamplesPerLine; ++i)
  {
    for (int j = 0; j < numLinesPerFrame; ++j)
    {
      acqSignalData[i][j] = cineData[startIdx + j * numSamplesPerLine + i];
    }
  }
  std::vector<std::vector<uint8_t>> displaySignal(numDisplaySample, std::vector<uint8_t>(numLinesPerFrame));
  for (int i = 0; i < numDisplaySample; ++i)
  {
    for (int j = 0; j < numLinesPerFrame; ++j)
    {
      displaySignal[i][j] = acqSignalData[i][j];
    }
  }
  if (debugMode)
  {
    std::cout << "acqSignalData size: " << acqSignalData.size() << " x " << acqSignalData[0].size() << std::endl;
    std::cout << "displaySignal size: " << displaySignal.size() << " x " << displaySignal[0].size() << std::endl;
  }
  return { acqSignalData, displaySignal };
}


template<typename T>
constexpr const T& clamp(const T& v, const T& lo, const T& hi)
{
  return (v < lo) ? lo : (hi < v) ? hi : v;
}


std::vector<std::vector<uint8_t>> dynamicRange(const std::vector<std::vector<uint8_t>>& inputData, double pivotIn, double pivotOut, int dynamicRangeDb, int LogCompressStrengthDb, bool debugMode)
{
  std::cout << "calculating dynamic range..." << std::endl;
  double dbPerLsb = 255.0 / LogCompressStrengthDb;
  double dR = dynamicRangeDb * dbPerLsb;
  std::vector<std::vector<uint8_t>> drOut(inputData.size(), std::vector<uint8_t>(inputData[0].size()));
  for (size_t i = 0; i < inputData.size(); ++i)
  {
    for (size_t j = 0; j < inputData[i].size(); ++j)
    {
      double temp = (255.0 / dR) * (inputData[i][j] - 255.0 * pivotIn) + 255.0 * pivotOut;
      drOut[i][j] = static_cast<uint8_t>(clamp(temp, 0.0, 255.0));
    }
  }
  if (debugMode)
  {
    cv::Mat image(drOut.size(), drOut[0].size(), CV_8UC1);
    for (int i = 0; i < image.rows; ++i)
    {
      for (int j = 0; j < image.cols; ++j)
      {
        image.at<uint8_t>(i, j) = drOut[i][j];
      }
    }
    cv::imshow("Dynamic Range Image", image);
    cv::waitKey(0);
  }
  return drOut;
}


std::vector<std::vector<uint8_t>> grayMap(const std::vector<std::vector<uint8_t>>& inputData, const std::vector<int>& grayMapIdx, const std::vector<int>& grayMapValue, bool debugMode)
{
  std::cout << "calculating gray map..." << std::endl;
  std::vector<std::vector<uint8_t>> outData(inputData.size(), std::vector<uint8_t>(inputData[0].size()));
  for (size_t i = 0; i < inputData.size(); ++i)
  {
    for (size_t j = 0; j < inputData[0].size(); ++j)
    {
      int inputValue = inputData[i][j];
      // Linear interpolation
      int lowerIdx = std::lower_bound(grayMapIdx.begin(), grayMapIdx.end(), inputValue) - grayMapIdx.begin();
      if (lowerIdx == 0)
      {
        outData[i][j] = grayMapValue[0];
      }
      else if (lowerIdx == grayMapIdx.size())
      {
        outData[i][j] = grayMapValue.back();
      }
      else {
        int upperIdx = lowerIdx - 1;
        double t = static_cast<double>(inputValue - grayMapIdx[upperIdx]) / (grayMapIdx[lowerIdx] - grayMapIdx[upperIdx]);
        outData[i][j] = static_cast<uint8_t>((1.0 - t) * grayMapValue[upperIdx] + t * grayMapValue[lowerIdx]);
      }
    }
  }
  if (debugMode)
  {
    cv::Mat image(outData.size(), outData[0].size(), CV_8UC1);
    for (int i = 0; i < image.rows; ++i) {
      for (int j = 0; j < image.cols; ++j) {
        image.at<uint8_t>(i, j) = outData[i][j];
      }
    }
    cv::imshow("Gray Mapped Image", image);
    cv::waitKey(0);
  }
  return outData;
}


//std::vector<std::vector<uint8_t>> scanConversion(const std::vector<std::vector<uint8_t>>& inputData, const TransactionParams& params, int downscale) {

void scanConversion(const std::vector<std::vector<uint8_t>>& inputData, const TransactionParams& params, int downscale) {
  std::cout << "calculating scan conversion..." << std::endl;
  std::string fovShape = params.FovShape;
  std::vector<std::vector<uint8_t>> scOut;



  if (fovShape == "CurvedVector")
  {
    //scOut = curvedScanConversion(inputData, params, downscale);
    curvedScanConversion(inputData, params, downscale);
  }
  //else if (fovShape == "Linear")
  // {
  //  scOut = linearScanConversion(inputData, params);
  //}
  //else if (fovShape == "Vector")
  // {
  //  scOut = phasedScanConversion(inputData, params, downscale);
  //}

//  return scOut;
}

//std::vector<std::vector<double>> curvedScanConversion(const std::vector<std::vector<uint8_t>>& inputData, const TransactionParams& params, int downscale) {

void curvedScanConversion(const std::vector<std::vector<uint8_t>>& inputData, const TransactionParams& params, int downscale) {
  int numGridY = OUTGRIDY / downscale;
  int numGridX = OUTGRIDX / downscale;
  int numSamples = inputData.size();
  int numLines = inputData[0].size();
  double PI = std::acos(-1.0);

  double radiusOfCurvatureAzimuthMm = params.radiusOfCurvatureAzimuthMm;
  double DisplayedLateralMin = params.DisplayedLateralMin;
  double DisplayedLateralSpan = params.DisplayedLateralSpan;
  double DisplayedAxialMin = params.DisplayedAxialMin;
  double DisplayedAxialSpan = params.DisplayedAxialSpan;

  std::vector<double> rangeMmA(numSamples);
  for (int i = 0; i < numSamples; ++i)
  {
    rangeMmA[i] = radiusOfCurvatureAzimuthMm + DisplayedAxialMin + (DisplayedAxialSpan * i / numSamples);
  }

  std::vector<double> thetaA(numLines);
  for (int i = 0; i < numLines; ++i)
  {
    thetaA[i] = DisplayedLateralMin + (DisplayedLateralSpan * i / numLines);
  }

  double initDepthMm = (radiusOfCurvatureAzimuthMm + DisplayedAxialMin) * cos(DisplayedLateralMin / 180 * PI);
  double endDepthMm = radiusOfCurvatureAzimuthMm + DisplayedAxialMin + DisplayedAxialSpan;
  double initAzMm = (radiusOfCurvatureAzimuthMm + DisplayedAxialMin + DisplayedAxialSpan) * sin(DisplayedLateralMin / 180 * PI);
  double endAzMm = (radiusOfCurvatureAzimuthMm + DisplayedAxialMin + DisplayedAxialSpan) * sin((DisplayedLateralMin + DisplayedAxialSpan) / 180 * PI);

  std::vector<double> gridY(numGridY);
  std::vector<double> gridX(numGridX);
  std::vector<std::vector<double>> mcX(numGridY, std::vector<double>(numGridX));
  std::vector<std::vector<double>> mcY(numGridY, std::vector<double>(numGridX));
  std::vector<std::vector<double>> scRangeMmA(numGridY, std::vector<double>(numGridX));
  std::vector<std::vector<double>> scThetaA(numGridY, std::vector<double>(numGridX));
  for (int i = 0; i < numGridY; ++i)
  {
    gridY[i] = initDepthMm + ((endDepthMm - initDepthMm) * i / numGridY);
    for (int j = 0; j < numGridX; ++j)
    {
      gridX[j] = initAzMm + ((endAzMm - initAzMm) * j / numGridX);
      mcX[i][j] = gridX[j];
      mcY[i][j] = gridY[i];
      scRangeMmA[i][j] = sqrt(mcX[i][j] * mcX[i][j] + mcY[i][j] * mcY[i][j]);
      scThetaA[i][j] = atan(mcX[i][j] / mcY[i][j]) / PI * 180;
    }
  }

  std::vector<std::vector<double>> scOut(numGridY, std::vector<double>(numGridX, 0.0));

  for (int yIdx = 0; yIdx < numGridY; ++yIdx)
  {
    for (int xIdx = 0; xIdx < numGridX; ++xIdx)
    {
      double scTheta = scThetaA[yIdx][xIdx];
      double scRangeMm = scRangeMmA[yIdx][xIdx];
      if (scTheta >= DisplayedLateralMin && scTheta <= DisplayedLateralMin + DisplayedLateralSpan &&
        scRangeMm >= radiusOfCurvatureAzimuthMm && scRangeMm <= radiusOfCurvatureAzimuthMm + DisplayedAxialMin + DisplayedAxialSpan)
      {
        cv::Mat data(numLines, numSamples, CV_64F);
      //  for (int i = 0; i < numLines; ++i)
      //  {
      //    for (int j = 0; j < numSamples; ++j)
      //    {
      //      data.at<double>(i, j) = inputData[j][i]; // OpenCV는 행렬을 열-행 순서로 저장합니다.
      //    }
      //  }
      //  cv::Mat xCoords = cv::Mat::zeros(1, 1, CV_64F); // 보간할 좌표
      //  cv::Mat yCoords = cv::Mat::zeros(1, 1, CV_64F); // 보간할 좌표
      //  xCoords.at<double>(0, 0) = scTheta;
      //  yCoords.at<double>(0, 0) = scRangeMm;
      //  cv::Mat interpolatedValue;
      //  cv::remap(data, interpolatedValue, xCoords, yCoords, cv::INTER_LINEAR);
      //  scOut[yIdx][xIdx] = interpolatedValue.at<double>(0, 0);
      }
    }
  }

  //if (downscale != 1) {
  //  // Interpolation logic for downscaling
  //}

//  return scOut;
}