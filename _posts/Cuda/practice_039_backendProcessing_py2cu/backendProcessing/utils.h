#ifndef TRANSACTION_PARSER_H
#define TRANSACTION_PARSER_H


#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cmath>
#include <sstream>
#include <cstdint> // for uint8_t
#include <algorithm> // for std::clip
#include <opencv2/opencv.hpp>



struct TransactionParams {
  int TransactionId;
  int VersionNum;
  int SizeBytes;
  int StreamType;
  int StreamNum;
  int StreamDataType;
  int BeginChunkId;
  std::string BeginChunkIdTimestamp;
  int EndChunkId;
  std::string EndChunkIdTimestamp;
  int MaxChunkSizeBytes;
  int MaxNumChunks;
  double steeringAngleAzimuthDeg;
  std::string CineTimestamp;
  double radiusOfCurvatureElevationMm;
  double radiusOfCurvatureAzimuthMm;
  double elevationTwistRate;
  int StreamSubType;
  double AcquiredLateralMin;
  double AcquiredLateralSpan;
  double DisplayedLateralMin;
  double DisplayedLateralSpan;
  double AcquiredAxialMin;
  double AcquiredAxialSpan;
  double DisplayedAxialMin;
  double DisplayedAxialSpan;
  double AcquiredElevationMin;
  double AcquiredElevationSpan;
  double DisplayedElevationMin;
  double DisplayedElevationSpan;
  double ChunkDurationUs;
  double VectorApexElevZMm;
  double VectorApexAzimZMm;
  std::string AzimBeamSpacing;
  std::string ElevBeamSpacing;
  int NumBytesPerSample;
  int NumSamplesPerLine;
  int NumLinesPerSlice;
  int NumSlicesPerChunk;
  std::string FovShape;
};

TransactionParams parseTransactionParams(const std::string& filePath, bool debugMode);
std::pair<std::vector<std::vector<uint8_t>>, std::vector<std::vector<uint8_t>>> importData(const std::string& signalFileName, int targetFrame, const TransactionParams& params, bool debugMode);
std::vector<std::vector<uint8_t>> dynamicRange(const std::vector<std::vector<uint8_t>>& inputData, double pivotIn, double pivotOut, int dynamicRangeDb, int LogCompressStrengthDb, bool debugMode);
std::vector<std::vector<uint8_t>> grayMap(const std::vector<std::vector<uint8_t>>& inputData, const std::vector<int>& grayMapIdx, const std::vector<int>& grayMapValue, bool debugMode);
//std::vector<std::vector<uint8_t>> scanConversion(const std::vector<std::vector<uint8_t>>& inputData, const TransactionParams& params, int downscale = 1);

void scanConversion(const std::vector<std::vector<uint8_t>>& inputData, const TransactionParams& params, int downscale = 1);
//std::vector<std::vector<double>> curvedScanConversion(const std::vector<std::vector<uint8_t>>& inputData, const TransactionParams& params, int downscale);
void curvedScanConversion(const std::vector<std::vector<uint8_t>>& inputData, const TransactionParams& params, int downscale);

#endif // TRANSACTION_PARSER_H