#pragma once
#include <opencv2/opencv.hpp>
#include <Eigen>
#include <iostream>
#include <algorithm> // std::min_element, std::max_element
#include <vector>


template<typename T> std::vector<T> arange(T start, T end, T step);
template<typename T> std::vector<T> linspace(T start, T end, int num, bool endpoint = true);
std::pair<Eigen::MatrixXd, Eigen::MatrixXd> meshgrid(const std::vector<double>& vector1, const std::vector<double>& vector2);
double calculateIndexWisePosition(const std::vector<double>& src, double target);
std::pair<std::vector<double>, std::vector<double>> generateXY(const std::vector<double>& srcRangeA, const std::vector<double>& srcAngleA, double targetResamplingUnitLength);
Eigen::MatrixXd generateIndexWiseMesh(const Eigen::MatrixXd& targetMesh, const std::vector<double>& sourceArray);
Eigen::MatrixXd generateMaskMesh(const Eigen::MatrixXd& dstIndexWiseRangeMesh, const Eigen::MatrixXd& dstIndexWiseAngleMesh, const Eigen::MatrixXd& dicomStencilMask);
Eigen::MatrixXd generateStencilMask(int height, int width, double maxCut);
std::vector<double> generateStencilMaskVec(int height, int width, double maxCut);
std::vector<unsigned char> generateStencilMaskVecTwist(int height, int width, double maxCut);
Eigen::MatrixXd bilinearInterpIndexWiseMesh
( const Eigen::MatrixXd& dstRangeIndexWiseMesh,
  const Eigen::MatrixXd& dstAngleIndexWiseMesh,
  const Eigen::MatrixXd& maskMesh,
  const Eigen::MatrixXd& stencilDicomSingleFrame);
Eigen::MatrixXd convertDicomVolumeVecToMatrix(const std::vector<unsigned char>& image, int frame, int rows, int columns);
std::vector<unsigned char> flipVolumeVector(std::vector<unsigned char>& imageVolume, int rows, int columns, int frames, bool flipMode);
