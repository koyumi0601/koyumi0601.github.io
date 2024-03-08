#pragma once
#include <opencv2/opencv.hpp>
#include <Eigen>
#include <iostream>
#include <algorithm> // std::min_element, std::max_element
#include <vector>
const double PI = 3.14159265358979323846;

void displayEigenMatrixAsImage(const Eigen::MatrixXd& eigenMatrix);
template<typename T> void displayMeshVec2Mat(const std::vector<T>& MeshVec, int cols, int rows);
template<typename T> void displayMeshVec2MatT(const std::vector<T>& MeshVec, int cols, int rows);
template<typename T> void displayMeshVec2MatTwist(const std::vector<T>& MeshVec, int cols, int rows);
template<typename T> void displayMeshVec2MatTwistT(const std::vector<T>& MeshVec, int cols, int rows);
void calculateVectorStats(const std::vector<double>& vec);
void calculateMatrixStats(const Eigen::MatrixXd& matrix);
Eigen::MatrixXd flipLeftRightMatrix(const Eigen::MatrixXd& matrix);
Eigen::MatrixXd getDicomImageFrameAndTranspose(const std::vector<unsigned char>& image, int frame, int rows, int columns);
double findIndexInSortedArray(const std::vector<double>& src, double target);
Eigen::MatrixXd computeIndexWiseMesh(const Eigen::MatrixXd& targetMesh, const std::vector<double>& sourceArray);
cv::Mat createImageWithColorbar(const cv::Mat& image);
template<typename T> std::vector<T> arange(T start, T end, T step);
template<typename T> std::vector<T> linspace(T start, T end, int num, bool endpoint = true);
std::pair<Eigen::MatrixXd, Eigen::MatrixXd> meshgrid(const std::vector<double>& vector1, const std::vector<double>& vector2);
Eigen::MatrixXd createMaskMesh(const Eigen::MatrixXd& dstIndexWiseRangeMesh, const Eigen::MatrixXd& dstIndexWiseAngleMesh, const Eigen::MatrixXd& dicomStencilMask);
Eigen::MatrixXd generateStencilMask(int height, int width, double maxCut);
std::vector<double> generateStencilMaskVec(int height, int width, double maxCut);
std::vector<unsigned char> generateStencilMaskVecTwist(int height, int width, double maxCut);
Eigen::MatrixXd bilinearInterpIndexWiseMesh(const Eigen::MatrixXd& dstRangeIndexWiseMesh,
  const Eigen::MatrixXd& dstAngleIndexWiseMesh,
  const Eigen::MatrixXd& maskMesh,
  const Eigen::MatrixXd& stencilDicomSingleFrame);
std::vector<double> matrixXdToVector(const Eigen::MatrixXd& matrix);
cv::Mat matrixToMat(const Eigen::MatrixXd& matrix);
std::pair<std::vector<double>, std::vector<double>> generateXY(const std::vector<double>& srcRangeA, const std::vector<double>& srcAngleA, double targetResamplingUnitLength);

#define PRINTARRAY(var) { std::cout << #var << ": " << std::endl; \
                               for (const auto& elem : var) { \
                                   std::cout << elem << " "; \
                               } \
                               std::cout << std::endl; \
                             }
#define PRINTSCALAR(var) std::cout << #var << ": " << var << std::endl;
                               