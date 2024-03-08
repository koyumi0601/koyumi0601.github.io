#include "utils.h"


/// image plot

void displayEigenMatrixAsImage(const Eigen::MatrixXd& eigenMatrix)
{
  cv::Mat cvMat(eigenMatrix.rows(), eigenMatrix.cols(), CV_64FC1);
  for (int i = 0; i < eigenMatrix.rows(); ++i)
  {
    for (int j = 0; j < eigenMatrix.cols(); ++j)
    {
      cvMat.at<double>(i, j) = eigenMatrix(i, j);
    }
  }
  cv::normalize(cvMat, cvMat, 0, 255, cv::NORM_MINMAX, CV_8UC1);
  double aspectRatio = (double)cvMat.cols / cvMat.rows;
  int displayHeight = 400;
  int displayWidth = (int)(displayHeight * aspectRatio);
  int figureHeight = 600;
  int figureWidth = 1600;
  int xPadding = (figureWidth - displayWidth) / 2;
  int yPadding = (figureHeight - displayHeight) / 2;
  cv::Mat resizedImage;
  cv::resize(cvMat, resizedImage, cv::Size(displayWidth, displayHeight));
  cv::Mat imageWithPadding(figureHeight, figureWidth, CV_8UC1, cv::Scalar(255));
  cv::Rect roi(xPadding, yPadding, displayWidth, displayHeight);
  resizedImage.copyTo(imageWithPadding(roi));
  double fontsize = 0.4;
  int textthickness = 1;
  cv::Scalar color(64, 64, 64, 64);
  int leftoffset = 200;
  int col = 0;
  int row = 0;
  std::string text = "Matrix(row, col): x(cols)=" + std::to_string(col) + ", y(rows)=" + std::to_string(row) + ", [" + std::to_string(static_cast<double>(eigenMatrix(row, col))) + "]"; // rows, cols 순서
  cv::putText(imageWithPadding, text, cv::Point(xPadding - displayWidth / 2 - leftoffset, yPadding - 12), cv::FONT_HERSHEY_SIMPLEX, fontsize, color, textthickness);
  col = eigenMatrix.cols() - 1;
  row = 0;
  text = "Matrix(row, col): x(cols)=" + std::to_string(col) + ", y(rows)=" + std::to_string(row) + ", [" + std::to_string(static_cast<double>(eigenMatrix(row, col))) + "]";
  cv::putText(imageWithPadding, text, cv::Point(xPadding + displayWidth/2, yPadding - 12), cv::FONT_HERSHEY_SIMPLEX, fontsize, color, textthickness);
  col = 0;
  row = eigenMatrix.rows() - 1;
  text = "Matrix(row, col): x(cols)=" + std::to_string(col) + ", y(rows)=" + std::to_string(row) + ", [" + std::to_string(static_cast<double>(eigenMatrix(row, col))) + "]";
  cv::putText(imageWithPadding, text, cv::Point(xPadding - displayWidth / 2 - leftoffset, yPadding + displayHeight + 12), cv::FONT_HERSHEY_SIMPLEX, fontsize, color, textthickness);
  col = eigenMatrix.cols() - 1;
  row = eigenMatrix.rows() - 1;
  text = "Matrix(row, col): x(cols)=" + std::to_string(col) + ", y(rows)=" + std::to_string(row) + ", [" + std::to_string(static_cast<double>(eigenMatrix(row, col))) + "]";
  cv::putText(imageWithPadding, text, cv::Point(xPadding + displayWidth/2, yPadding + displayHeight + 12), cv::FONT_HERSHEY_SIMPLEX, fontsize, color, textthickness);
  cv::imshow("Eigen Matrix as Image", imageWithPadding);
  cv::waitKey(0);
}


template<typename T> void displayMeshVec2MatTwist(const std::vector<T>& MeshVec, int cols, int rows)
{
  cv::Mat Vec2Mat = cv::Mat::zeros(rows, cols, CV_64F);
  for (int i = 0; i < (cols * rows); i++)
  {
    int rowIdx = i % rows;
    int colIdx = i / rows;
    Vec2Mat.at<double>(rowIdx, colIdx) = static_cast<double>(MeshVec[i]);
  }
  cv::Mat normalizedMat;
  cv::normalize(Vec2Mat, normalizedMat, 0, 255, cv::NORM_MINMAX, CV_8UC1);
  cv::imshow("Vec2Mat row, col twist", normalizedMat);
  cv::waitKey(0);
}
template void displayMeshVec2MatTwist<double>(const std::vector<double>&, int, int);
template void displayMeshVec2MatTwist<float>(const std::vector<float>&, int, int);
template void displayMeshVec2MatTwist<unsigned char>(const std::vector<unsigned char>&, int, int);


template<typename T> void displayMeshVec2MatTwistT(const std::vector<T>& MeshVec, int cols, int rows)
{
  cv::Mat Vec2Mat = cv::Mat::zeros(cols, rows, CV_64F);
  for (int i = 0; i < (cols * rows); i++)
  {
    int rowIdx = i / rows;
    int colIdx = i % rows;
    Vec2Mat.at<double>(rowIdx, colIdx) = static_cast<double>(MeshVec[i]);
  }
  cv::Mat normalizedMat;
  cv::normalize(Vec2Mat, normalizedMat, 0, 255, cv::NORM_MINMAX, CV_8UC1);
  cv::imshow("Vec2Mat row, col twist, transpose", normalizedMat);
  cv::waitKey(0);
}
template void displayMeshVec2MatTwistT<double>(const std::vector<double>&, int, int);
template void displayMeshVec2MatTwistT<float>(const std::vector<float>&, int, int);
template void displayMeshVec2MatTwistT<unsigned char>(const std::vector<unsigned char>&, int, int);

template<typename T> void displayMeshVec2Mat(const std::vector<T>& MeshVec, int cols, int rows)
{
  cv::Mat Vec2Mat = cv::Mat::zeros(rows, cols, CV_64F);
  for (int i = 0; i < (cols * rows); i++)
  {
    int colIdx = i % cols;
    int rowIdx = i / cols;
    Vec2Mat.at<double>(rowIdx, colIdx) = static_cast<double>(MeshVec[i]);
  }
  cv::Mat normalizedMat;
  cv::normalize(Vec2Mat, normalizedMat, 0, 255, cv::NORM_MINMAX, CV_8UC1);
  double aspectRatio = (double)normalizedMat.cols / normalizedMat.rows; // 1176, 450
  int displayHeight = 400;
  int displayWidth = (int)(displayHeight * aspectRatio);
  int figureHeight = 600;
  int figureWidth = 1600;
  int xPadding = (figureWidth - displayWidth) / 2; // padding 100
  int yPadding = (figureHeight - displayHeight) / 2; // padding 200
  cv::Mat resizedImage;
  cv::resize(normalizedMat, resizedImage, cv::Size(displayWidth, displayHeight)); 
  //cv::imshow("resizedImage", resizedImage);
  cv::Mat imageWithPadding(figureHeight, figureWidth, CV_8UC1, cv::Scalar(255));
  cv::Rect roi(xPadding, yPadding, displayWidth, displayHeight); 
  resizedImage.copyTo(imageWithPadding(roi)); 
  //cv::imshow("vec2mat as Image", imageWithPadding);
  double fontsize = 0.4;
  int textthickness = 1;
  cv::Scalar color(64, 64, 64, 64);
  int stringLength = 300;
  int leftoffset = 300;
  int updownoffset = 200;
  int col = 0;
  int row = 0;
  normalizedMat.at<unsigned char>(row, col);
  std::string text = "Matrix(row, col): x(cols)=" + std::to_string(col) + ", y(rows)=" + std::to_string(row) + ", [" +std::to_string(static_cast<float>(Vec2Mat.at<double>(row, col))) + "]"; // rows, cols 순서
  std::cout << text << std::endl;
  cv::putText(imageWithPadding, text, cv::Point(xPadding - stringLength/2, yPadding - 12), cv::FONT_HERSHEY_SIMPLEX, fontsize, color, textthickness);
  col = normalizedMat.cols - 1;
  row = 0;
  text = "Matrix(row, col): x(cols)=" + std::to_string(col) + ", y(rows)=" + std::to_string(row) + ", [" + std::to_string(static_cast<float>(Vec2Mat.at<double>(row, col))) + "]";
  cv::putText(imageWithPadding, text, cv::Point(xPadding + displayWidth - stringLength/2, yPadding - 12), cv::FONT_HERSHEY_SIMPLEX, fontsize, color, textthickness);
  col = 0;
  row = normalizedMat.rows - 1;
  text = "Matrix(row, col): x(cols)=" + std::to_string(col) + ", y(rows)=" + std::to_string(row) + ", [" + std::to_string(static_cast<float>(Vec2Mat.at<double>(row, col))) + "]";
  cv::putText(imageWithPadding, text, cv::Point(xPadding - stringLength / 2, yPadding + displayHeight + 12), cv::FONT_HERSHEY_SIMPLEX, fontsize, color, textthickness);
  col = normalizedMat.cols - 1;
  row = normalizedMat.rows - 1;
  text = "Matrix(row, col): x(cols)=" + std::to_string(col) + ", y(rows)=" + std::to_string(row) + ", [" + std::to_string(static_cast<float>(Vec2Mat.at<double>(row, col))) + "]";
  cv::putText(imageWithPadding, text, cv::Point(xPadding + displayWidth - stringLength / 2, yPadding + displayHeight + 12), cv::FONT_HERSHEY_SIMPLEX, fontsize, color, textthickness);
  cv::imshow("vec2mat as Image, font", imageWithPadding);
  cv::waitKey(0);
}
template void displayMeshVec2Mat<double>(const std::vector<double>&, int, int); // input double, unsigned char
template void displayMeshVec2Mat<float>(const std::vector<float>&, int, int); // input double, unsigned char
template void displayMeshVec2Mat<unsigned char>(const std::vector<unsigned char>&, int, int); // input double, unsigned char

template<typename T> void displayMeshVec2MatT(const std::vector<T>& MeshVec, int cols, int rows)
{
  cv::Mat Vec2Mat = cv::Mat::zeros(cols, rows, CV_64F);
  for (int i = 0; i < (cols * rows); i++)
  {
    int colIdx = i / cols;
    int rowIdx = i % cols;
    Vec2Mat.at<double>(rowIdx, colIdx) = static_cast<double>(MeshVec[i]); // input double, unsigned char
  }

  cv::Mat normalizedMat;
  cv::normalize(Vec2Mat, normalizedMat, 0, 255, cv::NORM_MINMAX, CV_8UC1);
  cv::imshow("Vec2Mat transpose", normalizedMat);
  cv::waitKey(0);
}
template void displayMeshVec2MatT<double>(const std::vector<double>&, int, int); // input double, unsigned char
template void displayMeshVec2MatT<float>(const std::vector<float>&, int, int); // input double, unsigned char
template void displayMeshVec2MatT<unsigned char>(const std::vector<unsigned char>&, int, int); // input double, unsigned char

Eigen::MatrixXd getDicomImageFrameAndTranspose(const std::vector<unsigned char>& image, int frame, int rows, int columns)
{
  std::vector<unsigned char> frameData(image.begin() + frame * rows * columns, image.begin() + (frame + 1) * rows * columns);
  Eigen::MatrixXd matrix(columns, rows);
  for (int i = 0; i < rows; ++i)
  {
    for (int j = 0; j < columns; ++j)
    {
      matrix(j, i) = static_cast<double>(frameData[i * columns + j]); // need transpose to match matrix outside
    }
  }
  return matrix;
}


cv::Mat createImageWithColorbar(const cv::Mat& image)
{
  int colorbarWidth = 20;
  cv::Mat colorbar(image.rows, colorbarWidth, CV_8UC1);
  for (int i = 0; i < colorbar.rows; ++i) {
    uchar value = static_cast<uchar>(255 * (1 - static_cast<double>(i) / colorbar.rows));
    colorbar.row(i).setTo(cv::Scalar(value));
  }
  cv::Mat combined;
  cv::hconcat(image, colorbar, combined);
  return combined;
}

/// verification

void calculateVectorStats(const std::vector<double>& vec)
{
  if (!vec.empty())
  {
    auto minIt = std::min_element(vec.begin(), vec.end());
    auto maxIt = std::max_element(vec.begin(), vec.end());
    double minValue = *minIt;
    double maxValue = *maxIt;
    size_t size = vec.size();
    std::cout << "Minimum Value: " << minValue << std::endl;
    std::cout << "Maximum Value: " << maxValue << std::endl;
    std::cout << "Size: " << size << std::endl;
    std::cout << "Vector Values: ";
    for (const auto& value : vec)
    {
      std::cout << value << " ";
    }
    std::cout << std::endl;
  }
  else
  {
    std::cout << "The vector is empty." << std::endl;
  }
}


void calculateMatrixStats(const Eigen::MatrixXd& matrix)
{
  if (matrix.size() == 0)
  {
    std::cout << "The matrix is empty." << std::endl;
    return;
  }
  double minValue = matrix.minCoeff();
  double maxValue = matrix.maxCoeff();
  size_t rows = matrix.rows();
  size_t cols = matrix.cols();
  size_t size = rows * cols;
  std::cout << "Minimum Value: " << minValue << std::endl;
  std::cout << "Maximum Value: " << maxValue << std::endl;
  std::cout << "Size: " << size << " (Rows: " << rows << ", Columns: " << cols << ")" << std::endl;
  std::cout << "First Row: ";
  for (int j = 0; j < cols; ++j)
  {
    std::cout << matrix(0, j) << " ";
  }
  std::cout << std::endl;
  std::cout << "First Column: ";
  for (int i = 0; i < rows; ++i)
  {
    std::cout << matrix(i, 0) << " ";
  }
  std::cout << std::endl;
  int midRow = rows / 2;
  std::cout << "Central Row: ";
  for (int j = 0; j < cols; ++j)
  {
    std::cout << matrix(midRow, j) << " ";
  }
  std::cout << std::endl;
  int midCol = cols / 2;
  std::cout << "Central Column: ";
  for (int j = 0; j < rows; ++j)
  {
    std::cout << matrix(j, midCol) << " ";
  }
  std::cout << std::endl;
}


Eigen::MatrixXd flipLeftRightMatrix(const Eigen::MatrixXd& matrix)
{
  Eigen::MatrixXd flipped(matrix.rows(), matrix.cols());
  for (int i = 0; i < matrix.rows(); ++i)
  {
    for (int j = 0; j < matrix.cols(); ++j)
    {
      flipped(i, j) = matrix(i, matrix.cols() - 1 - j);
    }
  }
  return flipped;
}


/// utils

double findIndexInSortedArray(const std::vector<double>& src, double target)
{
  if (src.empty()) return -1;
  int size = src.size();
  bool isAscending = src.back() > src.front();
  double minVal = isAscending ? src.front() : src.back();
  double maxVal = isAscending ? src.back() : src.front();
  double scale = (size - 1) / (maxVal - minVal);
  double exactPosition;
  if (isAscending) {
    exactPosition = (target - src.front()) * scale;
  }
  else {
    exactPosition = (src.front() - target) * scale;
  }
  if ((exactPosition < 0) || (exactPosition >= size))
  {
    exactPosition = -1;
  }
  return exactPosition;
}


Eigen::MatrixXd computeIndexWiseMesh( const Eigen::MatrixXd& targetMesh, const std::vector<double>& sourceArray)
{
  Eigen::MatrixXd indexMesh(targetMesh.rows(), targetMesh.cols());
  int i, j;
  for (i = 0; i < targetMesh.rows(); ++i)
  {
    for (j = 0; j < targetMesh.cols(); ++j)
    {
      double targetValue = targetMesh(i, j);
      double exactPosition = findIndexInSortedArray(sourceArray, targetValue);
      indexMesh(i, j) = exactPosition;
    }
  }
  return indexMesh;
}


Eigen::MatrixXd createMaskMesh(const Eigen::MatrixXd& dstIndexWiseRangeMesh, const Eigen::MatrixXd& dstIndexWiseAngleMesh, const Eigen::MatrixXd& dicomStencilMask)
{
  int rows = dstIndexWiseRangeMesh.rows();
  int cols = dstIndexWiseRangeMesh.cols();
  Eigen::MatrixXd dstMaskMesh(rows, cols);
  dstMaskMesh.setZero();
  for (int i = 0; i < rows; ++i)
  {
    for (int j = 0; j < cols; ++j)
    {
      if (dstIndexWiseRangeMesh(i, j) < 0 || dstIndexWiseAngleMesh(i, j) < 0)
      {
        dstMaskMesh(i, j) = 0;
      }
      else {
        dstMaskMesh(i, j) = 1;
      }
    }
  }
  Eigen::MatrixXd dstMaskMeshMaxcut = bilinearInterpIndexWiseMesh(dstIndexWiseRangeMesh, dstIndexWiseAngleMesh, dstMaskMesh, dicomStencilMask);
  for (int i = 0; i < rows; ++i)
  {
    for (int j = 0; j < cols; ++j)
    {
      dstMaskMeshMaxcut(i, j) = std::round(dstMaskMeshMaxcut(i, j));
    }
  }
  return dstMaskMeshMaxcut;
}


Eigen::MatrixXd generateStencilMask(int height, int width, double maxCut)
{
  Eigen::MatrixXd stencil_mask = Eigen::MatrixXd::Zero(height, width);
  std::vector<double> clipSize = (linspace<double>(height * maxCut, 0, width));
  for (int j = 0; j < width; ++j) {
    int clip = std::round(clipSize[j]);
    for (int i = 0; i < height; ++i) {
      if ((i >= clip) && (i < height - clip)) {
        stencil_mask(i, j) = 1;
      }
    }
  }
  return stencil_mask;
}


std::vector<double> generateStencilMaskVec(int height, int width, double maxCut)
{
  std::vector<double> stencil_mask(height * width, 0.0);
  std::vector<double> clipSize = (linspace<double>(height * maxCut, 0, width));
  int index = 0;
  for (int j = 0; j < width; ++j) {
    int clip = std::round(clipSize[j]);
    for (int i = 0; i < height; ++i)
    {
      if ((i >= clip) && (i < height - clip))
      {
        index = i * width + j;
        stencil_mask[index] = 1;
      }
    }
  }
  return stencil_mask;
}


std::vector<unsigned char> generateStencilMaskVecTwist(int width, int height, double maxCut)
{
  std::vector<unsigned char> stencil_mask(height * width, 0.0);
  std::vector<double> clipSize = (linspace<double>(width * maxCut, 0, height));

  int index = 0;
  for (int j = 0; j < height; ++j)
  {
    int clip = std::round(clipSize[j]);
    for (int i = 0; i < width; ++i)
    {
      if ((i >= clip) && (i < width - clip))
      {
        index = j * width + i;
        stencil_mask[index] = 1;
      }
    }
  }
  return stencil_mask;
}


template<typename T> std::vector<T> arange(T start, T end, T step)
{
  std::vector<T> arange;
  T currentValue = start;

  while (currentValue <= end) {
    arange.push_back(currentValue);
    currentValue += step;
  }
  return arange;
}

template<typename T> std::vector<T> linspace(T start, T end, int num, bool endpoint)
{
  std::vector<T> result;
  T step = (endpoint) ? (end - start) / static_cast<T>(num - 1) : (end - start) / static_cast<T>(num);
  for (int i = 0; i < num; ++i)
  {
    result.push_back(start + static_cast<T>(i) * step);
  }
  return result;
}


std::pair<Eigen::MatrixXd, Eigen::MatrixXd> meshgrid(const std::vector<double>& vector1, const std::vector<double>& vector2) {
  Eigen::MatrixXd mesh1(vector1.size(), vector2.size());
  Eigen::MatrixXd mesh2(vector1.size(), vector2.size());
  for (int i = 0; i < vector1.size(); ++i) {
    for (int j = 0; j < vector2.size(); ++j) {
      mesh1(i, j) = vector1[i];
      mesh2(i, j) = vector2[j];
    }
  }
  return std::make_pair(mesh1, mesh2);
}

Eigen::MatrixXd bilinearInterpIndexWiseMesh
(
  const Eigen::MatrixXd& dstIndexWiseRangeMesh,
  const Eigen::MatrixXd& dstIndexWiseAngleMesh,
  const Eigen::MatrixXd& dstMaskMesh,
  const Eigen::MatrixXd& stencilDicomSingleFrame
)
{
  Eigen::MatrixXd dstOutputPlane(dstIndexWiseRangeMesh.rows(), dstIndexWiseRangeMesh.cols());
  for (int j = 0; j < dstIndexWiseRangeMesh.cols(); ++j) {
    for (int i = 0; i < dstIndexWiseRangeMesh.rows(); ++i) {
      double r = dstIndexWiseRangeMesh(i, j);
      double theta = dstIndexWiseAngleMesh(i, j);
      double q = 0;
      if (dstMaskMesh(i, j) == 1)
      {
        int r1 = (int)r;
        int theta1 = (int)theta;
        int r2 = r1 + 1;
        int theta2 = theta1 + 1;
        if (r1 == stencilDicomSingleFrame.cols() - 1)
        { // edge
          r2 = r1;
        }
        if (theta1 == stencilDicomSingleFrame.rows() - 1)
        { // edge
          theta2 = theta1;
        }
        double dr = r - static_cast<double>(r1);
        double dtheta = theta - static_cast<double>(theta1);
        if ((r > -1) && (theta > -1))
        {
          double q11 = stencilDicomSingleFrame(theta1, r1);
          double q21 = stencilDicomSingleFrame(theta2, r1);
          double q12 = stencilDicomSingleFrame(theta1, r2);
          double q22 = stencilDicomSingleFrame(theta2, r2);
          q = (int)(((1.0f - dtheta) * ((1.0f - dr) * q11 + dr * q12) + dtheta * ((1.0f - dr) * q21 + dr * q22)) + 0.5f);
        }
      }
      dstOutputPlane(i, j) = q;
    }
  }
  return dstOutputPlane;
}

std::vector<double> matrixXdToVector(const Eigen::MatrixXd& matrix)
{
  std::vector<double> vec;
  vec.reserve(matrix.size());
  for (int i = 0; i < matrix.rows(); ++i) {
    for (int j = 0; j < matrix.cols(); ++j) {
      vec.push_back(matrix(i, j));
    }
  }
  return vec;
}

cv::Mat matrixToMat(const Eigen::MatrixXd& matrix)
{
  int rows = matrix.rows();
  int cols = matrix.cols();
  cv::Mat mat(rows, cols, CV_8UC1);
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      double value = matrix(i, j);
      mat.at<unsigned char>(i, j) = static_cast<unsigned char>(value);
    }
  }
  return mat;
}


std::pair<std::vector<double>, std::vector<double>> generateXY(const std::vector<double>& srcRangeA, const std::vector<double>& srcAngleA, double targetResamplingUnitLength)
{
  double minSrcRangeA = *std::min_element(srcRangeA.begin(), srcRangeA.end());
  double maxSrcRangeA = *std::max_element(srcRangeA.begin(), srcRangeA.end());
  double minSrcAngleA = *std::min_element(srcAngleA.begin(), srcAngleA.end());
  double maxSrcAngleA = *std::max_element(srcAngleA.begin(), srcAngleA.end());
  double targetXMin = (-1.0) * std::round(std::abs(maxSrcRangeA * std::sin(minSrcAngleA * PI / 180.0) / targetResamplingUnitLength)) * targetResamplingUnitLength;
  double targetXMax = std::round(maxSrcRangeA * std::sin(maxSrcAngleA * PI / 180.0) / targetResamplingUnitLength) * targetResamplingUnitLength;
  double targetYMin = std::round(minSrcRangeA * std::cos(minSrcAngleA * PI / 180.0) / targetResamplingUnitLength) * targetResamplingUnitLength;
  double targetYMax = std::round(maxSrcRangeA / targetResamplingUnitLength) * targetResamplingUnitLength;
  std::vector<double> X = arange<double>(targetXMin, targetXMax, targetResamplingUnitLength);
  std::vector<double> Y = arange<double>(targetYMin, targetYMax, targetResamplingUnitLength);
  return std::make_pair(X, Y);
}