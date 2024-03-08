#include "functions.h"


#define PI 3.14159265358979323846


/**
 * @brief Generates a range of values with a specified step size.
 *
 * This template function generates a range of values starting from 'start' to 'end' (inclusive), with a specified step size.
 *
 * @tparam T Data type of the range values.
 * @param start Starting value of the range.
 * @param end Ending value of the range.
 * @param step Step size between consecutive values.
 * @return A vector containing the generated range of values.
 */
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

/**
 * @brief Generate evenly spaced values over a specified interval using linear interpolation.
 *
 * This function generates evenly spaced values over a specified interval using linear interpolation.
 *
 * @tparam T Data type of the vector elements.
 * @param start Starting value of the interval.
 * @param end Ending value of the interval.
 * @param num Number of samples to generate.
 * @param endpoint True to include the endpoint value in the result, false otherwise.
 * @return Vector containing the evenly spaced values.
 */
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

/**
 * @brief Generate 2D mesh grids from two vectors.
 *
 * This function generates two 2D mesh grids from two input vectors.
 *
 * @param vector1 First vector.
 * @param vector2 Second vector.
 * @return A pair of Eigen matrices representing the mesh grids.
 */
std::pair<Eigen::MatrixXd, Eigen::MatrixXd> meshgrid(const std::vector<double>& vector1, const std::vector<double>& vector2)
{
  Eigen::MatrixXd mesh1(vector1.size(), vector2.size());
  Eigen::MatrixXd mesh2(vector1.size(), vector2.size());
  for (int i = 0; i < vector1.size(); ++i)
  {
    for (int j = 0; j < vector2.size(); ++j)
    {
      mesh1(i, j) = vector1[i];
      mesh2(i, j) = vector2[j];
    }
  }
  return std::make_pair(mesh1, mesh2);
}

/**
 * @brief Calculate the index-wise position of a target value in a sorted vector.
 *
 * This function calculates the index-wise position of a target value in a sorted vector.
 *
 * @param src The sorted vector.
 * @param target The target value.
 * @return The index-wise position of the target value. Returns -1 if the vector is empty or the target value is out of range.
 */
double calculateIndexWisePosition(const std::vector<double>& src, double target)
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

/**
 * @brief Generate X and Y vectors based on source range and angle vectors.
 *
 * This function generates X and Y vectors based on the source range and angle vectors,
 * with a target resampling unit length.
 *
 * @param srcRangeA The source range vector.
 * @param srcAngleA The source angle vector.
 * @param targetResamplingUnitLength The target resampling unit length.
 * @return A pair of X and Y vectors.
 */
std::pair<std::vector<double>, std::vector<double>> generateXY(const std::vector<double>& srcRangeA, const std::vector<double>& srcAngleA, double targetResamplingUnitLength)
{
  double minSrcRangeA = *std::min_element(srcRangeA.begin(), srcRangeA.end());
  double maxSrcRangeA = *std::max_element(srcRangeA.begin(), srcRangeA.end());
  double minSrcAngleA = *std::min_element(srcAngleA.begin(), srcAngleA.end());
  double maxSrcAngleA = *std::max_element(srcAngleA.begin(), srcAngleA.end());
  double targetXMin = (-1.0) * std::round(std::abs(maxSrcRangeA * std::sin(minSrcAngleA * 3.14 / 180.0) / targetResamplingUnitLength)) * targetResamplingUnitLength;
  double targetXMax = std::round(maxSrcRangeA * std::sin(maxSrcAngleA * 3.14 / 180.0) / targetResamplingUnitLength) * targetResamplingUnitLength;
  double targetYMin = std::round(minSrcRangeA * std::cos(minSrcAngleA * 3.14 / 180.0) / targetResamplingUnitLength) * targetResamplingUnitLength;
  double targetYMax = std::round(maxSrcRangeA / targetResamplingUnitLength) * targetResamplingUnitLength;
  std::vector<double> X = arange<double>(targetXMin, targetXMax, targetResamplingUnitLength);
  std::vector<double> Y = arange<double>(targetYMin, targetYMax, targetResamplingUnitLength);
  return std::make_pair(X, Y);
}


Eigen::MatrixXd generateIndexWiseMesh(const Eigen::MatrixXd& targetMesh, const std::vector<double>& sourceArray)
{
  Eigen::MatrixXd indexMesh(targetMesh.rows(), targetMesh.cols());
  int i, j;
  for (i = 0; i < targetMesh.rows(); ++i)
  {
    for (j = 0; j < targetMesh.cols(); ++j)
    {
      double targetValue = targetMesh(i, j);
      double exactPosition = calculateIndexWisePosition(sourceArray, targetValue);
      indexMesh(i, j) = exactPosition;
    }
  }
  return indexMesh;
}

Eigen::MatrixXd generateMaskMesh(const Eigen::MatrixXd& dstIndexWiseRangeMesh, const Eigen::MatrixXd& dstIndexWiseAngleMesh, const Eigen::MatrixXd& dicomStencilMask)
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
  for (int j = 0; j < width; ++j)
  {
    int clip = std::round(clipSize[j]);
    for (int i = 0; i < height; ++i)
    {
      if ((i >= clip) && (i < height - clip))
      {
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

Eigen::MatrixXd bilinearInterpIndexWiseMesh
(
  const Eigen::MatrixXd& dstIndexWiseRangeMesh,
  const Eigen::MatrixXd& dstIndexWiseAngleMesh,
  const Eigen::MatrixXd& dstMaskMesh,
  const Eigen::MatrixXd& stencilDicomSingleFrame
)
{
  Eigen::MatrixXd dstOutputPlane(dstIndexWiseRangeMesh.rows(), dstIndexWiseRangeMesh.cols());
  for (int j = 0; j < dstIndexWiseRangeMesh.cols(); ++j)
  {
    for (int i = 0; i < dstIndexWiseRangeMesh.rows(); ++i)
    {
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

Eigen::MatrixXd convertDicomVolumeVecToMatrix(const std::vector<unsigned char>& image, int frame, int rows, int columns)
{
  std::vector<unsigned char> frameData(image.begin() + frame * rows * columns, image.begin() + (frame + 1) * rows * columns);
  Eigen::MatrixXd matrix(rows, columns);
  for (int i = 0; i < rows; ++i)
  {
    for (int j = 0; j < columns; ++j)
    {
      matrix(i, j) = static_cast<double>(frameData[i * columns + j]);
    }
  }
  return matrix;
}

std::vector<unsigned char> flipVolumeVector(std::vector<unsigned char>& imageVolume, int rows, int columns, int frames, bool flipMode)
{
  std::vector<unsigned char> flippedImageVolumeVector(imageVolume.size());
  for (int frame = 0; frame < frames; ++frame)
  {
    cv::Mat frameMat(rows, columns, CV_8U, &imageVolume[frame * rows * columns]);
    cv::Mat flippedFrameMat;
    cv::flip(frameMat, flippedFrameMat, flipMode);
    std::memcpy(&flippedImageVolumeVector[frame * rows * columns], flippedFrameMat.data, rows * columns * sizeof(unsigned char));
  }
  return flippedImageVolumeVector;
}