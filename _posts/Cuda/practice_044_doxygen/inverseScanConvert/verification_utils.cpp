#include "verification_utils.h"


void showVolumeVec(const std::vector<unsigned char>& volumeData, int numRows, int numCols, int sliceIdx)
{
  cv::Mat img(numRows, numCols, CV_8U);
  std::memcpy(img.data, &volumeData[(size_t)numRows * (size_t)numCols * sliceIdx], ((size_t)numRows * (size_t)numCols));
  cv::imshow("volume vector", img.t());
  cv::waitKey(0);
}

void showMatrix(const Eigen::MatrixXd& eigenMatrix)
{
  // Convert Eigen matrix to OpenCV Mat
  cv::Mat cvMat(eigenMatrix.rows(), eigenMatrix.cols(), CV_64FC1);
  for (int i = 0; i < eigenMatrix.rows(); ++i)
  {
    for (int j = 0; j < eigenMatrix.cols(); ++j)
    {
      cvMat.at<double>(i, j) = eigenMatrix(i, j);
    }
  }

  // Normalize and convert matrix values to 8-bit unsigned integers
  cv::normalize(cvMat, cvMat, 0, 255, cv::NORM_MINMAX, CV_8UC1);

  // Define display and figure dimensions
  double aspectRatio = (double)cvMat.cols / cvMat.rows;
  int displayHeight = 400;
  int displayWidth = (int)(displayHeight * aspectRatio);
  int figureHeight = 600;
  int figureWidth = 1600;
  int xPadding = (figureWidth - displayWidth) / 2;
  int yPadding = (figureHeight - displayHeight) / 2;

  // Resize the image and add padding
  cv::Mat resizedImage;
  cv::resize(cvMat, resizedImage, cv::Size(displayWidth, displayHeight));
  cv::Mat imageWithPadding(figureHeight, figureWidth, CV_8UC1, cv::Scalar(255));
  cv::Rect roi(xPadding, yPadding, displayWidth, displayHeight);
  resizedImage.copyTo(imageWithPadding(roi));

  // Add text annotations to the image
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
  cv::putText(imageWithPadding, text, cv::Point(xPadding + displayWidth / 2, yPadding - 12), cv::FONT_HERSHEY_SIMPLEX, fontsize, color, textthickness);
  col = 0;
  row = eigenMatrix.rows() - 1;
  text = "Matrix(row, col): x(cols)=" + std::to_string(col) + ", y(rows)=" + std::to_string(row) + ", [" + std::to_string(static_cast<double>(eigenMatrix(row, col))) + "]";
  cv::putText(imageWithPadding, text, cv::Point(xPadding - displayWidth / 2 - leftoffset, yPadding + displayHeight + 12), cv::FONT_HERSHEY_SIMPLEX, fontsize, color, textthickness);
  col = eigenMatrix.cols() - 1;
  row = eigenMatrix.rows() - 1;
  text = "Matrix(row, col): x(cols)=" + std::to_string(col) + ", y(rows)=" + std::to_string(row) + ", [" + std::to_string(static_cast<double>(eigenMatrix(row, col))) + "]";
  cv::putText(imageWithPadding, text, cv::Point(xPadding + displayWidth / 2, yPadding + displayHeight + 12), cv::FONT_HERSHEY_SIMPLEX, fontsize, color, textthickness);

  // Display the image
  cv::imshow("Eigen Matrix as Image", imageWithPadding);
  cv::waitKey(0);
}


template<typename T> void showMeshVec(const std::vector<T>& MeshVec, int cols, int rows, MatOrientation orientation)
{
  cv::Mat Vec2Mat;
  std::cout << "why!!\n";
  if (orientation == MatOrientation::normal)
  {
    Vec2Mat = cv::Mat::zeros(rows, cols, CV_64F);
    for (int i = 0; i < (cols * rows); i++)
    {
      int colIdx = i % cols;
      int rowIdx = i / cols;
      Vec2Mat.at<double>(rowIdx, colIdx) = static_cast<double>(MeshVec[i]);
    }
  }
  else if (orientation == MatOrientation::normalTranspose)
  {
    Vec2Mat = cv::Mat::zeros(cols, rows, CV_64F);
    for (int i = 0; i < (cols * rows); i++)
    {
      int colIdx = i / cols;
      int rowIdx = i % cols;
      Vec2Mat.at<double>(rowIdx, colIdx) = static_cast<double>(MeshVec[i]); // input double, unsigned char
    }
  }
  else if (orientation == MatOrientation::twist)
  {
    Vec2Mat = cv::Mat::zeros(rows, cols, CV_64F);
    for (int i = 0; i < (cols * rows); i++)
    {
      int rowIdx = i % rows;
      int colIdx = i / rows;
      Vec2Mat.at<double>(rowIdx, colIdx) = static_cast<double>(MeshVec[i]);

    }
  }
  else if (orientation == MatOrientation::twistTranspose)
  {
    Vec2Mat = cv::Mat::zeros(cols, rows, CV_64F);
    for (int i = 0; i < (cols * rows); i++)
    {
      int rowIdx = i / rows;
      int colIdx = i % rows;
      Vec2Mat.at<double>(rowIdx, colIdx) = static_cast<double>(MeshVec[i]);
    }
  }
  cv::Mat normalizedMat;
  cv::normalize(Vec2Mat, normalizedMat, 0, 255, cv::NORM_MINMAX, CV_8UC1);
  cv::imshow("MeshVec", normalizedMat);
  cv::waitKey(0);
}
template void showMeshVec<double>(const std::vector<double>&, int, int, MatOrientation);
template void showMeshVec<float>(const std::vector<float>&, int, int, MatOrientation);
template void showMeshVec<unsigned char>(const std::vector<unsigned char>&, int, int, MatOrientation);


void printVectorStats(const std::vector<double>& vec)
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


void printMatrixStats(const Eigen::MatrixXd& matrix)
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
  std::cout << "Minimum Value: " << minValue << " Maximum Value: " << maxValue << " Size: " << size << " Rows: " << rows << " Columns: " << cols << std::endl;
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


// Class Timer
Timer::Timer() : duration(0.0){}

Timer::~Timer() {}

void Timer::on(const std::string& name)
{
  start = std::chrono::high_resolution_clock::now();
  taskName = name;
}

void Timer::elapsed()
{
  end = std::chrono::high_resolution_clock::now();
  duration = end - start;
  std::cout << "Elapsed time for " << taskName << ": " << duration.count() << " ms\n";
}
