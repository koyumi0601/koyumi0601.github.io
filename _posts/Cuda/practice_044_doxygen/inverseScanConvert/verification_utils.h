#pragma once
#include <opencv2/opencv.hpp>
#include <Eigen>
#include <iostream>
#include <algorithm>
#include <vector>
#include <chrono>
#include <string>


void showVolumeVec(const std::vector<unsigned char>& volumeData, int numRows, int numCols, int sliceIdx);
void showMatrix(const Eigen::MatrixXd& eigenMatrix);
enum class MatOrientation { normal, twist, normalTranspose, twistTranspose };
template<typename T> void showMeshVec(const std::vector<T>& MeshVec, int cols, int rows, MatOrientation orientation);

/*!
 * \brief Print statistics of a given vector of doubles.
 * \param vec The vector of doubles to compute statistics for.
 *
 * This function computes and prints statistics such as mean, variance, minimum, and maximum values
 * of the elements in the given vector of doubles.
 */
void printVectorStats(const std::vector<double>& vec);

/*!
 * \brief Print statistics of a given Eigen matrix.
 * \param matrix The Eigen matrix to compute statistics for.
 *
 * This function computes and prints statistics such as mean, variance, minimum, and maximum values
 * of the elements in the given Eigen matrix.
 */
void printMatrixStats(const Eigen::MatrixXd& matrix);

/**
 * @brief Macro for printing array elements.
 *
 * This macro prints the name and elements of the given array.
 * @param var The array whose elements will be printed.
 *
 * Here is an example:
 * @code{.cpp}
 * std::vector<int> vec = {1, 2, 3, 4, 5};
 * PRINTARRAY(vec);
 * // Output: vec:
 * // 1 2 3 4 5
 * @endcode
 * 
 */
#define PRINTARRAY(var) { std::cout << #var << ": " << std::endl; \
                               for (const auto& elem : var) { \
                                   std::cout << elem << " "; \
                               } \
                               std::cout << std::endl; \
                             }

/**
 * @brief Macro for printing scalar values.
 *
 * This macro prints the name and value of the given variable.
 * @param var The variable whose scalar value will be printed.
 *
 * Here is an example:
 * @code{.cpp}
 * int x = 10;
 * PRINTSCALAR(x);
 * // Output: x: 10
 * @endcode
 *
 */
#define PRINTSCALAR(var) std::cout << #var << ": " << var << std::endl;


class Timer {
private:
  std::chrono::high_resolution_clock::time_point start;
  std::chrono::high_resolution_clock::time_point end;
  std::chrono::duration<double, std::milli> duration;
  std::string taskName;

public:
  Timer();
  ~Timer();
  void on(const std::string& name);
  void elapsed();
};
