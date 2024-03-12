#include "example01.h"

void exampleVector() {
  utils::Timer timer;
  timer.on("This is start!");
  timer.elapsed();

  size_t size_x = 1024;
  size_t size_y = 1024;
  std::vector<float> Vector1dPtr(size_y, 0.0);
  std::vector<std::vector<float >> Matrix2dPtr(size_x, Vector1dPtr);

  // 1d vector
  Vector1dPtr[0] = 1.0;
  timer.on("Vector1dPtr[0]");
  std::cout << Vector1dPtr[0] << std::endl; // [] operator. 제일 빠르다고 한다. 근데 실제로 재보면 제일 느린데.
  timer.elapsed();
  timer.on("Vector1dPtr.at(0)");
  std::cout << Vector1dPtr.at(0) << std::endl; // at() 함수. 범위 안에 있는 지 확인하고 벗어나는 경우 std::out_of_range 예외 발생
  timer.elapsed();
  timer.on("Vector1dPtr.begin()");
  std::vector<float>::iterator it = Vector1dPtr.begin(); // iterator. Vector1dPtr.begin(), Vector1dPtr.end()
  std::cout << *it << std::endl;
  timer.elapsed();

  // 2d matrix
  Matrix2dPtr[0][0] = 2.0;
  timer.on("Matrix2dPtr[0][0]");
  std::cout << Matrix2dPtr[0][0] << std::endl;
  timer.elapsed();
  timer.on("Matrix2dPtr.at(0).at(0)");
  std::cout << Matrix2dPtr.at(0).at(0) << std::endl;
  timer.elapsed();
  timer.on("Matrix2dPtr.begin(), row_it->begin()");
  std::vector<std::vector<float>>::iterator row_it = Matrix2dPtr.begin();
  std::vector<float>::iterator col_it = row_it->begin();
  std::cout << *col_it << std::endl;
  timer.elapsed();
}
