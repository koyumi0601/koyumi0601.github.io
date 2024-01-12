// usedll.cpp : This file contains the 'main' function. Program execution begins and ends there.
//
#include "math_clang.h"
#include <iostream>
#include <iostream>
#include <ctime>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <array>

int main() {
  // Python과 동일한 데이터 및 변수 초기화
  int pnsz[] = { 300, 1024, 760 };
  float mul = static_cast<float>(std::rand()) / RAND_MAX;
  float add = static_cast<float>(std::rand()) / RAND_MAX;
  float* src = new float[pnsz[0] * pnsz[1] * pnsz[2]];

  // 데이터 초기화 (예: 랜덤 값 채우기)
  for (int i = 0; i < pnsz[0] * pnsz[1] * pnsz[2]; i++) {
    src[i] = static_cast<float>(std::rand()) / RAND_MAX;
  }

  // 작업 시작 시간 기록
  clock_t start_time = clock();

  // C++ 함수 호출하여 연산 수행
  float* dst = new float[pnsz[0] * pnsz[1] * pnsz[2]];

  // mul_const 함수 호출
  mul_const(dst, src, mul, pnsz);

  // add_const 함수 호출
  add_const(dst, dst, add, pnsz);

  // mul_mat 함수 호출
  mul_mat(dst, dst, dst, pnsz);

  // add_mat 함수 호출
  add_mat(dst, dst, dst, pnsz);

  // 작업 종료 시간 기록
  clock_t end_time = clock();
  double elapsed_time = static_cast<double>(end_time - start_time) / CLOCKS_PER_SEC;

  // 결과 출력
  std::cout << "elapsed time C++: " << elapsed_time << " seconds" << std::endl;
  std::cout << "Result: " << dst[0] << std::endl;

  // 메모리 해제
  delete[] src;
  delete[] dst;

  return 0;
}