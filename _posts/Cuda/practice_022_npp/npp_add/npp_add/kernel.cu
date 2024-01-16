//
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "npp.h"
//#include <stdio.h>

#include <iostream>


int main() {
  // 이미지 크기 및 데이터를 정의합니다.
  int width = 640;  // 이미지 너비
  int height = 480;  // 이미지 높이
  int imageSize = width * height * sizeof(Npp8u);  // 이미지 데이터 크기

  // 소스 이미지와 대상 이미지를 위한 메모리를 할당합니다.
  Npp8u* srcImage = new Npp8u[imageSize];
  Npp8u* dstImage = new Npp8u[imageSize];

  // 이미지 데이터 초기화 (예: 여기에서는 단순히 흰 바탕 이미지로 초기화)
  memset(srcImage, 255, imageSize);  // 흰 바탕 이미지로 초기화

  // 이미지 데이터를 복사합니다.
  NppiSize imageSizeROI = { width, height };
  int imageStep = width * sizeof(Npp8u);

  nppiCopy_8u_C1R(srcImage, imageStep, dstImage, imageStep, imageSizeROI);

  //// 결과를 확인하기 위해 이미지 데이터를 파일로 저장하거나 출력할 수 있습니다.
  //// 여기에 추가 작업을 수행할 수도 있습니다.

  //// 메모리를 해제합니다.
  //delete[] srcImage;
  //delete[] dstImage;

  return 0;
}

