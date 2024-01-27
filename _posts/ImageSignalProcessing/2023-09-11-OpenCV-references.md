---
layout: single
title: "Open CV Getting Started"
categories: imagesignalprocessing
tags: [Image Signal Processing, Open CV, Python]
toc: true
author_profile: false
sidebar:
    nav: "docs"
search: true
---

*Open CV References*

https://076923.github.io/posts/Python-opencv-3/

# Install, Ubuntu
- [https://simsamo.tistory.com/25](https://simsamo.tistory.com/25)

# Install, Window, C++
## 설치
- [https://www.opencv-srf.com/2017/11/install-opencv-with-visual-studio.html](https://www.opencv-srf.com/2017/11/install-opencv-with-visual-studio.html)
- [https://nowonbun.tistory.com/745](https://nowonbun.tistory.com/745)
- Go to official site / release [https://opencv.org/releases/](https://opencv.org/releases/)
- Download and install opencv-4.9.0-windows.exe
- D:\opencv\build > dll 가져다가 쓸 때
- D:\opencv\sources > opencv 라이브러리를 직접 수정해서 사용할 때
- 윈도우 환경변수 추가: OPENCV_DIR 새로 생성 d:\opencv\build
## visual studio project
- new project > console application
- project property > c/c++ > general > additional include directories > D:\opencv\build\include
- project property > linker > general > additional library derectories > D:\opencv\build\x64\vc16\lib
- project property > linker > input > additional dependencies > opencv_world480d.lib (d는 debug용)
- dll copy and paste from "D:\opencv\build\x64\vc16\bin\opencv_world480d.dll" to project folder


```cpp
#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

int main(int argc, char** argv)
{
  // Read the image file
  Mat image = imread("D:/Downloads/opencv_images/Eagle2.jpg");

  if (image.empty()) // Check for failure
  {
    cout << "Could not open or find the image" << endl;
    system("pause"); //wait for any key press
    return -1;
  }

  String windowName = "My HelloWorld Window"; //Name of the window

  namedWindow(windowName); // Create a window

  imshow(windowName, image); // Show our image inside the created window.

  waitKey(0); // Wait for any keystroke in the window

  destroyWindow(windowName); //destroy the created window

  return 0;
}
```