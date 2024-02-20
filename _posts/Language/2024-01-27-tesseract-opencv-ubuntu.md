---
layout: single
title: "Tesseract OCR and OpenCV in Ubuntu"
categories: language
tags: [language, tesseract, ai, opencv, cpp, ubuntu, build]
toc: true
author_profile: false
sidebar:
    nav: "docs"
search: true

---

*Create cpp and build it using tesseract OCR and openCV in ubuntu*

# install openCV

```bash
sudo apt-get update
sudo apt-get install libopencv-dev
pkg-config --cflags --libs opencv4 # 설치 확인
```

- output

```bash
-I/usr/include/opencv4/opencv -I/usr/include/opencv4 -lopencv_stitching -lopencv_aruco -lopencv_bgsegm -lopencv_bioinspired -lopencv_ccalib -lopencv_dnn_objdetect -lopencv_dnn_superres -lopencv_dpm -lopencv_highgui -lopencv_face -lopencv_freetype -lopencv_fuzzy -lopencv_hdf -lopencv_hfs -lopencv_img_hash -lopencv_line_descriptor -lopencv_quality -lopencv_reg -lopencv_rgbd -lopencv_saliency -lopencv_shape -lopencv_stereo -lopencv_structured_light -lopencv_phase_unwrapping -lopencv_superres -lopencv_optflow -lopencv_surface_matching -lopencv_tracking -lopencv_datasets -lopencv_text -lopencv_dnn -lopencv_plot -lopencv_ml -lopencv_videostab -lopencv_videoio -lopencv_viz -lopencv_ximgproc -lopencv_video -lopencv_xobjdetect -lopencv_objdetect -lopencv_calib3d -lopencv_imgcodecs -lopencv_features2d -lopencv_flann -lopencv_xphoto -lopencv_photo -lopencv_imgproc -lopencv_core
```

- 환경변수 등록

```bash
export PKG_CONFIG_PATH=/usr/lib/x86_64-linux-gnu/pkgconfig:$PKG_CONFIG_PATH
```

# install tesseract

- 설치

```bash
sudo apt update
sudo add-apt-repository ppa:alex-p/tesseract-ocr-devel # 최신 개발 버전 5.3.4, 추론 정확도 높음.
# sudo add-apt-repository ppa:alex-p/tesseract-ocr # 안정화 버전 4.1, 추론 정확도 낮음.
sudo apt-get install tesseract-ocr
sudo apt install libtesseract-dev
```

- tessdata 다운로드 [https://github.com/tesseract-ocr/tessdata](https://github.com/tesseract-ocr/tessdata)

- train data 복사해서 붙여 넣기 /usr/local/share/tessdata


# build and run

- g++로 컴파일한다. 

```bash
g++ -O3 -std=c++11 OCR_depth.cpp $(pkg-config --cflags --libs opencv4 tesseract) -o OCR_depth
./OCR_depth ./resource/image_00001.png 
```


# code

```cpp
#include <string>
#include <tesseract/baseapi.h>
#include <leptonica/allheaders.h>
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

int main(int argc, char* argv[]) {
    if (argc != 2) {
        cout << "Usage: " << argv[0] << " <image_path>" << endl;
        return 1;
    }
    string outText, imPath = argv[1];
    Mat im = cv::imread(imPath, IMREAD_COLOR);
    if (im.empty()) {
        cout << "Failed to read the image." << endl;
        return 1;
    }
    tesseract::TessBaseAPI *ocr = new tesseract::TessBaseAPI();
    ocr->Init(NULL, "eng", tesseract::OEM_LSTM_ONLY);
    ocr->SetPageSegMode(tesseract::PSM_AUTO);
    ocr->SetImage(im.data, im.cols, im.rows, 3, im.step);
    ocr->SetSourceResolution(197); // image resolution (warning: invalid resolution 0 dpi. Using 70 instead. Estimatin resolution as 197)
    outText = string(ocr->GetUTF8Text());
    cout << outText;
    ocr->End();
    return 0;
}

```


# window

- visual studio가 설치되어 있어야 한다고 함.
- vcpkg(패키지 매니저) 다운받기 [https://github.com/microsoft/vcpkg/tree/master](https://github.com/microsoft/vcpkg/tree/master)
- 배치파일 실행 
```cmd
.\vcpkg\bootstrap-vcpkg.bat
```
- vcpkg 경로에서 cmd 열기
- vcpkg를 통해서 tesseract 설치
```bash
vcpkg install tesseract:x64-windows
# 꽤 오래 걸림 10 분 이상...?
```
- vcpkg에서 관리하는 installed경로로 이동 D:\Github_Blog\vcpkg\installed\x64-windows
    - 아래가 생성 되어 있음.
    - D:\Github_Blog\vcpkg\installed\tesseract_x64-windows\bin\tesseract53.dll 
    - D:\Github_Blog\vcpkg\installed\tesseract_x64-windows\include
    - D:\Github_Blog\vcpkg\installed\tesseract_x64-windows\lib\tesseract53.lib

- 참고(메인): https://m.blog.naver.com/killkimno/222338574004
- 참고(보조) https://tayrelgrin.blogspot.com/2021/08/mfc-windows-10-ocr-1tesseract.html


- CMakeLists.txt 수정
```cmake
if (UNIX AND NOT APPLE)
    find_package(PkgConfig REQUIRED)
    pkg_search_module(TESSERACT REQUIRED tesseract)
elseif(WIN32)
    # CMAKE_PREFIX_PATH, Tesseract_DIR  
    find_package(Tesseract REQUIRED PATHS "D:/Github_Blog/vcpkg/installed/tesseract_x64-windows")
    set(CMAKE_PREFIX_PATH "D:/Github_Blog/vcpkg/installed/tesseract_x64-windows")
endif()

include_directories(${TESSERACT_INCLUDE_DIRS})

if(UNIX AND NOT APPLE)
    target_link_libraries(MyProject PRIVATE tesseract)
elseif(WIN32)
    target_link_libraries(MyProject PRIVATE ${TESSERACT_LIBRARIES})
endif()
```

