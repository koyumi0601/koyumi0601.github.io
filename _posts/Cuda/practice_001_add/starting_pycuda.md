# CUDA C++ Programming Guide
: [https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)

# install and setting
- C/C++ 실행하기 블로그 [링크](https://blog.naver.com/PostView.naver?blogId=cylife3556&logNo=223055320907&parentCategoryNo=94&categoryNo=&viewDate=&isShowPopularPosts=true&from=search)
  - msys2 설치 [https://www.msys2.org/](https://www.msys2.org/)
  - g++, clang 둘 다 설치 
- 
- C/C++ compiler setting as MSVC: 
  - 일회성 (powershell) $env:CC = "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Tools\Llvm\x64\bin\clang.exe"
  - 
- cuda toolkit
```
nvcc --version
```

```
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2023 NVIDIA Corporation
Built on Mon_Apr__3_17:36:15_Pacific_Daylight_Time_2023
Cuda compilation tools, release 12.1, V12.1.105
Build cuda_12.1.r12.1/compiler.32688072_0
```
- visual studio code
- cuda extension
- project
- vs code terminal, compile and execute

# error case 1. nvcc fatal : cannot find compiler 'cl.exe' in path
- NVIDIA CUDA 컴파일러 (nvcc)가 Visual Studio C++ 컴파일러 (cl.exe)를 찾지 못해서 발생
- 해결방법
  - Visual Studio 설치 확인: visual studio installer 설치되어 있는지 확인 후 설치 경로를 알아낸다.
  - 시스템 환경 변수 설정: CUDA 컴파일러가 cl.exe를 찾을 수 있도록 시스템 환경 변수를 설정해야 합니다.
    - 시작 메뉴에서 "환경 변수 편집"을 검색하여 "환경 변수 편집"을 엽니다.
    - "시스템 변수" 섹션에서 "Path"를 선택하고 "편집"을 클릭합니다.
    - 환경 변수 편집 창에서 "새로 만들기"를 클릭하고 다음 경로를 추가합니다.
  ```
  #C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Tools\MSVC\14.30.30705\bin\Hostx64\x64
  D:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Tools\MSVC\14.38.33130\bin\Hostx64\x86\cl.exe
  ```
    - 주의: 경로는 설치된 Visual Studio 버전 및 위치에 따라 다를 수 있으며, 위 경로는 Visual Studio 2019 Community Edition의 예입니다. 설치된 Visual Studio 버전에 따라 경로를 조정해야 할 수 있습니다.
  - 컴퓨터 재부팅: 환경 변수 변경 사항이 적용되도록 컴퓨터를 재부팅합니다.


