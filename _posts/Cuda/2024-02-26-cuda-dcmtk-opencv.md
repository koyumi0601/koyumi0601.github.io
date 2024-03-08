---
layout: single
title: "visual studio - cuda, dcmtk, opencv, eigen "
categories: cuda
tags: [language, programming, cpp, cuda]
toc: true
author_profile: false
sidebar:
    nav: "docs"
search: true

---

*visual studio - cuda, dcmtk, opencv, eigen 경로*


## header, library, dll path

```cpp
// dcmtk
D:\Github_Blog\Development_libraries\dcmtk_build\include
D:\Github_Blog\Development_libraries\dcmtk_build\lib
dcmtk.lib
D:\Github_Blog\Development_libraries\dcmtk_build\bin
dcmtk.dll
dcmtkcrypto_d-1_1-x64.dll
dcmtkcrypto-1_1-x64.dll
dcmtkssl_d-1_1-x64.dll
dcmtkssl-1_1-x64.dll
// opencv
D:\Github_Blog\Development_libraries\opencv\build\include
D:\Github_Blog\Development_libraries\opencv\build\x64\vc16\lib
opencv_world480d.lib
opencv_world480.lib
D:\Github_Blog\Development_libraries\opencv\build\x64\vc16\bin
opencv_world480.dll
opencv_world480d.dll
opencv_videoio_ffmpeg480_64.dll
opencv_videoio_msmf480_64.dll
opencv_videoio_msmf480_64d.dll
```

## post build event

```bash
# xcopy /Y (덮어쓰기) "path\*.dll" "$(TargetDir)"
xcopy /Y "D:\Github_Blog\Development_libraries\dcmtk_build\bin\*.dll" "$(TargetDir)"
xcopy /Y "D:\Github_Blog\Development_libraries\opencv\build\x64\vc16\bin\*.dll" "$(TargetDir)"
```


# Eigen - numerical optimization
- download [https://eigen.tuxfamily.org/index.php?title=Main_Page](https://eigen.tuxfamily.org/index.php?title=Main_Page)
    - zip 다운로드 받고 압축을 푼다.
    - cmake gui를 통해서 빌드솔루션을 만든다
    - visual studio로 build -> batch build -> all build(debug, release) -> install build(debug, release)
    - visual studio project에 eigen의 header만 include하면 된다. 라이브러리나 dll은 없다. 
        - D:\Github_Blog\Development_libraries\eigen-3.4.0-build\installed\include\eigen3\Eigen
- refer to [https://velog.io/@mir21c/App.2-Eigen-%EC%84%A4%EC%B9%98](https://velog.io/@mir21c/App.2-Eigen-%EC%84%A4%EC%B9%98)