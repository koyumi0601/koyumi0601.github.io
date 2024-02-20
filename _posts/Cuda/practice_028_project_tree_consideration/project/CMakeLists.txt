cmake_minimum_required(VERSION 3.10)
project(MyProject)


# set(CMAKE_AUTOUIC ON) # UI 파일을 자동으로 컴파일
# set(CMAKE_AUTOMOC ON) # MOC(Qt의 특수한 프리컴파일러, Meta Object Systm)을 자동으로 실행
# set(CMAKE_AUTORCC ON) # 리소스 파일을 자동으로 처리 (이미지, 아이콘, 문자열, .qrc)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# Qt가 설치된 경로를 지정
#set(CMAKE_PREFIX_PATH "/home/ko/Qt5.12.12")
#CMAKE_PREFIX_PATH=/home/ko/Qt5.12.12/5.12.12/gcc_64/lib/cmake/Qt5:$CMAKE_PREFIX_PATH


if (UNIX AND NOT APPLE)
    find_package(PkgConfig REQUIRED)
    pkg_search_module(TESSERACT REQUIRED tesseract)
    find_library(SQLITE3_LIBRARY sqlite3)
elseif(WIN32)
    message(STATUS "WINDOW!")
    # 시스템이 윈도우일 경우 OpenCV와 Tesseract는 각각의 CMake 모듈을 사용하여 찾음
    find_package(OpenCV REQUIRED COMPONENTS core highgui imgproc)

    # tesseract
    # CMAKE_PREFIX_PATH, Tesseract_DIR  

        # curl
    LIST(APPEND CMAKE_PREFIX_PATH "D:/Github_Blog/vcpkg/packages/curl_x64-windows")
    find_package(CURL REQUIRED)


    # find_package(Tesseract REQUIRED)
    find_package(Tesseract REQUIRED PATHS "D:/Github_Blog/vcpkg/installed/tesseract_x64-windows")
    set(CMAKE_PREFIX_PATH "D:/Github_Blog/vcpkg/installed/tesseract_x64-windows")
    
    # include_directories("D:/Github_Blog/vcpkg/installed/x64-windows/include")
    # link_directories("D:/Github_Blog/vcpkg/installed/x64-windows/lib")

    # set(ZLIB_INCLUDE_DIR "D:/Github_Blog/vcpkg/installed/x64-windows/include")
    # set(ZLIB_LIBRARY "D:/Github_Blog/vcpkg/installed/x64-windows/lib/zlib.lib")

    # include_directories(${ZLIB_INCLUDE_DIR})
    # target_link_libraries(MyProject PRIVATE ${ZLIB_LIBRARY})
    set(ZLIB_INCLUDE_DIR "D:/Github_Blog/vcpkg/installed/x64-windows/include")
    set(ZLIB_LIBRARY "D:/Github_Blog/vcpkg/installed/x64-windows/lib/zlib.lib")
    find_package(ZLIB REQUIRED)
    include_directories(${ZLIB_INCLUDE_DIRS})
    target_link_libraries(MyProject PRIVATE ${ZLIB_LIBRARIES})
    

    # SQLite3 라이브러리는 시스템에 직접 설치되어 있다고 가정


    set(SQLITE3_LIBRARY sqlite3)

    add_compile_definitions(USE_FILESYSTEM) # experimental/filesystem -> filesystem

endif()

#find_package(Qt5 COMPONENTS Widgets REQUIRED)
find_package(OpenCV REQUIRED)


file(GLOB_RECURSE SOURCES "src/*.cpp" "src/*.ui")
set(CMAKE_BINARY_DIR ${CMAKE_SOURCE_DIR}/build)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR})


include_directories(${TESSERACT_INCLUDE_DIRS})

include_directories(include) # target_link_libraries(include) 하면 안됨.
#add_executable(MyProject ${SOURCES})

if(USE_FILESYSTEM)
    add_compile_definitions(USE_FILESYSTEM)
endif()



add_executable(MyProject 
src/main.cpp
src/database/database_manager.cpp
src/imaging_system/image_filter.cpp
src/imaging_system/image_loader.cpp
src/imaging_system/image_transform.cpp  
src/imaging_system/ocr.cpp
src/networking/network_manager.cpp
src/userinterface/mainwindow.cpp
src/logger/logger.cpp)


if(USE_FILESYSTEM)
    target_compile_definitions(MyProject PRIVATE USE_FILESYSTEM)
    target_link_libraries(MyProject PRIVATE stdc++fs)
endif()



#target_link_libraries(MyProject PRIVATE include Qt${QT_VERSION_MAJOR}::Widgets ${OpenCV_LIBS} tesseract)
#target_link_libraries(MyProject PRIVATE tesseract)


# 시스템이 우분투일 경우
if(UNIX AND NOT APPLE)
    target_link_libraries(MyProject PRIVATE ${OpenCV_LIBS} tesseract stdc++fs pthread ${SQLITE3_LIBRARY})
# 시스템이 윈도우일 경우
elseif(WIN32)
    target_link_libraries(MyProject PRIVATE ${OpenCV_LIBS} ${TESSERACT_LIBRARIES} stdc++fs pthread ${SQLITE3_LIBRARY} CURL::libcurl)
    #target_link_libraries(MyProject PRIVATE ${OpenCV_LIBS} stdc++fs pthread ${SQLITE3_LIBRARY})

endif()