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

find_package(PkgConfig REQUIRED)
pkg_search_module(TESSERACT REQUIRED tesseract)
#find_package(Qt5 COMPONENTS Widgets REQUIRED)
find_package(OpenCV REQUIRED)

find_library(SQLITE3_LIBRARY sqlite3)

file(GLOB_RECURSE SOURCES "src/*.cpp" "src/*.ui")
set(CMAKE_BINARY_DIR ${CMAKE_SOURCE_DIR}/build)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR})


include_directories(${TESSERACT_INCLUDE_DIRS})
include_directories(include) # target_link_libraries(include) 하면 안됨.
#add_executable(MyProject ${SOURCES})
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


#target_link_libraries(MyProject PRIVATE include Qt${QT_VERSION_MAJOR}::Widgets ${OpenCV_LIBS} tesseract)
#target_link_libraries(MyProject PRIVATE tesseract)

target_link_libraries(MyProject PRIVATE ${OpenCV_LIBS} tesseract stdc++fs pthread ${SQLITE3_LIBRARY})
