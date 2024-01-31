cmake_minimum_required(VERSION 3.0)
project(OCR_OPENCCV_CPP)

find_package(OpenCV 4 REQUIRED)
find_package(Tesseract REQUIRED)

add_executable(main basic_ocr.cpp)

target_include_directories(main PRIVATE ${OpenCV_INCLUDE_DIRS} ${TESSERACT_INCLUDE_DIRS})
target_link_libraries(main PRIVATE ${OpenCV_LIBS} ${TESSERACT_LIBRARIES} archive)
