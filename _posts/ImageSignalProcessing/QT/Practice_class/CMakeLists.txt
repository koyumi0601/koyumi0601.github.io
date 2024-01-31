cmake_minimum_required(VERSION 3.1)
project(ImageLoader)

add_executable(ImageLoader main.cpp)

find_package(Qt5 COMPONENTS Core Gui Widgets OpenGL REQUIRED)
find_package(OpenGL REQUIRED)

target_include_directories(ImageLoader PRIVATE ${OPENGL_INCLUDE_DIRS})
target_link_libraries(ImageLoader Qt5::Core Qt5::Gui Qt5::Widgets Qt5::OpenGL ${OPENGL_LIBRARIES})