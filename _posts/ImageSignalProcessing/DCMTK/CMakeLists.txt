cmake_minimum_required(VERSION 3.0)
project(my_dcmtk_project)

set(DCMTK_ROOT_DIR "/usr/local")
find_package(DCMTK REQUIRED)

add_executable(my_dcmtk_app main.cpp)
target_link_libraries(my_dcmtk_app PRIVATE ${DCMTK_LIBRARIES})