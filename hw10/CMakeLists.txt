
CMAKE_MINIMUM_REQUIRED(VERSION 3.0)

project(HW5)

find_package(OpenCV REQUIRED)
add_executable(morphology.exe src/morphology.cpp)
target_link_libraries(morphology.exe ${OpenCV_LIBS})

