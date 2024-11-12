CMAKE_MINIMUM_REQUIRED(VERSION 3.0)

project(HW1)

find_package(OpenCV REQUIRED)
add_executable(hw1.exe hw1.cpp)
target_link_libraries(hw1.exe ${OpenCV_LIBS})
