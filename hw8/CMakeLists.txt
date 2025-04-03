
CMAKE_MINIMUM_REQUIRED(VERSION 3.0)

project(HW3)

find_package(OpenCV REQUIRED)
add_executable(canny.exe src/canny_detect.cpp)
target_link_libraries(canny.exe ${OpenCV_LIBS})

add_executable(hough.exe src/hough_detect.cpp)
target_link_libraries(hough.exe ${OpenCV_LIBS})