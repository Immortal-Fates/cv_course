
CMAKE_MINIMUM_REQUIRED(VERSION 3.0)

project(HW2)

find_package(OpenCV REQUIRED)
add_executable(ft.exe src/ft.cpp)
target_link_libraries(ft.exe ${OpenCV_LIBS})