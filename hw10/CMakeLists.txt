
CMAKE_MINIMUM_REQUIRED(VERSION 3.0)

project(HW4)

find_package(OpenCV REQUIRED)
add_executable(blur_restoration.exe src/blur_restoration.cpp)
target_link_libraries(blur_restoration.exe ${OpenCV_LIBS})

add_executable(restoration.exe src/restoration.cpp)
target_link_libraries(restoration.exe ${OpenCV_LIBS})
