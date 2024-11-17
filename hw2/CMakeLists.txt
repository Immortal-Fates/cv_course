
CMAKE_MINIMUM_REQUIRED(VERSION 3.0)

project(HW2)

file(GLOB SOURCES "src/*.cpp")

find_package(OpenCV REQUIRED)
add_executable(hw2 ${SOURCES})
target_link_libraries(hw2 ${OpenCV_LIBS})
