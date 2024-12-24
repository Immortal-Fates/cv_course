CMAKE_MINIMUM_REQUIRED(VERSION 3.0)

project(HW5)

file(GLOB SOURCES "src/*.cpp")

include_directories(inc)


find_package(OpenCV REQUIRED)
add_executable(hw5 ${SOURCES})
target_link_libraries(hw5 ${OpenCV_LIBS})
