CMAKE_MINIMUM_REQUIRED(VERSION 3.0)

project(HW1)

find_package(OpenCV REQUIRED)
add_executable(problem1_1.exe src/problem1_1.cpp)
target_link_libraries(problem1_1.exe ${OpenCV_LIBS})

add_executable(problem1_2.exe src/problem1_2.cpp)
target_link_libraries(problem1_2.exe ${OpenCV_LIBS})

add_executable(problem1_3.exe src/problem1_3.cpp)
target_link_libraries(problem1_3.exe ${OpenCV_LIBS})

add_executable(problem2_1.exe src/problem2_1.cpp)
target_link_libraries(problem2_1.exe ${OpenCV_LIBS})

add_executable(problem2_2.exe src/problem2_2.cpp)
target_link_libraries(problem2_2.exe ${OpenCV_LIBS})

add_executable(problem2_3.exe src/problem2_3.cpp)
target_link_libraries(problem2_3.exe ${OpenCV_LIBS})
