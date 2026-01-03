cmake_minimum_required(VERSION 3.10)
project(bin)

# Find OpenCV package
find_package(OpenCV REQUIRED)

# Include OpenCV directories
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(bin main.cpp)

target_link_libraries(bin ${OpenCV_LIBS})
