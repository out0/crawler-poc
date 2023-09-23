# require CMake 2.8 or greater
cmake_minimum_required(VERSION 3.9)

# declare project name
set(CMAKE_CUDA_COMPILER "/usr/local/cuda-12.2/bin/nvcc")
project(bev-module LANGUAGES CXX CUDA)

set(CMAKE_BUILD_TYPE "Debug")

# import jetson-inference and jetson-utils packages.
# note that if you didn't do "sudo make install"
# while building jetson-inference, this will error.
find_package(OpenCV REQUIRED)
find_package(Threads REQUIRED) 
find_package(PkgConfig REQUIRED)

# CUDA is required
find_package(CUDA REQUIRED)

message(STATUS "OpenCV_INCLUDE_DIRS = ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV_LIBS = ${OpenCV_LIBS}")


set(CMAKE_CXX_STANDARD 17)
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR})
include_directories(/usr/include/gstreamer-1.0/usr/include/glib-2.0 /usr/include/libxml2)
include_directories(/usr/include/driveless)


SET(TARGET_H
    matmul.h
)

cuda_add_executable(
    bev
    bev_video.cpp 
    matmul.cu
)

# link my-recognition to jetson-inference library
target_link_libraries(bev
    ${OpenCV_LIBS}
    ${CMAKE_THREAD_LIBS_INIT}
    ${CUDA_LIBRARIES}
)