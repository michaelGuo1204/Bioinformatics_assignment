project(Cuda)
cmake_minimum_required(VERSION 2.8)
find_package(CUDA REQUIRED)

file(GLOB_RECURSE CURRENT_HEADERS *.cuh)
file(GLOB CURRENT_SOURCES *.cu)

list(APPEND CUDA_NVCC_FLAGS "-std=c++11")
include_directories("/usr/local/cuda/samples/common/inc/")
source_group("Include" FILES ${CURRENT_HEADERS})
source_group("Source" FILES ${CURRENT_SOURCES})

cuda_add_library(Cuda SHARED ${CURRENT_HEADERS} ${CURRENT_SOURCES})

#not used in this demo
TARGET_LINK_LIBRARIES(Cuda pthread)
TARGET_LINK_LIBRARIES(Cuda cufft)
TARGET_LINK_LIBRARIES(Cuda pcap)