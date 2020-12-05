project(cpart)
cmake_minimum_required(VERSION 2.8)

file(GLOB_RECURSE CURRENT_HEADERS *.h)
file(GLOB CURRENT_SOURCES *.c)
set(CMAKE_C_COMPILER mpicc)
#list(APPEND CMAKE_C_FLAGS "-lstdc")

source_group("Include" FILES ${CURRENT_HEADERS})
source_group("Source" FILES ${CURRENT_SOURCES})

add_library(cpart SHARED ${CURRENT_HEADERS} ${CURRENT_SOURCES})
#target_link_libraries(cpart Cuda)