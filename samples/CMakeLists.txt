cmake_minimum_required(VERSION 2.8)
project( main )
find_package( OpenCV REQUIRED )
FIND_PACKAGE(OpenMP REQUIRED)

set(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
add_executable( main cascade_detect_sequence.cpp)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

include_directories(${CMAKE_SOURCE_DIR}/../include)

MESSAGE(STATUS "Include dirs ${OpenCV_INCLUDE_DIRS}")
MESSAGE(STATUS "LINK LIBRARIES ${OpenCV_LIBS}")

target_link_libraries(main ${OpenCV_LIBS} )
target_link_libraries(main ${CMAKE_SOURCE_DIR}/libdpm.a)


