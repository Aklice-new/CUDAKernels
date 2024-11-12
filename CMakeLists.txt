cmake_minimum_required(VERSION 3.25)
project(CUDAKERNELS LANGUAGES CXX CUDA) 
## actually, set cuda here, you dont need to include FindCUDA.cmake

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR})

##  Dependencies include(cuda, cuBLAS, cuDNN) ##
include(${ROOT_DIR}/cmake/FindCUDA.cmake)

# include(${ROOT_DIR}/cmake/*.cmake)
set(HEADER_FILES ${ROOT_DIR}/include/common.h)
include_directories(${ROOT_DIR}/include)

add_executable(elementwise elementwise/elementwise.cu ${HEADER_FILES})
target_link_libraries(elementwise PRIVATE CUDA::cudart CUDA::cublas)


add_executable(softmax softmax/softmax.cu ${HEADER_FILES})
target_link_libraries(softmax PRIVATE CUDA::cudart CUDA::cublas)
