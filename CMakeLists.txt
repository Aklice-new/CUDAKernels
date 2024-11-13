cmake_minimum_required(VERSION 3.25)
project(CUDAKERNELS LANGUAGES CXX CUDA) 
## actually, set cuda here, you dont need to include FindCUDA.cmake

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_INCLUDES 0)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_LIBRARIES 0)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_OBJECTS 0)

##  Dependencies include(cuda, cuBLAS, cuDNN) ##
# include(${ROOT_DIR}/cmake/FindCUDA.cmake)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_86,code=sm_86")

# include(${ROOT_DIR}/cmake/*.cmake)
include_directories(${ROOT_DIR}/include)

add_executable(elementwise elementwise/elementwise.cu)
target_link_libraries(elementwise PRIVATE cudart cublas)


add_executable(softmax softmax/softmax.cu)
target_link_libraries(softmax PRIVATE cudart cublas)


add_executable(layernorm layernorm/layernorm.cu)
target_link_libraries(layernorm PRIVATE cudart cublas)
