cmake_minimum_required(VERSION 3.25)
project(CUDAKERNELS LANGUAGES CXX CUDA) 
## actually, set cuda here, you dont need to include FindCUDA.cmake

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR})
# set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_INCLUDES 0)
# set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_LIBRARIES 0)
# set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_OBJECTS 0)

##  Dependencies include(cuda, cuBLAS, cuDNN) ##
# include(${ROOT_DIR}/cmake/FindCUDA.cmake)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_86,code=sm_86")

set(Torch_DIR /home/aklice/miniconda3/envs/d2l/lib/python3.9/site-packages/torch/include/)
set(Torch_DIR ${Torch_DIR} /home/aklice/miniconda3/envs/d2l/lib/python3.9/site-packages/torch/include/torch/csrc/api/include)
include_directories(${Torch_DIR})
# set(PYTHON_DIR /home/aklice/miniconda3/envs/d2l/include/python3.9/)
# include_directories(${PYTHON_DIR})

# set(CMAKE_BUILD_TYPE "Debug")
# SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g -ggdb")

# include(${ROOT_DIR}/cmake/*.cmake)
include_directories(${ROOT_DIR}/include)
# cutlass
include_directories(${ROOT_DIR}/3rd/cutlass/include)

# add_executable(elementwise elementwise/elementwise.cu)
# target_link_libraries(elementwise PRIVATE cudart cublas)


# add_executable(softmax softmax/softmax.cu)
# target_link_libraries(softmax PRIVATE cudart cublas)


# add_executable(layernorm layernorm/layernorm.cu)
# target_link_libraries(layernorm PRIVATE cudart cublas)

add_executable(test_mma cutlass/code/test_mma.cu)
target_link_libraries(test_mma PRIVATE cudart cublas)


add_executable(test_gemm gemm/gemm.cu)
target_link_libraries(test_gemm PRIVATE cudart cublas)
add_executable(test_cublas gemm/test_cublas.cu)
target_link_libraries(test_cublas PRIVATE cudart cublas)
add_executable(wmma_gemm gemm/gemm_wmma.cu)
target_link_libraries(wmma_gemm PRIVATE cudart cublas)


add_executable(test_transpose transpose/transpose.cu)
target_link_libraries(test_transpose PRIVATE cudart cublas)


add_executable(test_top_k top-k/top_k.cu)
target_link_libraries(test_top_k PRIVATE cudart cublas)





