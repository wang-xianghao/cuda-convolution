cmake_minimum_required(VERSION 3.28)

project(CUDA-CONVOLUTION VERSION 0.0.1 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find CUDA Toolkit
find_package(CUDAToolkit REQUIRED)

find_path(CUDA_CONVOLUTION_INCLUDE_DIRS cuda_convolution_utils.hpp HINTS ${CMAKE_SOURCE_DIR}/include)
# file(GLOB CUDA_CONVOLUTION_HEADERS ${CMAKE_SOURCE_DIR}/include/*.hpp ${CMAKE_SOURCE_DIR}/include/*.cuh)

# Add all the source files in the current directory to build the library
add_library(
    cuda_convolution
    SHARED
    cuda_convolution_utils.cu
    v00_naive.cu
)

# Add the include directory of the library to the include directories of the project
target_include_directories(cuda_convolution PUBLIC ${CUDA_CONVOLUTION_INCLUDE_DIRS})

# Set the CUDA architecture to compile the code for
# https://cmake.org/cmake/help/latest/prop_tgt/CUDA_ARCHITECTURES.html
set_target_properties(cuda_convolution PROPERTIES CUDA_ARCHITECTURES native)

# Profile FP32 main program
add_executable(profile_cuda_convolution_fp32 profile_cuda_convolution_fp32.cu)
target_link_libraries(profile_cuda_convolution_fp32 cuda_convolution CUDA::cublas)
set_target_properties(profile_cuda_convolution_fp32 PROPERTIES CUDA_ARCHITECTURES native)
