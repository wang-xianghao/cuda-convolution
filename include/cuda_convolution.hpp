#ifndef CUDA_GEMM_HPP
#define CUDA_GEMM_HPP

#include <cuda_runtime.h>

constexpr size_t MAX_CONST_MEM = 64 * 1024;
extern __constant__ std::byte const_mem[];

template <typename T>
void launch_kernel_convolution_v00(size_t m, size_t n, size_t r, T const* A,
                                   size_t lda, T* B, size_t ldb, T const* W,
                                   size_t ldw, cudaStream_t stream);

template <typename T>
void launch_kernel_convolution_v01(size_t m, size_t n, size_t r, T const* A,
                                   size_t lda, T* B, size_t ldb, T const* W,
                                   size_t ldw, cudaStream_t stream);

template <typename T>
void launch_kernel_convolution_v02(size_t m, size_t n, size_t r, T const* A,
                                   size_t lda, T* B, size_t ldb, T const* W,
                                   size_t ldw, cudaStream_t stream);

template <typename T>
void launch_kernel_convolution_v03(size_t m, size_t n, size_t r, T const* A,
                                   size_t lda, T* B, size_t ldb, T const* W,
                                   size_t ldw, cudaStream_t stream);

#endif