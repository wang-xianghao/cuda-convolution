#ifndef CUDA_GEMM_HPP
#define CUDA_GEMM_HPP

#include <cuda_runtime.h>

template <typename T>
void launch_kernel_convolution_v00(size_t m, size_t n, size_t r, T const* A,
                                   size_t lda, T* B, size_t ldb, T const* W,
                                   size_t ldw, cudaStream_t stream);

#endif