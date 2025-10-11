#include "cuda_convolution.hpp"
#include "cuda_convolution_utils.hpp"

template <typename T>
__global__ void convolution_v03(size_t m, size_t n, size_t r, T const* A,
                                size_t lda, T* B, size_t ldb, T const* W,
                                size_t ldw)
{
    const size_t B_col_idx{blockIdx.x * blockDim.x + threadIdx.x};
    const size_t B_row_idx{blockIdx.y * blockDim.y + threadIdx.y};

    // Access constant memory
    T const* W_const = reinterpret_cast<T const*>(const_mem);

    if (B_row_idx >= m || B_col_idx >= n)
    {
        return;
    }

    T sum{static_cast<T>(0)};

    for (size_t w_row_idx{0}; w_row_idx < 2U * r + 1U; w_row_idx++)
    {
        for (size_t w_col_idx{0}; w_col_idx < 2U * r + 1U; w_col_idx++)
        {
            ssize_t A_row_idx{static_cast<ssize_t>(B_row_idx - r + w_row_idx)};
            ssize_t A_col_idx{static_cast<ssize_t>(B_col_idx - r + w_col_idx)};
            if (A_row_idx >= 0 && A_row_idx < m && A_col_idx >= 0 &&
                A_col_idx < n)
            {
                sum += A[A_row_idx * lda + A_col_idx] *
                       W_const[w_row_idx * ldw + w_col_idx];
            }
        }
    }

    B[B_row_idx * ldb + B_col_idx] = sum;
}

template <typename T>
void launch_kernel_convolution_v03(size_t m, size_t n, size_t r, T const* A,
                                   size_t lda, T* B, size_t ldb, T const* W,
                                   size_t ldw, cudaStream_t stream)
{
    const dim3 block_dim{32U, 32U, 1U};
    const dim3 grid_dim{
        (static_cast<unsigned int>(m) + block_dim.x - 1U) / block_dim.x,
        (static_cast<unsigned int>(n) + block_dim.y - 1U) / block_dim.y, 1U};
    convolution_v03<T>
        <<<grid_dim, block_dim, 0U, stream>>>(m, n, r, A, lda, B, ldb, W, ldw);
    CHECK_LAST_CUDA_ERROR();
}

template void launch_kernel_convolution_v03<float>(size_t m, size_t n, size_t r,
                                                   float const* A, size_t lda,
                                                   float* B, size_t ldb,
                                                   float const* W, size_t ldw,
                                                   cudaStream_t stream);