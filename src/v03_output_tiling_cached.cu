#include "cuda_convolution.hpp"
#include "cuda_convolution_utils.hpp"

template <typename T, size_t OUTPUT_BLOCK_X, size_t OUTPUT_BLOCK_Y>
__global__ void convolution_v03(size_t m, size_t n, size_t r, T const* A,
                                size_t lda, T* B, size_t ldb, T const* W,
                                size_t ldw)
{
    const ssize_t r_signed{static_cast<ssize_t>(r)};
    const size_t k{2U * r + 1U};
    const size_t row{blockIdx.y * OUTPUT_BLOCK_Y + threadIdx.y};
    const size_t col{blockIdx.x * OUTPUT_BLOCK_X + threadIdx.x};
    const size_t thread_row{threadIdx.y};
    const size_t thread_col{threadIdx.x};

    // Access constant memory
    T const* W_const = reinterpret_cast<T const*>(const_mem);

    // Copy tile to shared memory
    __shared__ T A_tile[OUTPUT_BLOCK_Y][OUTPUT_BLOCK_X];
    if (row < m && col < n)
    {
        A_tile[thread_row][thread_col] = A[row * lda + col];
    }
    else
    {
        A_tile[thread_row][thread_col] = static_cast<T>(0);
    }
    __syncthreads();

    // Compute cell
    if (row < m && col < n)
    {
        T sum{static_cast<T>(0)};
        for (size_t w_row{0U}; w_row < k; ++w_row)
        {
            for (size_t w_col{0U}; w_col < k; ++w_col)
            {
                const ssize_t input_row{
                    static_cast<ssize_t>(row - r_signed + w_row)};
                const ssize_t input_col{
                    static_cast<ssize_t>(col - r_signed + w_col)};
                const ssize_t input_thread_row{
                    static_cast<ssize_t>(thread_row - r_signed + w_row)};
                const ssize_t input_thread_col{
                    static_cast<ssize_t>(thread_col - r_signed + w_col)};
                if (input_thread_row >= 0 &&
                    input_thread_row < OUTPUT_BLOCK_Y &&
                    input_thread_col >= 0 && input_thread_col < OUTPUT_BLOCK_X)
                {
                    // Use shared memory
                    sum += A_tile[input_thread_row][input_thread_col] *
                           W_const[w_row * ldw + w_col];
                }
                else
                {
                    // Use L2 cache
                    if (input_row >= 0 && input_row < m && input_col >= 0 &&
                        input_col < n)
                    {
                        sum += A[input_row * lda + input_col] *
                               W_const[w_row * ldw + w_col];
                    }
                }
            }
        }
        B[row * ldb + col] = sum;
    }
}

template <typename T>
void launch_kernel_convolution_v03(size_t m, size_t n, size_t r, T const* A,
                                   size_t lda, T* B, size_t ldb, T const* W,
                                   size_t ldw, cudaStream_t stream)
{
    const size_t OUTPUT_BLOCK_X{32U};
    const size_t OUTPUT_BLOCK_Y{32U};
    const dim3 block_dim{OUTPUT_BLOCK_X, OUTPUT_BLOCK_Y, 1U};
    const dim3 grid_dim{
        (static_cast<unsigned int>(m) + block_dim.x - 1U) / block_dim.x,
        (static_cast<unsigned int>(n) + block_dim.y - 1U) / block_dim.y, 1U};
    convolution_v03<T, OUTPUT_BLOCK_X, OUTPUT_BLOCK_Y>
        <<<grid_dim, block_dim, 0U, stream>>>(m, n, r, A, lda, B, ldb, W, ldw);
    CHECK_LAST_CUDA_ERROR();
}

template void launch_kernel_convolution_v03<float>(size_t m, size_t n, size_t r,
                                                   float const* A, size_t lda,
                                                   float* B, size_t ldb,
                                                   float const* W, size_t ldw,
                                                   cudaStream_t stream);