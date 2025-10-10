#include "cuda_convolution.hpp"
#include "cuda_convolution_utils.hpp"

template <typename T, size_t INPUT_BLOCK_X, size_t INPUT_BLOCK_Y>
__global__ void convolution_v02(size_t m, size_t n, size_t r, T const* A,
                                size_t lda, T* B, size_t ldb, T const* W,
                                size_t ldw)
{
    const size_t k = 2U * r + 1U;
    const long long r_ll = static_cast<long long>(r);
    const size_t OUTPUT_BLOCK_X{INPUT_BLOCK_X - 2U * r};
    const size_t OUTPUT_BLOCK_Y{INPUT_BLOCK_Y - 2U * r};
    const ssize_t tile_row{threadIdx.y - r_ll};
    const ssize_t tile_col{threadIdx.x - r_ll};
    const ssize_t row{static_cast<ssize_t>(blockIdx.y * OUTPUT_BLOCK_Y) +
                      tile_row};
    const ssize_t col{static_cast<ssize_t>(blockIdx.x * OUTPUT_BLOCK_X) +
                      tile_col};

    // Access constant memory
    T const* W_const = reinterpret_cast<T const*>(const_mem);

    // Copy tile to shared memory
    __shared__ T A_tile[INPUT_BLOCK_Y][INPUT_BLOCK_X];
    if (row >= 0 && row < m && col >= 0 && col < n)
    {
        A_tile[threadIdx.y][threadIdx.x] = A[row * lda + col];
    }
    else
    {
        A_tile[threadIdx.y][threadIdx.x] = static_cast<T>(0);
    }
    __syncthreads();

    // Compute the cell
    if (row >= 0 && row < m && col >= 0 && col < n)
    {
        if (tile_row >= 0 && tile_row < OUTPUT_BLOCK_Y && tile_col >= 0 &&
            tile_col < OUTPUT_BLOCK_X)
        {
            T sum{static_cast<T>(0)};
            for (size_t w_row{0U}; w_row < k; ++w_row)
            {
                for (size_t w_col{0U}; w_col < k; ++w_col)
                {
                    sum += A_tile[tile_row + w_row][tile_col + w_col] *
                           W_const[w_row * ldw + w_col];
                }
            }
            B[row * ldb + col] = sum;
        }
    }
}

template <typename T>
void launch_kernel_convolution_v02(size_t m, size_t n, size_t r, T const* A,
                                   size_t lda, T* B, size_t ldb, T const* W,
                                   size_t ldw, cudaStream_t stream)
{
    const size_t INPUT_BLOCK_X{32U};
    const size_t INPUT_BLOCK_Y{32U};
    const size_t OUTPUT_BLOCK_X{INPUT_BLOCK_X - 2U * r};
    const size_t OUTPUT_BLOCK_Y{INPUT_BLOCK_Y - 2U * r};

    const dim3 block_dim{INPUT_BLOCK_X, INPUT_BLOCK_Y, 1U};
    const dim3 grid_dim{
        static_cast<unsigned int>((m + OUTPUT_BLOCK_X - 1U) / OUTPUT_BLOCK_X),
        static_cast<unsigned int>((n + OUTPUT_BLOCK_Y - 1U) / OUTPUT_BLOCK_Y),
        1U};
    convolution_v02<T, INPUT_BLOCK_X, INPUT_BLOCK_Y>
        <<<grid_dim, block_dim, 0U, stream>>>(m, n, r, A, lda, B, ldb, W, ldw);
    CHECK_LAST_CUDA_ERROR();
}

template void launch_kernel_convolution_v02<float>(size_t m, size_t n, size_t r,
                                                   float const* A, size_t lda,
                                                   float* B, size_t ldb,
                                                   float const* W, size_t ldw,
                                                   cudaStream_t stream);