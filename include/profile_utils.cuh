#ifndef PROFILE_UTILS_CUH
#define PROFILE_UTILS_CUH

#include <assert.h>
#include <cuda_runtime.h>
#include <iostream>
#include <random>

#include "cuda_convolution.hpp"
#include "cuda_convolution_utils.hpp"

template <typename T>
using ConvolutionLauncher =
    std::function<void(size_t m, size_t n, size_t r, const T* A, size_t lda,
                       T* B, size_t ldb, const T* W, size_t ldw,
                       cudaStream_t stream)>;

void print_device_info()
{
    int device_id{0}, memoryClockRate;
    cudaGetDevice(&device_id);
    cudaDeviceProp device_prop;
    cudaGetDeviceProperties(&device_prop, device_id);
    cudaDeviceGetAttribute(&memoryClockRate, cudaDevAttrMemoryClockRate,
                           device_id);

    std::cout << "Device Name: " << device_prop.name << std::endl;
    float const memory_size{static_cast<float>(device_prop.totalGlobalMem) /
                            (1 << 30)};
    std::cout << "Memory Size: " << memory_size << " GB" << std::endl;
    float const peak_bandwidth{static_cast<float>(
        2.0f * memoryClockRate * (device_prop.memoryBusWidth / 8) / 1.0e6)};
    std::cout << "Peak Bandwitdh: " << peak_bandwidth << " GB/s" << std::endl;
    std::cout << "Bus Width: " << device_prop.memoryBusWidth << " Bit"
              << std::endl;
    std::cout << std::endl;
}

template <typename T>
float compute_effective_bandwidth(size_t m, size_t n, size_t r, float latency)
{
    size_t k = 2U * r + 1U;
    return ((2 * m * n + k * k) * sizeof(T)) / (latency * 1e-3) / 1e9;
}

float compute_effective_tflops(size_t m, size_t n, size_t r, float latency)
{
    size_t k = 2U * r + 1U;
    return (2 * m * n * k * k) / (latency * 1e-3) / 1e12;
}

void print_performance_result(size_t m, size_t n, size_t r, float latency)
{
    float const effective_bandwidth{
        compute_effective_bandwidth<float>(m, n, r, latency)};
    float const effective_tflops{compute_effective_tflops(m, n, r, latency)};

    std::cout << "Latency: " << latency << " ms" << std::endl;
    std::cout << "Effective Bandwidth: " << effective_bandwidth << " GB/s"
              << std::endl;
    std::cout << "Effective TFLOPS: " << effective_tflops << " TFLOPS"
              << std::endl;
}

template <typename T>
void random_initialize_matrix(T* A, size_t m, size_t n, size_t lda,
                              unsigned int seed = 0U)
{
    std::default_random_engine eng(seed);
    // The best way to verify is to use integer values.
    std::uniform_int_distribution<int> dis(0, 5);
    // std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    auto const rand = [&dis, &eng]() { return dis(eng); };
    for (size_t i{0U}; i < m; ++i)
    {
        for (size_t j{0U}; j < n; ++j)
        {
            A[i * lda + j] = static_cast<T>(rand());
        }
    }
}

template <typename T>
void launch_convolution_cpu(size_t m, size_t n, size_t r, T const* A,
                            size_t lda, T* B, size_t ldb, T const* W,
                            size_t ldw)
{
    size_t k = 2U * r + 1U;
    for (size_t i{0U}; i < m; ++i)
    {
        for (size_t j{0U}; j < n; ++j)
        {
            T sum{static_cast<T>(0)};
            for (size_t kernel_i{0U}; kernel_i < k; ++kernel_i)
            {
                for (size_t kernel_j{0U}; kernel_j < k; ++kernel_j)
                {
                    ssize_t ii{static_cast<ssize_t>(i - r + kernel_i)};
                    ssize_t jj{static_cast<ssize_t>(j - r + kernel_j)};
                    if (ii >= 0 && ii < m && jj >= 0 && jj < n)
                    {
                        sum += A[ii * lda + jj] * W[kernel_i * ldw + kernel_j];
                    }
                }
            }
            B[i * ldb + j] = sum;
        }
    }
}

template <typename T>
bool all_close(T const* B, T const* B_ref, size_t m, size_t n, size_t ldb,
               T abs_tol, double rel_tol)
{
    bool status{true};
    for (size_t i{0U}; i < m; ++i)
    {
        for (size_t j{0U}; j < n; ++j)
        {
            double const C_val{static_cast<double>(B[i * ldb + j])};
            double const C_ref_val{static_cast<double>(B_ref[i * ldb + j])};
            double const diff{C_val - C_ref_val};
            double const diff_val{std::abs(diff)};
            if (diff_val >
                std::max(static_cast<double>(abs_tol),
                         static_cast<double>(std::abs(C_ref_val)) * rel_tol))
            {
                std::cout << "B[" << i << ", " << j << "] = " << C_val
                          << " B_ref[" << i << ", " << j << "] = " << C_ref_val
                          << " Abs Diff: " << diff_val
                          << " Abs Diff Threshold: "
                          << static_cast<double>(abs_tol)
                          << " Rel->Abs Diff Threshold: "
                          << static_cast<double>(
                                 static_cast<double>(std::abs(C_ref_val)) *
                                 rel_tol)
                          << std::endl;
                status = false;
                return status;
            }
        }
    }
    return status;
}

template <typename T>
float measure_performance(std::function<T(cudaStream_t)> bound_function,
                          cudaStream_t stream, size_t num_repeats = 100,
                          size_t num_warmups = 100)
{
    cudaEvent_t start, stop;
    float time;

    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));

    // Warmup
    for (size_t i{0U}; i < num_warmups; ++i)
    {
        bound_function(stream);
    }
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));

    // Meausre
    CHECK_CUDA_ERROR(cudaEventRecord(start, stream));
    for (size_t i{0U}; i < num_repeats; ++i)
    {
        bound_function(stream);
    }
    CHECK_CUDA_ERROR(cudaEventRecord(stop, stream));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    CHECK_LAST_CUDA_ERROR();
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&time, start, stop));
    CHECK_CUDA_ERROR(cudaEventDestroy(start));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop));

    const float latency{time / num_repeats};

    return latency;
}

template <typename T>
float profile_convolution(size_t m, size_t n, size_t r, size_t lda, size_t ldb,
                          size_t ldw,
                          ConvolutionLauncher<T> convolution_launcher,
                          T abs_tol, double rel_tol, size_t num_repeats = 10,
                          size_t num_warmups = 10, unsigned int seed = 0U)
{
    // Create CUDA stream
    cudaStream_t stream;
    CHECK_CUDA_ERROR(cudaStreamCreate(&stream));

    // Calculate convolution kernel width
    size_t k = 2U * r + 1U;

    // Allocate memory on host
    T* A_host{nullptr};
    T* B_host{nullptr};
    T* B_host_ref{nullptr};
    T* W_host{nullptr};
    CHECK_CUDA_ERROR(cudaMallocHost(&A_host, m * lda * sizeof(T)));
    CHECK_CUDA_ERROR(cudaMallocHost(&B_host, m * ldb * sizeof(T)));
    CHECK_CUDA_ERROR(cudaMallocHost(&B_host_ref, m * ldb * sizeof(T)));
    CHECK_CUDA_ERROR(cudaMallocHost(&W_host, k * ldw * sizeof(T)));

    // Initialize matrix A, B and W
    random_initialize_matrix(A_host, m, n, lda);
    random_initialize_matrix(W_host, k, k, ldw);

    // Allocate memory on device
    T* A_device{nullptr};
    T* B_device{nullptr};
    T* W_device{nullptr};
    CHECK_CUDA_ERROR(cudaMalloc(&A_device, m * lda * sizeof(T)));
    CHECK_CUDA_ERROR(cudaMalloc(&B_device, m * ldb * sizeof(T)));
    CHECK_CUDA_ERROR(cudaMalloc(&W_device, k * ldw * sizeof(T)));

    // Copy matrices from host to device
    CHECK_CUDA_ERROR(cudaMemcpy(A_device, A_host, m * lda * sizeof(T),
                                cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(W_device, W_host, k * ldw * sizeof(T),
                                cudaMemcpyHostToDevice));

    // Copy filter to constant memory
    CHECK_CUDA_ERROR(
        cudaMemcpyToSymbol(const_mem, W_host, k * ldw * sizeof(T)));

    // Compute reference output using CPU
    std::cout << "Computing reference output using CPU..." << std::endl;
    launch_convolution_cpu<T>(m, n, r, A_host, lda, B_host_ref, ldb, W_host,
                              ldw);
    std::cout << "Done." << std::endl;

    // Launch CUDA convolution
    convolution_launcher(m, n, r, A_device, lda, B_device, ldb, W_device, ldw,
                         stream);
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));
    CHECK_CUDA_ERROR(cudaMemcpy(B_host, B_device, m * ldb * sizeof(T),
                                cudaMemcpyDeviceToHost));
    assert(all_close<T>(B_host, B_host_ref, m, n, ldb, abs_tol, rel_tol));

    // Measure CUDA convolution performance
    float const latency_cuda_convolution{measure_performance<void>(
        [&](cudaStream_t stream)
        {
            convolution_launcher(m, n, r, A_device, lda, B_device, ldb,
                                 W_device, ldw, stream);
            return;
        },
        stream, num_repeats, num_warmups)};

    // Release resources
    CHECK_CUDA_ERROR(cudaFree(A_device));
    CHECK_CUDA_ERROR(cudaFree(B_device));
    CHECK_CUDA_ERROR(cudaFree(W_device));
    CHECK_CUDA_ERROR(cudaFreeHost(A_host));
    CHECK_CUDA_ERROR(cudaFreeHost(B_host));
    CHECK_CUDA_ERROR(cudaFreeHost(B_host_ref));
    CHECK_CUDA_ERROR(cudaFreeHost(W_host));
    CHECK_CUDA_ERROR(cudaStreamDestroy(stream));

    // Print results
    std::cout << "Custom Convolution Kernel Performance" << std::endl;
    print_performance_result(m, n, r, latency_cuda_convolution);

    return latency_cuda_convolution;
}

#endif