#include <cuda_runtime.h>
#include <functional>
#include <iostream>
#include <vector>

#include "cuda_convolution.hpp"
#include "profile_utils.cuh"

__constant__ std::byte const_mem[MAX_CONST_MEM];

int main()
{
    print_device_info();

    constexpr size_t num_repeats{1U};
    constexpr size_t num_warmups{1U};

    const float fp32_abs_tol{1.0e-3f};
    const double fp32_rel_tol{0.0e-4f};

    constexpr size_t m{4096U};
    constexpr size_t n{4096U};
    constexpr size_t r{4U};          // Radius of the filter
    constexpr size_t k{2U * r + 1U}; // Width of the filter

    constexpr size_t lda{(n + 16U - 1U) / 16U * 16U};
    constexpr size_t ldb{(n + 16U - 1U) / 16U * 16U};
    constexpr size_t ldw{(k + 16U - 1U) / 16U * 16U};

    static_assert(lda >= n);
    static_assert(ldb >= n);
    static_assert(ldw >= k);

    printf("Matrix A: %zux%zu Leading Dimension Size = %zu\n", m, n, lda);
    printf("Matrix B: %zux%zu Leading Dimension Size = %zu\n", m, n, ldb);
    printf("Kernel W: %zux%zu Radius = %zu Leading Dimension Size = %zu\n", k,
           k, r, ldw);
    printf("\n");

    const std::vector<std::pair<std::string, ConvolutionLauncher<float>>>
        convolution_launchers{{"Custom Convolution Kernel V00",
                               launch_kernel_convolution_v00<float>},
                              {"Custom Convolution Kernel V01",
                               launch_kernel_convolution_v01<float>},
                              {"Custom Convolution Kernel V02",
                               launch_kernel_convolution_v02<float>},
                              {"Custom Convolution Kernel V03",
                               launch_kernel_convolution_v03<float>}};

    for (const auto& convolution_launcher : convolution_launchers)
    {
        std::cout << convolution_launcher.first << std::endl;
        float convolution_kernel_latency{profile_convolution<float>(
            m, n, r, lda, ldb, ldw, convolution_launcher.second, fp32_abs_tol,
            fp32_rel_tol, num_repeats, num_warmups)};

        std::cout << std::endl;
    }

    return 0;
}