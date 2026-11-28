#pragma once

/// @file
/// @brief A thin CBLAS wrapper used by the benchmark only, never by the library.

#include <cstddef>

namespace lazuli::bench::openblas {

/// Sets the OpenBLAS worker count for the calling thread.
void set_threads(int threads);

/// The worker count OpenBLAS will use.
[[nodiscard]] int thread_count() noexcept;

/// The number of physical processors OpenBLAS detects.
[[nodiscard]] int core_count() noexcept;

/// Row-major `C = A B` through OpenBLAS.
void matmul(std::size_t rows,
            std::size_t inner,
            std::size_t cols,
            const double* a,
            const double* b,
            double* c);

/// Row-major `C = A B` through OpenBLAS, in single precision.
void matmul(std::size_t rows,
            std::size_t inner,
            std::size_t cols,
            const float* a,
            const float* b,
            float* c);

}  // namespace lazuli::bench::openblas
