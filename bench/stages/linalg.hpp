#pragma once

/// @file
/// @brief Bench-only references for the staged speedup table.
///
/// These are not part of the library. `*_simd` uses AVX2/FMA but still has no cache blocking,
/// no packing and no threads. The library kernel is the "full" stage; the naive references live
/// in `naive/` and are compiled with -O0.

#include <lazuli/matrix/view.hpp>

#include <cstddef>
#include <vector>

namespace lazuli::bench::stages {

using lazuli::detail::matrix_view;

/// True on x86-64 with AVX2 support.
[[nodiscard]] bool avx2_available() noexcept;

/// Explicit AVX2 i-k-j product, no blocking, packing or threads. Adds into `out`.
void matmul_simd(matrix_view<const double> a, matrix_view<const double> b, matrix_view<double> out);

/// AVX2 LU with the trailing rank-1 update vectorised across the row.
[[nodiscard]] bool lu_simd(matrix_view<double> a, std::vector<std::size_t>& pivots);

/// AVX2 Cholesky with the row dot products vectorised.
[[nodiscard]] bool cholesky_simd(matrix_view<double> a);

}  // namespace lazuli::bench::stages
