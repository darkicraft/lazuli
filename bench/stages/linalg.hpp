#pragma once

/// @file
/// @brief Bench-only references for the staged speedup table.
///
/// These are not part of the library. `*_naive` is the obvious scalar algorithm; `*_simd` uses
/// AVX2/FMA but still has no cache blocking, no packing and no threads. The library kernel is the
/// "full" stage.

#include <lazuli/matrix/view.hpp>

#include <cstddef>
#include <vector>

namespace lazuli::bench::stages {

using lazuli::detail::matrix_view;

/// True on x86-64 with AVX2 support.
[[nodiscard]] bool avx2_available() noexcept;

/// Explicit AVX2 i-k-j product, no blocking, packing or threads. Adds into `out`.
void matmul_simd(matrix_view<const double> a, matrix_view<const double> b, matrix_view<double> out);

/// Scalar right-looking LU with partial pivoting.
[[nodiscard]] bool lu_naive(matrix_view<double> a, std::vector<std::size_t>& pivots);

/// The same LU, with the trailing rank-1 update vectorised across the row.
[[nodiscard]] bool lu_simd(matrix_view<double> a, std::vector<std::size_t>& pivots);

/// Scalar column Cholesky; the lower triangle holds L on return.
[[nodiscard]] bool cholesky_naive(matrix_view<double> a);

/// The same Cholesky, with the row dot products vectorised.
[[nodiscard]] bool cholesky_simd(matrix_view<double> a);

}  // namespace lazuli::bench::stages
