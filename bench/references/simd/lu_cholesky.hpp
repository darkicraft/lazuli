#pragma once

/// @file
/// @brief Explicit AVX2/FMA LU and Cholesky references for the benchmark menu.

#include <lazuli/matrix/view.hpp>

#include <cstddef>
#include <vector>

namespace lazuli::bench::simd {

using lazuli::detail::matrix_view;

/// LU with the trailing rank-1 update vectorised across the row.
[[nodiscard]] bool lu(matrix_view<double> a, std::vector<std::size_t>& pivots);

/// Cholesky with the row dot products vectorised.
[[nodiscard]] bool cholesky(matrix_view<double> a);

}  // namespace lazuli::bench::simd
