#pragma once

/// @file
/// @brief Naive LU and Cholesky references, compiled with -O0.
///
/// The target is deliberately unoptimised: no blocking, no vectorisation and no threads. The
/// benchmark prints a warning that the naive stage is not a like-for-like optimised baseline.

#include <lazuli/matrix/view.hpp>

#include <cstddef>
#include <vector>

namespace lazuli::bench::naive {

using lazuli::detail::matrix_view;

/// Scalar right-looking LU with partial pivoting.
[[nodiscard]] bool lu(matrix_view<double> a, std::vector<std::size_t>& pivots);

/// Scalar column Cholesky; the lower triangle holds L on return.
[[nodiscard]] bool cholesky(matrix_view<double> a);

}  // namespace lazuli::bench::naive
