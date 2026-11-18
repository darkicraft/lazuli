#pragma once

/// @file
/// @brief The matrix product behind the expression layer.
///
/// The definitions live in the library, not in this header, so the blocking, packing and SIMD
/// stay out of user translation units. `bench/baseline` keeps a frozen naive copy for comparison.

#include "lazuli/matrix/view.hpp"

namespace lazuli::detail::kernel {

/// `out = a * b`. Preconditions: `a.cols() == b.rows()`, `out` has the right shape and does not
/// overlap the operands. The internals pick the best available implementation at run time.
void matmul(matrix_view<const float> a, matrix_view<const float> b, matrix_view<float> out);
void matmul(matrix_view<const double> a, matrix_view<const double> b, matrix_view<double> out);

}  // namespace lazuli::detail::kernel
