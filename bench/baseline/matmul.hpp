#pragma once

/// @file
/// @brief The frozen baseline matrix multiplication.
///
/// This is the denominator of every speedup number the project reports. It is deliberately
/// unoptimised: no blocking, no threads, no explicit vectorisation, i-k-j order so the sweep is
/// row-major. The compiler is still free to vectorise it, because a fair comparison builds the
/// baseline with the same flags as the optimised version.
///
/// Do not optimise this file. If it changes, every speedup measured before the change is void.

#include "lazuli/matrix/view.hpp"

namespace lazuli::bench::baseline {

/// `out = a * b`, with `out` zeroed by the caller.
///
/// Preconditions: `a.cols() == b.rows()`, `out.rows() == a.rows()`,
/// `out.cols() == b.cols()`, and `out` does not overlap `a` or `b`.
void matmul(const lazuli::detail::matrix_view<const double>& a,
            const lazuli::detail::matrix_view<const double>& b,
            lazuli::detail::matrix_view<double> out);

}  // namespace lazuli::bench::baseline
