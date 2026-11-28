#pragma once

/// @file
/// @brief Explicit AVX2/FMA matmul reference for the benchmark menu.
///
/// No packing, no cache blocking and no threads. This is the "SIMD" stage, not the library.

#include <lazuli/matrix/view.hpp>

namespace lazuli::bench::simd {

using lazuli::detail::matrix_view;

/// True on x86-64 with AVX2 support.
[[nodiscard]] bool available() noexcept;

/// Adds `a * b` into `out` with an explicit AVX2 i-k-j loop.
void matmul(matrix_view<const double> a, matrix_view<const double> b, matrix_view<double> out);

}  // namespace lazuli::bench::simd
