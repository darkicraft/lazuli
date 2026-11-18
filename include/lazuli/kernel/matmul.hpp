#pragma once

/// @file
/// @brief The naive matrix product behind the expression layer.
///
/// Deliberately unoptimised and deliberately separate from `bench/baseline`: the bench copy is
/// the frozen denominator, this one is allowed to be replaced by the blocked, threaded kernel in
/// M3/M4 without invalidating a single recorded number.

#include <cstddef>

#include "lazuli/core/scalar.hpp"
#include "lazuli/matrix/like.hpp"
#include "lazuli/matrix/view.hpp"

namespace lazuli::detail::kernel {

/// `out = a * b`. Preconditions: `a.cols() == b.rows()`, `out` is the right shape, no overlap.
template <scalar T, matrix_like A, matrix_like B>
void matmul(const A& a, const B& b, matrix_view<T> out) {
    out.fill(T{});
    for (std::size_t row = 0; row < a.rows(); ++row) {
        for (std::size_t inner = 0; inner < a.cols(); ++inner) {
            const T value = static_cast<T>(a(row, inner));
            for (std::size_t column = 0; column < b.cols(); ++column) {
                out(row, column) += value * static_cast<T>(b(inner, column));
            }
        }
    }
}

}  // namespace lazuli::detail::kernel
