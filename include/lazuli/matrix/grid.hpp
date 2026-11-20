#pragma once

/// @file
/// @brief The shape of a dense grid, and the operations that only need indexing.
///
/// This sits below the view and the containers so all of them can use it without a cycle.

#include <cmath>
#include <concepts>
#include <cstddef>
#include <type_traits>

namespace lazuli::detail {

/// A dense grid that can be read element by element: an owning matrix, a view or an expression.
template <typename M>
concept matrix_like = requires(const M& grid) {
    typename M::value_type;
    { grid.rows() } -> std::convertible_to<std::size_t>;
    { grid.cols() } -> std::convertible_to<std::size_t>;
    { grid(std::size_t{0}, std::size_t{0}) } -> std::convertible_to<typename M::value_type>;
};

/// Frobenius norm: the square root of the sum of the squared entries.
template <matrix_like M>
[[nodiscard]] typename M::value_type frobenius_norm(const M& grid) {
    using element = typename M::value_type;
    element sum{};
    for (std::size_t row = 0; row < grid.rows(); ++row) {
        for (std::size_t column = 0; column < grid.cols(); ++column) {
            const element value = static_cast<element>(grid(row, column));
            sum += value * value;
        }
    }
    return std::sqrt(sum);
}

/// True when the shapes match and every entry is within `tolerance` of the other's.
template <matrix_like A, matrix_like B>
[[nodiscard]] bool approx_equal(const A& left, const B& right, typename A::value_type tolerance) {
    using element = typename A::value_type;
    if (left.rows() != right.rows() || left.cols() != right.cols()) {
        return false;
    }
    for (std::size_t row = 0; row < left.rows(); ++row) {
        for (std::size_t column = 0; column < left.cols(); ++column) {
            const element difference =
                static_cast<element>(left(row, column)) - static_cast<element>(right(row, column));
            if (std::abs(difference) > tolerance) {
                return false;
            }
        }
    }
    return true;
}

}  // namespace lazuli::detail
