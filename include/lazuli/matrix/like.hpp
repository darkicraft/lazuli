#pragma once

/// @file
/// @brief The shape of things that can take part in an expression, and how an operand is stored.

#include <lazuli/matrix/view.hpp>

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <utility>

namespace lazuli::detail {

/// A dense grid that can be read element by element: an owning matrix, a view or an expression.
template <typename M>
concept matrix_like = requires(const M& grid) {
    typename M::value_type;
    { grid.rows() } -> std::convertible_to<std::size_t>;
    { grid.cols() } -> std::convertible_to<std::size_t>;
    { grid(std::size_t{0}, std::size_t{0}) } -> std::convertible_to<typename M::value_type>;
};

/// Stores an operand the way an expression node wants it: a node is copied, a mutable view is
/// widened to a read-only view, an owning container becomes a read-only view of its storage.
///
/// Copying an owning matrix here would be a silent O(n) cost, which is exactly what the lazy
/// layer exists to avoid, so there is no such branch.
template <typename M>
[[nodiscard]] auto to_node(M&& value) {
    using decayed = std::decay_t<M>;
    using element = typename decayed::value_type;
    if constexpr (requires { typename decayed::node_tag; }) {
        return decayed{std::forward<M>(value)};
    } else if constexpr (requires { typename decayed::view_tag; }) {
        return matrix_view<const element>{value};
    } else if constexpr (requires { typename decayed::column_tag; }) {
        return matrix_view<const element>{value.data(), value.size(), 1, 1, value.size()};
    } else {
        return matrix_view<const element>{
            value.data(), value.rows(), value.cols(), value.row_stride(), value.col_stride()};
    }
}

/// Writes any grid into `out`: expression nodes know how to do it themselves, plain views and
/// containers are copied element by element.
template <typename Node, typename T>
void assign_into(const Node& node, matrix_view<T> out) {
    if constexpr (requires { node.assign_to(out); }) {
        node.assign_to(out);
    } else {
        for (std::size_t row = 0; row < out.rows(); ++row) {
            for (std::size_t column = 0; column < out.cols(); ++column) {
                out(row, column) = static_cast<T>(node(row, column));
            }
        }
    }
}

/// True when the destination and the grid share storage.
template <typename Elem, typename Grid>
[[nodiscard]] bool overlaps(matrix_view<Elem> destination, const Grid& grid) {
    if (destination.empty() || grid.rows() == 0 || grid.cols() == 0) {
        return false;
    }

    const auto span = [](auto* data,
                         std::size_t rows,
                         std::size_t cols,
                         std::size_t row_stride,
                         std::size_t col_stride) {
        const auto first = reinterpret_cast<std::uintptr_t>(data);
        const auto last =
            first + ((rows - 1) * row_stride + (cols - 1) * col_stride) * sizeof(*data);
        return std::pair{first, last};
    };

    const auto here = span(destination.data(),
                           destination.rows(),
                           destination.cols(),
                           destination.row_stride(),
                           destination.col_stride());
    const auto there =
        span(grid.data(), grid.rows(), grid.cols(), grid.row_stride(), grid.col_stride());
    return here.first <= there.second && there.first <= here.second;
}

}  // namespace lazuli::detail
