#pragma once

/// @file
/// @brief `std::format` support for matrices and vectors.

#include <lazuli/matrix/matrix.hpp>
#include <lazuli/matrix/vector.hpp>

#include <format>

namespace std {

/// Formats a matrix as `[[1 2]; [3 4]]`, row-major, one bracket group per row.
template <typename T>
struct formatter<lazuli::detail::matrix<T>, char> {
    constexpr auto parse(format_parse_context& context) { return context.begin(); }

    template <typename Context>
    auto format(const lazuli::detail::matrix<T>& value, Context& context) const {
        auto out = std::format_to(context.out(), "[");
        for (std::size_t row = 0; row < value.rows(); ++row) {
            if (row != 0) {
                out = std::format_to(out, "; ");
            }
            out = std::format_to(out, "[");
            for (std::size_t column = 0; column < value.cols(); ++column) {
                if (column != 0) {
                    out = std::format_to(out, " ");
                }
                out = std::format_to(out, "{}", value(row, column));
            }
            out = std::format_to(out, "]");
        }
        return std::format_to(out, "]");
    }
};

/// Formats a vector as `[1 2 3]`.
template <typename T>
struct formatter<lazuli::detail::vector<T>, char> {
    constexpr auto parse(format_parse_context& context) { return context.begin(); }

    template <typename Context>
    auto format(const lazuli::detail::vector<T>& value, Context& context) const {
        auto out = std::format_to(context.out(), "[");
        for (std::size_t index = 0; index < value.size(); ++index) {
            if (index != 0) {
                out = std::format_to(out, " ");
            }
            out = std::format_to(out, "{}", value[index]);
        }
        return std::format_to(out, "]");
    }
};

}  // namespace std
