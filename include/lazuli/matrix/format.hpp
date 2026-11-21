#pragma once

/// @file
/// @brief `std::format` support for matrices and vectors.
///
/// Display only. The output is bracketed, may be shortened with `...`, and honours precision
/// specs, none of which survive a round trip; use `lazuli::io` for data.

#include <lazuli/matrix/matrix.hpp>
#include <lazuli/matrix/vector.hpp>

#include <format>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace lazuli::detail::format_detail {

/// Beyond this many elements the output is shortened, like NumPy does.
inline constexpr std::size_t summarise_threshold = 1000;

/// How many leading and trailing rows or columns are kept when shortening.
inline constexpr std::size_t edge_items = 3;

/// Marker for "a run of entries was skipped".
inline constexpr std::size_t ellipsis = std::numeric_limits<std::size_t>::max();

[[nodiscard]] inline std::vector<std::size_t> visible_indices(std::size_t count, bool summarise) {
    std::vector<std::size_t> result;
    if (!summarise || count <= 2 * edge_items + 1) {
        for (std::size_t index = 0; index < count; ++index) {
            result.push_back(index);
        }
        return result;
    }
    for (std::size_t index = 0; index < edge_items; ++index) {
        result.push_back(index);
    }
    result.push_back(ellipsis);
    for (std::size_t index = count - edge_items; index < count; ++index) {
        result.push_back(index);
    }
    return result;
}

template <typename T, typename Output>
Output write_value(Output out, T value, const std::string& element_format) {
    if (element_format.empty()) {
        return std::format_to(out, "{}", value);
    }
    return std::vformat_to(out, element_format, std::make_format_args(value));
}

/// Reads a format spec of the form `[#][float-spec]`: `#` selects one row per line, everything
/// else is handed to the scalar formatter for each element. Returns the spec and the position of
/// the closing brace, which is what `parse` must return.
template <typename Iterator>
[[nodiscard]] constexpr std::pair<std::string, Iterator> read_spec(bool& multiline,
                                                                   Iterator iterator,
                                                                   Iterator end) {
    if (iterator != end && *iterator == '#') {
        multiline = true;
        ++iterator;
    }
    std::string spec;
    while (iterator != end && *iterator != '}') {
        spec.push_back(*iterator);
        ++iterator;
    }
    return {spec, iterator};
}

}  // namespace lazuli::detail::format_detail

namespace std {

/// Formats a matrix as `[[1 2] [3 4]]`, or one row per line with `{:#}`.
template <typename T>
struct formatter<lazuli::detail::matrix<T>, char> {
    bool multiline = false;
    std::string element_format;

    constexpr auto parse(format_parse_context& context) {
        auto [spec, position] =
            lazuli::detail::format_detail::read_spec(multiline, context.begin(), context.end());
        if (!spec.empty()) {
            element_format = "{:" + spec + "}";
        }
        return position;
    }

    template <typename Context>
    auto format(const lazuli::detail::matrix<T>& value, Context& context) const {
        using namespace lazuli::detail::format_detail;
        auto out = context.out();
        if (value.rows() == 0 || value.cols() == 0) {
            return std::format_to(out, "[]");
        }

        const bool summarise = value.rows() * value.cols() > summarise_threshold;
        const auto rows = visible_indices(value.rows(), summarise);
        const auto columns = visible_indices(value.cols(), summarise);

        out = std::format_to(out, "[");
        bool first_row = true;
        for (const std::size_t row : rows) {
            if (!first_row) {
                if (multiline) {
                    out = std::format_to(out, "\n ");
                } else {
                    out = std::format_to(out, " ");
                }
            }
            first_row = false;
            if (row == ellipsis) {
                out = std::format_to(out, "...");
                continue;
            }
            out = std::format_to(out, "[");
            bool first_column = true;
            for (const std::size_t column : columns) {
                if (!first_column) {
                    out = std::format_to(out, " ");
                }
                first_column = false;
                if (column == ellipsis) {
                    out = std::format_to(out, "...");
                    continue;
                }
                out = write_value(out, value(row, column), element_format);
            }
            out = std::format_to(out, "]");
        }
        return std::format_to(out, "]");
    }
};

/// Formats a vector as `[1 2 3]`, shortened beyond the same threshold. The `#` flag has no
/// meaning for a vector, which NumPy also prints on one line.
template <typename T>
struct formatter<lazuli::detail::vector<T>, char> {
    std::string element_format;

    constexpr auto parse(format_parse_context& context) {
        bool ignored = false;
        auto [spec, position] =
            lazuli::detail::format_detail::read_spec(ignored, context.begin(), context.end());
        if (!spec.empty()) {
            element_format = "{:" + spec + "}";
        }
        return position;
    }

    template <typename Context>
    auto format(const lazuli::detail::vector<T>& value, Context& context) const {
        using namespace lazuli::detail::format_detail;
        auto out = context.out();
        if (value.size() == 0) {
            return std::format_to(out, "[]");
        }

        const bool summarise = value.size() > summarise_threshold;
        const auto indices = visible_indices(value.size(), summarise);

        out = std::format_to(out, "[");
        bool first = true;
        for (const std::size_t index : indices) {
            if (!first) {
                out = std::format_to(out, " ");
            }
            first = false;
            if (index == ellipsis) {
                out = std::format_to(out, "...");
                continue;
            }
            out = write_value(out, value[index], element_format);
        }
        return std::format_to(out, "]");
    }
};

}  // namespace std
