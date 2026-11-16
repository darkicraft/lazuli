#pragma once

/// @file
/// @brief Reading and writing matrices in the compact binary format.

#include <cstdint>
#include <expected>
#include <filesystem>
#include <type_traits>

#include "lazuli/core/error.hpp"
#include "lazuli/matrix/matrix.hpp"
#include "lazuli/matrix/view.hpp"

namespace lazuli::detail::io {

/// Scalar tags stored in the file header.
enum class scalar_tag : std::uint8_t {
    float32 = 1,
    float64 = 2,
    complex64 = 3,
    complex128 = 4,
};

/// Reads a matrix previously written by `save`.
///
/// Any problem (missing file, wrong magic, wrong scalar type, truncated payload,
/// impossible dimensions) is reported as `error_code::io_parse_error`.
template <typename T>
[[nodiscard]] std::expected<matrix<T>, error> load(const std::filesystem::path& path);

/// Writes a view in the compact binary format. `T` may be const.
template <typename T>
[[nodiscard]] std::expected<void, error> save(const std::filesystem::path& path,
                                              matrix_view<T> data);

/// Writes an owning matrix in the compact binary format.
template <typename T>
[[nodiscard]] std::expected<void, error> save(const std::filesystem::path& path,
                                              const matrix<T>& data) {
    return save<const T>(
        path,
        matrix_view<const T>{
            data.data(), data.rows(), data.cols(), data.row_stride(), data.col_stride()});
}

/// Number of bytes a scalar of type `T` occupies in the file payload.
template <typename T>
[[nodiscard]] constexpr std::size_t scalar_size() noexcept {
    return sizeof(std::remove_const_t<T>);
}

}  // namespace lazuli::detail::io
