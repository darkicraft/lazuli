#pragma once

/// @file
/// @brief Reading and writing matrices in the compact binary format.

#include <lazuli/core/error.hpp>
#include <lazuli/core/scalar.hpp>
#include <lazuli/matrix/matrix.hpp>
#include <lazuli/matrix/view.hpp>

#include <cstdint>
#include <expected>
#include <filesystem>
#include <type_traits>

namespace lazuli::detail::io {

/// Scalar tags stored in the file header.
/// 3 and 4 were complex float and complex double; complex is parked, so those tags are
/// reserved and a file carrying one is reported as a scalar type mismatch.
enum class scalar_tag : std::uint8_t {
    float32 = 1,
    float64 = 2,
    reserved_complex64 = 3,
    reserved_complex128 = 4,
};

/// Reads a matrix previously written by `save`.
///
/// Any problem (missing file, wrong magic, wrong scalar type, truncated payload,
/// impossible dimensions) is reported as `error_code::io_parse_error`.
template <scalar T>
[[nodiscard]] std::expected<matrix<T>, error> load(const std::filesystem::path& path);

/// Writes a view in the compact binary format. `T` may be const.
template <scalar T>
[[nodiscard]] std::expected<void, error> save(const std::filesystem::path& path,
                                              matrix_view<T> data);

/// Writes an owning matrix in the compact binary format.
template <scalar T>
[[nodiscard]] std::expected<void, error> save(const std::filesystem::path& path,
                                              const matrix<T>& data) {
    return save<const T>(
        path,
        matrix_view<const T>{
            data.data(), data.rows(), data.cols(), data.row_stride(), data.col_stride()});
}

}  // namespace lazuli::detail::io
