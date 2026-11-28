#pragma once

/// @file
/// @brief The default throwing facade, and its no-throw counterpart.
///
/// The algorithms underneath report failure as a value. `lazuli::` unwraps that value and throws
/// a typed exception; `lazuli::nt::` exposes it unchanged, for code that must not throw. Both
/// reach the same implementation.
///
/// When the compiler is invoked with exceptions disabled, `lazuli::nt::` remains unaffected and
/// every call to a `lazuli::` name stops at compile time with a message pointing at its `nt`
/// equivalent.

#include <lazuli/core/error.hpp>
#include <lazuli/decomp/decomp.hpp>
#include <lazuli/io/io.hpp>

#include <cstddef>
#include <filesystem>
#include <utility>
#include <vector>

/// The no-throw facade: every fallible operation returns `expected`.
namespace lazuli::nt {

using detail::decomp::cholesky;
using detail::decomp::det;
using detail::decomp::eigenvalues;
using detail::decomp::inverse;
using detail::decomp::lu;
using detail::decomp::lu_in_place;
using detail::decomp::qr;
using detail::decomp::solve;

/// Reading and writing matrices without exceptions.
namespace io {

using detail::io::load;
using detail::io::save;

}  // namespace io

}  // namespace lazuli::nt

#if defined(__cpp_exceptions)

/// The default facade: the answer on success, a typed exception on failure.
namespace lazuli {

/// Factors a square matrix: `P A = L U`.
template <detail::scalar T>
detail::decomp::lu_result<T> lu(const detail::matrix<T>& a, detail::options settings = {}) {
    auto result = detail::decomp::lu(a, settings);
    if (!result) {
        detail::fail(result.error());
    }
    return *std::move(result);
}

/// Solves `A x = b` through the LU factors.
template <detail::scalar T>
detail::vector<T> solve(const detail::matrix<T>& a,
                        const detail::vector<T>& b,
                        detail::options settings = {}) {
    auto result = detail::decomp::solve(a, b, settings);
    if (!result) {
        detail::fail(result.error());
    }
    return *std::move(result);
}

/// Inverse through the LU factors.
template <detail::scalar T>
detail::matrix<T> inverse(const detail::matrix<T>& a, detail::options settings = {}) {
    auto result = detail::decomp::inverse(a, settings);
    if (!result) {
        detail::fail(result.error());
    }
    return *std::move(result);
}

/// Cholesky factor `L` with `L L^T = A`, for symmetric positive-definite matrices.
template <detail::scalar T>
detail::matrix<T> cholesky(const detail::matrix<T>& a, detail::options settings = {}) {
    auto result = detail::decomp::cholesky(a, settings);
    if (!result) {
        detail::fail(result.error());
    }
    return *std::move(result);
}

/// Factors a square matrix in place, overwriting it with `L` below the diagonal and `U` on and
/// above it. Returns the row permutation.
template <detail::scalar T>
std::vector<std::size_t> lu_in_place(detail::matrix_view<T> work, detail::options settings = {}) {
    auto result = detail::decomp::lu_in_place(work, settings);
    if (!result) {
        detail::fail(result.error());
    }
    return *std::move(result);
}

/// Householder QR: `A = Q R` with `Q` orthogonal.
template <detail::scalar T>
detail::decomp::qr_result<T> qr(const detail::matrix<T>& a, detail::options settings = {}) {
    auto result = detail::decomp::qr(a, settings);
    if (!result) {
        detail::fail(result.error());
    }
    return *std::move(result);
}

/// Eigenvalues of a symmetric matrix, ascending.
template <detail::scalar T>
detail::vector<T> eigenvalues(const detail::matrix<T>& a) {
    auto result = detail::decomp::eigenvalues(a);
    if (!result) {
        detail::fail(result.error());
    }
    return *std::move(result);
}

/// Reading and writing matrices, with failures as exceptions.
namespace io {

/// Reads a matrix previously written by `save`, throwing `io_error` on any problem.
template <detail::scalar T>
detail::matrix<T> load(const std::filesystem::path& path) {
    auto result = detail::io::load<T>(path);
    if (!result) {
        detail::fail(result.error());
    }
    return *std::move(result);
}

/// Writes a view in the compact binary format, throwing `io_error` on any problem.
template <detail::scalar T>
void save(const std::filesystem::path& path, detail::matrix_view<T> data) {
    auto result = detail::io::save<T>(path, data);
    if (!result) {
        detail::fail(result.error());
    }
}

/// Writes an owning matrix in the compact binary format.
template <detail::scalar T>
void save(const std::filesystem::path& path, const detail::matrix<T>& data) {
    auto result = detail::io::save<T>(path, data);
    if (!result) {
        detail::fail(result.error());
    }
}

}  // namespace io

}  // namespace lazuli

#else

/// The default facade is unavailable without exceptions. These stubs exist only so that a misuse
/// names the fix instead of failing with "no member named 'lu' in namespace 'lazuli'".
namespace lazuli::detail {

template <typename>
inline constexpr bool dependent_false = false;

}  // namespace lazuli::detail

namespace lazuli {

template <detail::scalar T>
detail::decomp::lu_result<T> lu(const detail::matrix<T>&, detail::options = {}) {
    static_assert(detail::dependent_false<T>,
                  "lazuli: this translation unit has exceptions disabled; use lazuli::nt::lu "
                  "instead");
    return {};
}

template <detail::scalar T>
detail::vector<T> solve(const detail::matrix<T>&, const detail::vector<T>&, detail::options = {}) {
    static_assert(detail::dependent_false<T>,
                  "lazuli: this translation unit has exceptions disabled; use lazuli::nt::solve "
                  "instead");
    return {};
}

template <detail::scalar T>
detail::matrix<T> inverse(const detail::matrix<T>&, detail::options = {}) {
    static_assert(detail::dependent_false<T>,
                  "lazuli: this translation unit has exceptions disabled; use "
                  "lazuli::nt::inverse instead");
    return {};
}

template <detail::scalar T>
detail::matrix<T> cholesky(const detail::matrix<T>&, detail::options = {}) {
    static_assert(detail::dependent_false<T>,
                  "lazuli: this translation unit has exceptions disabled; use "
                  "lazuli::nt::cholesky instead");
    return {};
}

template <detail::scalar T>
std::vector<std::size_t> lu_in_place(detail::matrix_view<T>, detail::options = {}) {
    static_assert(detail::dependent_false<T>,
                  "lazuli: this translation unit has exceptions disabled; use "
                  "lazuli::nt::lu_in_place instead");
    return {};
}

template <detail::scalar T>
detail::decomp::qr_result<T> qr(const detail::matrix<T>&, detail::options = {}) {
    static_assert(detail::dependent_false<T>,
                  "lazuli: this translation unit has exceptions disabled; use lazuli::nt::qr "
                  "instead");
    return {};
}

template <detail::scalar T>
detail::vector<T> eigenvalues(const detail::matrix<T>&) {
    static_assert(detail::dependent_false<T>,
                  "lazuli: this translation unit has exceptions disabled; use "
                  "lazuli::nt::eigenvalues instead");
    return {};
}

namespace io {

template <detail::scalar T>
detail::matrix<T> load(const std::filesystem::path&) {
    static_assert(detail::dependent_false<T>,
                  "lazuli: this translation unit has exceptions disabled; use "
                  "lazuli::nt::io::load instead");
    return {};
}

template <detail::scalar T>
void save(const std::filesystem::path&, detail::matrix_view<T>) {
    static_assert(detail::dependent_false<T>,
                  "lazuli: this translation unit has exceptions disabled; use "
                  "lazuli::nt::io::save instead");
}

template <detail::scalar T>
void save(const std::filesystem::path&, const detail::matrix<T>&) {
    static_assert(detail::dependent_false<T>,
                  "lazuli: this translation unit has exceptions disabled; use "
                  "lazuli::nt::io::save instead");
}

}  // namespace io

}  // namespace lazuli

#endif  // defined(__cpp_exceptions)
