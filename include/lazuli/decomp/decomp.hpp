#pragma once

/// @file
/// @brief Factorisations and the operations derived from them.
///
/// Everything returns `std::expected` rather than throwing: the throwing facade is a thin layer
/// that is not written yet, and adding it here would put `throw` back into a public header.

#include <lazuli/core/error.hpp>
#include <lazuli/core/options.hpp>
#include <lazuli/matrix/matrix.hpp>
#include <lazuli/matrix/vector.hpp>

#include <cstddef>
#include <expected>
#include <vector>

namespace lazuli::detail::decomp {

/// The factors of `P A = L U`.
template <scalar T>
struct lu_result {
    matrix<T> l;
    matrix<T> u;
    /// Row permutation: row `i` of `P A` is row `pivots[i]` of `A`.
    std::vector<std::size_t> pivots;
    /// Number of row interchanges, whose parity is the sign of the determinant.
    std::size_t swaps = 0;
};

std::expected<lu_result<float>, error> lu(const matrix<float>& a, options settings = {});
std::expected<lu_result<double>, error> lu(const matrix<double>& a, options settings = {});

std::expected<vector<float>, error> solve(const matrix<float>& a,
                                          const vector<float>& b,
                                          options settings = {});
std::expected<vector<double>, error> solve(const matrix<double>& a,
                                           const vector<double>& b,
                                           options settings = {});

float det(const matrix<float>& a);
double det(const matrix<double>& a);

std::expected<matrix<float>, error> inverse(const matrix<float>& a, options settings = {});
std::expected<matrix<double>, error> inverse(const matrix<double>& a, options settings = {});

std::expected<matrix<float>, error> cholesky(const matrix<float>& a, options settings = {});
std::expected<matrix<double>, error> cholesky(const matrix<double>& a, options settings = {});

}  // namespace lazuli::detail::decomp
