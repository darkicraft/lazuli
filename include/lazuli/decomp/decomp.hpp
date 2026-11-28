#pragma once

/// @file
/// @brief Factorisations and the operations derived from them.
///
/// Everything returns `expected` rather than throwing. The public throwing facade is a
/// header-only layer in `lazuli/facade.hpp`; keeping `throw` out of this header means it stays
/// usable with exceptions disabled.

#include <lazuli/core/error.hpp>
#include <lazuli/core/expected.hpp>
#include <lazuli/core/options.hpp>
#include <lazuli/matrix/matrix.hpp>
#include <lazuli/matrix/vector.hpp>

#include <cstddef>
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

expected<lu_result<float>, error> lu(const matrix<float>& a, options settings = {});
expected<lu_result<double>, error> lu(const matrix<double>& a, options settings = {});

expected<vector<float>, error> solve(const matrix<float>& a,
                                     const vector<float>& b,
                                     options settings = {});
expected<vector<double>, error> solve(const matrix<double>& a,
                                      const vector<double>& b,
                                      options settings = {});

float det(const matrix<float>& a);
double det(const matrix<double>& a);

expected<matrix<float>, error> inverse(const matrix<float>& a, options settings = {});
expected<matrix<double>, error> inverse(const matrix<double>& a, options settings = {});

expected<matrix<float>, error> cholesky(const matrix<float>& a, options settings = {});
expected<matrix<double>, error> cholesky(const matrix<double>& a, options settings = {});

/// Factors a square matrix in place, overwriting it with `L` below the diagonal and `U` on and
/// above it. Returns the row permutation.
expected<std::vector<std::size_t>, error> lu_in_place(matrix_view<float> work,
                                                      options settings = {});
expected<std::vector<std::size_t>, error> lu_in_place(matrix_view<double> work,
                                                      options settings = {});

/// The factors of `A = Q R`, with `Q` square and orthogonal and `R` upper triangular.
template <scalar T>
struct qr_result {
    matrix<T> q;
    matrix<T> r;
};

expected<qr_result<float>, error> qr(const matrix<float>& a, options settings = {});
expected<qr_result<double>, error> qr(const matrix<double>& a, options settings = {});

/// Eigenvalues of a symmetric matrix, ascending. Jacobi rotations, so this is O(n^3) per sweep
/// and meant for modest sizes.
expected<vector<float>, error> eigenvalues(const matrix<float>& a);
expected<vector<double>, error> eigenvalues(const matrix<double>& a);

}  // namespace lazuli::detail::decomp
