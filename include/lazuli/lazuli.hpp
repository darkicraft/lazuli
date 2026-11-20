#pragma once

/// @file
/// @brief Umbrella header for the public Lazuli API.
///
/// This is the only include a user needs. Every declaration below lives in `lazuli::detail`;
/// this block is the public surface, and nothing outside it is supported.

#include <lazuli/core/error.hpp>
#include <lazuli/core/options.hpp>
#include <lazuli/decomp/decomp.hpp>
#include <lazuli/expr/expr.hpp>
#include <lazuli/io/io.hpp>
#include <lazuli/matrix/format.hpp>
#include <lazuli/matrix/matrix.hpp>
#include <lazuli/matrix/vector.hpp>
#include <lazuli/matrix/view.hpp>

/// Public Lazuli API.
namespace lazuli {

/// Classification of every failure the library reports.
using detail::error_code;

/// Value form of a failure, used by the `std::expected` facade.
using detail::error;

/// Two operands, or a request, disagreed about a size.
using detail::dimension_mismatch;

/// A factorisation met a matrix it cannot decompose.
using detail::singular_matrix;

/// A Cholesky factorisation met a matrix that is not positive definite.
using detail::not_positive_definite;

/// An index fell outside its container.
using detail::out_of_bounds;

/// Per-call settings, such as the worker thread count.
using detail::options;

/// Half-open sampling bounds for `matrix<T>::random`.
using detail::interval;

/// Row-major, contiguous, owning dense matrix.
using detail::matrix;

/// Non-owning window over a strided, dense two-dimensional block.
using detail::matrix_view;

/// Row-major, contiguous, owning one-dimensional array.
using detail::vector;

/// The factors of `P A = L U`.
using detail::decomp::lu_result;

/// Factors a square matrix: `P A = L U`.
using detail::decomp::lu;

/// Solves `A x = b` through the LU factors.
using detail::decomp::solve;

/// Determinant, zero for a singular matrix.
using detail::decomp::det;

/// Inverse through the LU factors.
using detail::decomp::inverse;

/// Cholesky factor `L` with `L L^T = A`, for symmetric positive-definite matrices.
using detail::decomp::cholesky;

/// Factors a square matrix in place, overwriting it with L and U.
using detail::decomp::lu_in_place;

/// The factors of `A = Q R`.
using detail::decomp::qr_result;

/// Householder QR: `A = Q R` with `Q` orthogonal.
using detail::decomp::qr;

/// Eigenvalues of a symmetric matrix, ascending.
using detail::decomp::eigenvalues;

/// Stable, human readable spelling of an error code.
using detail::to_string;

}  // namespace lazuli

/// Reading and writing matrices.
namespace lazuli::io {

/// Reads a matrix previously written by `save<T>`.
using detail::io::load;

/// Writes a matrix in the compact binary format.
using detail::io::save;

}  // namespace lazuli::io
