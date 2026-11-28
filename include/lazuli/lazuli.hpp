#pragma once

/// @file
/// @brief Umbrella header for the public Lazuli API.
///
/// This is the only include a user needs. Every type below lives in `lazuli::detail`; this block
/// is the public surface, and nothing outside it is supported. The algorithms are declared here
/// in two forms: `lazuli::` throws on failure, `lazuli::nt::` returns `expected`.

#include <lazuli/core/error.hpp>
#include <lazuli/core/expected.hpp>
#include <lazuli/core/options.hpp>
#include <lazuli/decomp/decomp.hpp>
#include <lazuli/expr/expr.hpp>
#include <lazuli/facade.hpp>
#include <lazuli/io/io.hpp>
#include <lazuli/matrix/format.hpp>
#include <lazuli/matrix/matrix.hpp>
#include <lazuli/matrix/vector.hpp>
#include <lazuli/matrix/view.hpp>

/// Public Lazuli API.
namespace lazuli {

/// Classification of every failure the library reports.
using detail::error_code;

/// Value form of a failure, used by the `expected` facade.
using detail::error;

/// A value or an error, used by the `lazuli::nt` facade.
using detail::expected;

/// The error half of an `expected`.
using detail::unexpected;

/// Tag for constructing an error result in place.
using detail::unexpect;
using detail::unexpect_t;

/// Thrown by `expected::value()` when it holds an error.
using detail::bad_expected_access;

/// Two operands, or a request, disagreed about a size.
using detail::dimension_mismatch;

/// A factorisation was handed a matrix that is not square.
using detail::not_square;

/// A symmetric-only algorithm was handed a matrix that is not symmetric.
using detail::not_symmetric;

/// A factorisation met a matrix it cannot decompose.
using detail::singular_matrix;

/// A Cholesky factorisation met a matrix that is not positive definite.
using detail::not_positive_definite;

/// An iterative algorithm ran out of sweeps before meeting its tolerance.
using detail::not_converged;

/// A file could not be read or written in the expected format.
using detail::io_error;

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

/// The factors of `A = Q R`.
using detail::decomp::qr_result;

/// Determinant, zero for a singular matrix.
using detail::decomp::det;

/// Stable, human readable spelling of an error code.
using detail::to_string;

}  // namespace lazuli
