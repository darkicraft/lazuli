#pragma once

/// @file
/// @brief Umbrella header for the public Lazuli API.
///
/// This is the only include a user needs. Every declaration below lives in
/// `lazuli::detail`; this block is the public surface, and nothing outside it
/// is supported.

#include "lazuli/core/error.hpp"
#include "lazuli/matrix/matrix.hpp"
#include "lazuli/matrix/vector.hpp"
#include "lazuli/matrix/view.hpp"

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

/// An index fell outside its container.
using detail::out_of_bounds;

/// Row-major, contiguous, owning dense matrix.
using detail::matrix;

/// Non-owning window over a strided, dense two-dimensional block.
using detail::matrix_view;

/// Row-major, contiguous, owning one-dimensional array.
using detail::vector;

/// Stable, human readable spelling of an error code.
using detail::to_string;

}  // namespace lazuli
