#pragma once

/// @file
/// @brief Template implementations of the decompositions.
///
/// Private to the library: only `decomp.cpp` includes it, which is what keeps the algorithms out
/// of user translation units while still letting them be written once for both scalar types.

#include <lazuli/core/error.hpp>
#include <lazuli/core/options.hpp>
#include <lazuli/core/scalar.hpp>
#include <lazuli/decomp/decomp.hpp>
#include <lazuli/kernel/matmul.hpp>
#include <lazuli/matrix/vector.hpp>
#include <lazuli/matrix/view.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace lazuli::detail::decomp {

namespace impl {

/// Panel width shared by the factorisations.
inline constexpr std::size_t panel_width = 64;

[[nodiscard]] inline error make_error(error_code code, std::string message) {
    return error{code, std::move(message)};
}

/// Factors `work` in place; `pivots` must start as the identity permutation.
///
/// The panel is copied into a compact buffer before it is factorised: a panel of a row-major
/// matrix is column-oriented, so every access in the pivot search, the scaling and the rank-1
/// update would otherwise be an n-element stride. Row swaps are applied to the parts of the row
/// that live outside the panel, because a permutation touches the whole row.
template <scalar T>
[[nodiscard]] std::expected<std::size_t, error> lu_factor(matrix_view<T> work,
                                                          std::vector<std::size_t>& pivots,
                                                          options settings) {
    const std::size_t n = work.rows();
    std::size_t swaps = 0;

    for (std::size_t k = 0; k < n; k += panel_width) {
        const std::size_t end = std::min(k + panel_width, n);
        const std::size_t width = end - k;
        const std::size_t rows = n - k;

        std::vector<T> panel(rows * width);
        const auto at = [&panel, width](std::size_t row, std::size_t column) -> T& {
            return panel[row * width + column];
        };

        for (std::size_t row = 0; row < rows; ++row) {
            for (std::size_t column = 0; column < width; ++column) {
                at(row, column) = work(k + row, k + column);
            }
        }

        for (std::size_t column = 0; column < width; ++column) {
            std::size_t pivot = column;
            T largest = std::abs(at(column, column));
            for (std::size_t row = column + 1; row < rows; ++row) {
                const T value = std::abs(at(row, column));
                if (value > largest) {
                    largest = value;
                    pivot = row;
                }
            }
            if (largest == T{}) {
                return std::unexpected{make_error(error_code::singular, "zero pivot")};
            }

            if (pivot != column) {
                for (std::size_t c = 0; c < width; ++c) {
                    std::swap(at(column, c), at(pivot, c));
                }
                // The permutation applies to the whole row, not just the panel.
                const std::size_t first = k + column;
                const std::size_t second = k + pivot;
                for (std::size_t c = 0; c < k; ++c) {
                    std::swap(work(first, c), work(second, c));
                }
                for (std::size_t c = end; c < n; ++c) {
                    std::swap(work(first, c), work(second, c));
                }
                std::swap(pivots[first], pivots[second]);
                ++swaps;
            }

            const T pivot_value = at(column, column);
            for (std::size_t row = column + 1; row < rows; ++row) {
                at(row, column) /= pivot_value;
            }
            for (std::size_t row = column + 1; row < rows; ++row) {
                const T multiplier = at(row, column);
                if (multiplier == T{}) {
                    continue;
                }
                for (std::size_t c = column + 1; c < width; ++c) {
                    at(row, c) -= multiplier * at(column, c);
                }
            }
        }

        for (std::size_t row = 0; row < rows; ++row) {
            for (std::size_t column = 0; column < width; ++column) {
                work(k + row, k + column) = at(row, column);
            }
        }

        // A22 -= L21 * U12, with the kernel doing the multiply.
        if (end < n) {
            const std::size_t trailing = n - end;
            matrix<T> product(trailing, trailing);
            kernel::matmul(work.block(end, k, trailing, width),
                           work.block(k, end, width, trailing),
                           product.view(),
                           settings);
            for (std::size_t row = 0; row < trailing; ++row) {
                for (std::size_t column = 0; column < trailing; ++column) {
                    work(end + row, end + column) -= product(row, column);
                }
            }
        }
    }
    return swaps;
}

template <scalar T>
[[nodiscard]] std::expected<lu_result<T>, error> lu_impl(const matrix<T>& a, options settings) {
    if (a.rows() != a.cols()) {
        return std::unexpected{make_error(error_code::not_square, "LU needs a square matrix")};
    }
    const std::size_t n = a.rows();

    matrix<T> work = a;
    std::vector<std::size_t> pivots(n);
    for (std::size_t index = 0; index < n; ++index) {
        pivots[index] = index;
    }

    auto swaps = lu_factor<T>(work.view(), pivots, settings);
    if (!swaps) {
        return std::unexpected{swaps.error()};
    }

    lu_result<T> result;
    result.l = matrix<T>(n, n);
    result.u = matrix<T>(n, n);
    for (std::size_t row = 0; row < n; ++row) {
        for (std::size_t column = 0; column < n; ++column) {
            if (row > column) {
                result.l(row, column) = work(row, column);
            } else if (row == column) {
                result.l(row, column) = T{1};
                result.u(row, column) = work(row, column);
            } else {
                result.u(row, column) = work(row, column);
            }
        }
    }
    result.pivots = std::move(pivots);
    result.swaps = *swaps;
    return result;
}

/// Solves with the factors of `P A = L U` for a matrix right-hand side.
template <scalar T>
[[nodiscard]] matrix<T> substitute(const lu_result<T>& factors, const matrix<T>& rhs) {
    const std::size_t n = factors.u.rows();
    matrix<T> y(n, rhs.cols());
    for (std::size_t row = 0; row < n; ++row) {
        for (std::size_t column = 0; column < rhs.cols(); ++column) {
            y(row, column) = rhs(factors.pivots[row], column);
        }
    }

    for (std::size_t row = 0; row < n; ++row) {
        for (std::size_t inner = 0; inner < row; ++inner) {
            const T factor = factors.l(row, inner);
            if (factor == T{}) {
                continue;
            }
            for (std::size_t column = 0; column < rhs.cols(); ++column) {
                y(row, column) -= factor * y(inner, column);
            }
        }
    }

    for (std::size_t index = n; index > 0; --index) {
        const std::size_t row = index - 1;
        for (std::size_t inner = row + 1; inner < n; ++inner) {
            const T factor = factors.u(row, inner);
            if (factor == T{}) {
                continue;
            }
            for (std::size_t column = 0; column < rhs.cols(); ++column) {
                y(row, column) -= factor * y(inner, column);
            }
        }
        const T pivot = factors.u(row, row);
        for (std::size_t column = 0; column < rhs.cols(); ++column) {
            y(row, column) /= pivot;
        }
    }
    return y;
}

template <scalar T>
[[nodiscard]] std::expected<vector<T>, error> solve_impl(const matrix<T>& a,
                                                         const vector<T>& b,
                                                         options settings) {
    if (b.size() != a.rows()) {
        return std::unexpected{
            make_error(error_code::dimension_mismatch, "right-hand side has the wrong length")};
    }
    auto factors = lu_impl(a, settings);
    if (!factors) {
        return std::unexpected{factors.error()};
    }

    matrix<T> rhs(b.size(), 1);
    for (std::size_t row = 0; row < b.size(); ++row) {
        rhs(row, 0) = b[row];
    }
    const matrix<T> solution = substitute(*factors, rhs);

    vector<T> result(b.size());
    for (std::size_t row = 0; row < b.size(); ++row) {
        result[row] = solution(row, 0);
    }
    return result;
}

template <scalar T>
[[nodiscard]] std::expected<matrix<T>, error> inverse_impl(const matrix<T>& a, options settings) {
    if (a.rows() != a.cols()) {
        return std::unexpected{make_error(error_code::not_square, "inverse needs a square matrix")};
    }
    auto factors = lu_impl(a, settings);
    if (!factors) {
        return std::unexpected{factors.error()};
    }

    const std::size_t n = a.rows();
    matrix<T> identity(n, n);
    for (std::size_t index = 0; index < n; ++index) {
        identity(index, index) = T{1};
    }
    return substitute(*factors, identity);
}

template <scalar T>
[[nodiscard]] T det_impl(const matrix<T>& a) {
    if (a.rows() != a.cols()) {
        fail_dimension_mismatch("det needs a square matrix");
    }
    auto factors = lu_impl<T>(a, options{});
    if (!factors) {
        return T{};
    }
    T value = (factors->swaps % 2 == 0) ? T{1} : T{-1};
    for (std::size_t index = 0; index < a.rows(); ++index) {
        value *= factors->u(index, index);
    }
    return value;
}

template <scalar T>
[[nodiscard]] std::expected<matrix<T>, error> cholesky_impl(const matrix<T>& a, options settings) {
    if (a.rows() != a.cols()) {
        return std::unexpected{
            make_error(error_code::not_square, "Cholesky needs a square matrix")};
    }
    const std::size_t n = a.rows();

    // A product-based SPD matrix is symmetric only to rounding, so compare with a tolerance
    // scaled to the values involved instead of demanding bit equality.
    const T tolerance = std::sqrt(std::numeric_limits<T>::epsilon());
    for (std::size_t row = 0; row < n; ++row) {
        for (std::size_t column = 0; column < row; ++column) {
            const T left = a(row, column);
            const T right = a(column, row);
            const T scale_value = std::max(std::abs(left), std::abs(right)) + T{1};
            if (std::abs(left - right) > tolerance * scale_value) {
                return std::unexpected{make_error(error_code::not_positive_definite,
                                                  "Cholesky needs a symmetric matrix")};
            }
        }
    }

    matrix<T> work = a;
    for (std::size_t k = 0; k < n; k += panel_width) {
        const std::size_t end = std::min(k + panel_width, n);
        const std::size_t width = end - k;

        // Factor the diagonal block in place.
        for (std::size_t column = 0; column < width; ++column) {
            T diagonal = work(k + column, k + column);
            for (std::size_t inner = 0; inner < column; ++inner) {
                diagonal -= work(k + column, k + inner) * work(k + column, k + inner);
            }
            if (diagonal <= T{}) {
                return std::unexpected{make_error(error_code::not_positive_definite,
                                                  "matrix is not positive definite")};
            }
            const T root = std::sqrt(diagonal);
            work(k + column, k + column) = root;
            for (std::size_t row = column + 1; row < width; ++row) {
                T value = work(k + row, k + column);
                for (std::size_t inner = 0; inner < column; ++inner) {
                    value -= work(k + row, k + inner) * work(k + column, k + inner);
                }
                work(k + row, k + column) = value / root;
            }
        }

        if (end == n) {
            continue;
        }

        // Solve L21 * L11^T = A21 for the rows below the block.
        for (std::size_t row = end; row < n; ++row) {
            for (std::size_t column = 0; column < width; ++column) {
                T value = work(row, k + column);
                for (std::size_t inner = 0; inner < column; ++inner) {
                    value -= work(row, k + inner) * work(k + column, k + inner);
                }
                work(row, k + column) = value / work(k + column, k + column);
            }
        }

        // A22 -= L21 * L21^T, with the kernel doing the multiply.
        const std::size_t trailing = n - end;
        matrix<T> product(trailing, trailing);
        kernel::matmul(work.block(end, k, trailing, width),
                       work.block(end, k, trailing, width).transpose(),
                       product.view(),
                       settings);
        for (std::size_t row = 0; row < trailing; ++row) {
            for (std::size_t column = 0; column < trailing; ++column) {
                work(end + row, end + column) -= product(row, column);
            }
        }
    }

    matrix<T> lower(n, n);
    for (std::size_t row = 0; row < n; ++row) {
        for (std::size_t column = 0; column <= row; ++column) {
            lower(row, column) = work(row, column);
        }
    }
    return lower;
}

}  // namespace impl

}  // namespace lazuli::detail::decomp
