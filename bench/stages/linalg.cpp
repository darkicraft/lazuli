#include <lazuli/matrix/view.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stages/linalg.hpp>
#include <utility>
#include <vector>

#if defined(__x86_64__)
#include <immintrin.h>
#endif

namespace lazuli::bench::stages {

using lazuli::detail::matrix_view;

namespace {

void swap_rows(matrix_view<double> a, std::size_t first, std::size_t second) {
    for (std::size_t column = 0; column < a.cols(); ++column) {
        std::swap(a(first, column), a(second, column));
    }
}

void matmul_scalar(matrix_view<const double> a,
                   matrix_view<const double> b,
                   matrix_view<double> out) {
    for (std::size_t row = 0; row < a.rows(); ++row) {
        for (std::size_t inner = 0; inner < a.cols(); ++inner) {
            const double value = a(row, inner);
            for (std::size_t column = 0; column < b.cols(); ++column) {
                out(row, column) += value * b(inner, column);
            }
        }
    }
}

#if defined(__x86_64__)

[[nodiscard]] __attribute__((target("avx2,fma"))) double dot_simd(const double* left,
                                                                  const double* right,
                                                                  std::size_t count) noexcept {
    __m256d total = _mm256_setzero_pd();
    std::size_t index = 0;
    for (; index + 3 < count; index += 4) {
        const __m256d lhs = _mm256_loadu_pd(left + index);
        const __m256d rhs = _mm256_loadu_pd(right + index);
        total = _mm256_fmadd_pd(lhs, rhs, total);
    }
    alignas(32) double lanes[4];
    _mm256_store_pd(lanes, total);
    double result = lanes[0] + lanes[1] + lanes[2] + lanes[3];
    for (; index < count; ++index) {
        result += left[index] * right[index];
    }
    return result;
}

__attribute__((target("avx2,fma"))) void matmul_simd_impl(matrix_view<const double> a,
                                                          matrix_view<const double> b,
                                                          matrix_view<double> out) {
    const std::size_t inner_size = a.cols();
    const std::size_t column_count = b.cols();
    for (std::size_t row = 0; row < a.rows(); ++row) {
        double* out_row = out.data() + row * out.row_stride();
        for (std::size_t inner = 0; inner < inner_size; ++inner) {
            const __m256d broadcast = _mm256_set1_pd(a(row, inner));
            const double* b_row = b.data() + inner * b.row_stride();
            std::size_t column = 0;
            for (; column + 3 < column_count; column += 4) {
                __m256d target = _mm256_loadu_pd(out_row + column);
                const __m256d source = _mm256_loadu_pd(b_row + column);
                target = _mm256_fmadd_pd(broadcast, source, target);
                _mm256_storeu_pd(out_row + column, target);
            }
            for (; column < column_count; ++column) {
                out_row[column] += a(row, inner) * b_row[column];
            }
        }
    }
}

__attribute__((target("avx2,fma"))) bool lu_simd_impl(matrix_view<double> a,
                                                      std::vector<std::size_t>& pivots) {
    const std::size_t size = a.rows();
    for (std::size_t index = 0; index < size; ++index) {
        pivots[index] = index;
    }

    for (std::size_t step = 0; step < size; ++step) {
        std::size_t pivot = step;
        double best = std::abs(a(step, step));
        for (std::size_t row = step + 1; row < size; ++row) {
            const double candidate = std::abs(a(row, step));
            if (candidate > best) {
                best = candidate;
                pivot = row;
            }
        }
        if (best == 0.0) {
            return false;
        }
        if (pivot != step) {
            swap_rows(a, step, pivot);
            std::swap(pivots[step], pivots[pivot]);
        }

        const double diagonal = a(step, step);
        for (std::size_t row = step + 1; row < size; ++row) {
            a(row, step) /= diagonal;
        }

        const std::size_t count = size - (step + 1);
        const double* pivot_row = a.data() + step * a.row_stride() + step + 1;
        for (std::size_t row = step + 1; row < size; ++row) {
            const double factor = a(row, step);
            double* target_row = a.data() + row * a.row_stride() + step + 1;
            const __m256d broadcast = _mm256_set1_pd(factor);
            std::size_t column = 0;
            for (; column + 3 < count; column += 4) {
                __m256d target = _mm256_loadu_pd(target_row + column);
                const __m256d source = _mm256_loadu_pd(pivot_row + column);
                target = _mm256_fnmadd_pd(broadcast, source, target);
                _mm256_storeu_pd(target_row + column, target);
            }
            for (; column < count; ++column) {
                target_row[column] -= factor * pivot_row[column];
            }
        }
    }
    return true;
}

__attribute__((target("avx2,fma"))) bool cholesky_simd_impl(matrix_view<double> a) {
    const std::size_t size = a.rows();
    for (std::size_t step = 0; step < size; ++step) {
        const double* pivot_row = a.data() + step * a.row_stride();
        const double diagonal = a(step, step) - dot_simd(pivot_row, pivot_row, step);
        if (diagonal <= 0.0) {
            return false;
        }
        const double pivot = std::sqrt(diagonal);
        a(step, step) = pivot;
        for (std::size_t row = step + 1; row < size; ++row) {
            const double* target_row = a.data() + row * a.row_stride();
            a(row, step) = (a(row, step) - dot_simd(target_row, pivot_row, step)) / pivot;
        }
    }
    return true;
}

#endif  // defined(__x86_64__)

}  // namespace

bool avx2_available() noexcept {
#if defined(__x86_64__)
    return __builtin_cpu_supports("avx2") != 0;
#else
    return false;
#endif
}

void matmul_simd(matrix_view<const double> a,
                 matrix_view<const double> b,
                 matrix_view<double> out) {
#if defined(__x86_64__)
    if (b.col_stride() == 1 && out.col_stride() == 1) {
        matmul_simd_impl(a, b, out);
        return;
    }
#endif
    matmul_scalar(a, b, out);
}

bool lu_simd(matrix_view<double> a, std::vector<std::size_t>& pivots) {
#if defined(__x86_64__)
    if (a.col_stride() == 1) {
        return lu_simd_impl(a, pivots);
    }
#endif
    return false;
}

bool cholesky_simd(matrix_view<double> a) {
#if defined(__x86_64__)
    if (a.col_stride() == 1) {
        return cholesky_simd_impl(a);
    }
#endif
    return false;
}

}  // namespace lazuli::bench::stages
