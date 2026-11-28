#include <lazuli/matrix/view.hpp>

#include <cmath>
#include <cstddef>
#include <references/simd/lu_cholesky.hpp>
#include <utility>
#include <vector>

#if defined(__x86_64__)
#include <immintrin.h>
#endif

namespace lazuli::bench::simd {

using lazuli::detail::matrix_view;

namespace {

void swap_rows(matrix_view<double> a, std::size_t first, std::size_t second) {
    for (std::size_t column = 0; column < a.cols(); ++column) {
        std::swap(a(first, column), a(second, column));
    }
}

#if defined(__x86_64__)

__attribute__((target("avx2,fma"))) double dot(const double* left,
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

__attribute__((target("avx2,fma"))) bool lu_avx2(matrix_view<double> a,
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

__attribute__((target("avx2,fma"))) bool cholesky_avx2(matrix_view<double> a) {
    const std::size_t size = a.rows();
    for (std::size_t step = 0; step < size; ++step) {
        const double* pivot_row = a.data() + step * a.row_stride();
        const double diagonal = a(step, step) - dot(pivot_row, pivot_row, step);
        if (diagonal <= 0.0) {
            return false;
        }
        const double pivot = std::sqrt(diagonal);
        a(step, step) = pivot;
        for (std::size_t row = step + 1; row < size; ++row) {
            const double* target_row = a.data() + row * a.row_stride();
            a(row, step) = (a(row, step) - dot(target_row, pivot_row, step)) / pivot;
        }
    }
    return true;
}

#endif  // defined(__x86_64__)

}  // namespace

bool lu(matrix_view<double> a, std::vector<std::size_t>& pivots) {
#if defined(__x86_64__)
    if (a.col_stride() == 1) {
        return lu_avx2(a, pivots);
    }
#endif
    return false;
}

bool cholesky(matrix_view<double> a) {
#if defined(__x86_64__)
    if (a.col_stride() == 1) {
        return cholesky_avx2(a);
    }
#endif
    return false;
}

}  // namespace lazuli::bench::simd
