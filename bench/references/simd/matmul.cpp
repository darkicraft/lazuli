#include <lazuli/matrix/view.hpp>

#include <cstddef>

#if defined(__x86_64__)
#include <immintrin.h>
#endif

namespace lazuli::bench::simd {

using lazuli::detail::matrix_view;

namespace {

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

__attribute__((target("avx2,fma"))) void matmul_avx2(matrix_view<const double> a,
                                                     matrix_view<const double> b,
                                                     matrix_view<double> out) {
    const std::size_t inner_size = a.cols();
    const std::size_t column_count = b.cols();
    for (std::size_t row = 0; row < a.rows(); ++row) {
        double* out_row = out.data() + row * out.row_stride();
        for (std::size_t inner = 0; inner < inner_size; ++inner) {
            const double factor = a(row, inner);
            const __m256d broadcast = _mm256_set1_pd(factor);
            const double* b_row = b.data() + inner * b.row_stride();
            std::size_t column = 0;
            for (; column + 3 < column_count; column += 4) {
                __m256d target = _mm256_loadu_pd(out_row + column);
                const __m256d source = _mm256_loadu_pd(b_row + column);
                target = _mm256_fmadd_pd(broadcast, source, target);
                _mm256_storeu_pd(out_row + column, target);
            }
            for (; column < column_count; ++column) {
                out_row[column] += factor * b_row[column];
            }
        }
    }
}

#endif  // defined(__x86_64__)

}  // namespace

bool available() noexcept {
#if defined(__x86_64__)
    return __builtin_cpu_supports("avx2") != 0;
#else
    return false;
#endif
}

void matmul(matrix_view<const double> a, matrix_view<const double> b, matrix_view<double> out) {
#if defined(__x86_64__)
    if (b.col_stride() == 1 && out.col_stride() == 1) {
        matmul_avx2(a, b, out);
        return;
    }
#endif
    matmul_scalar(a, b, out);
}

}  // namespace lazuli::bench::simd
