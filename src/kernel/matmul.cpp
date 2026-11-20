#include <lazuli/kernel/matmul.hpp>

#include <algorithm>
#include <cstddef>
#include <vector>

#if defined(__x86_64__)
#include <immintrin.h>
#endif

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace lazuli::detail::kernel {

namespace {

// Blocking, chosen from the sweep recorded in bench/README.md rather than guessed.
constexpr std::size_t block_m = 96;
constexpr std::size_t block_n = 256;
constexpr std::size_t block_k = 128;

// The register tile: 12 rows of accumulators, which leaves room for the broadcast operand and
// the vector of B on a 16-register machine.
constexpr std::size_t tile_rows = 12;

// Below this much arithmetic the fork/join costs more than it saves.
constexpr std::size_t parallel_threshold = 64 * 64 * 64;

/// Straightforward i-k-j product. Used for non-x86 targets, pre-AVX2 CPUs and destinations whose
/// columns are not contiguous, where the vector stores cannot be used.
template <typename T, bool subtract>
void scalar_matmul(matrix_view<const T> a, matrix_view<const T> b, matrix_view<T> out) {
    if constexpr (!subtract) {
        out.fill(T{});
    }
    for (std::size_t row = 0; row < a.rows(); ++row) {
        for (std::size_t inner = 0; inner < a.cols(); ++inner) {
            const T value = a(row, inner);
            for (std::size_t column = 0; column < b.cols(); ++column) {
                if constexpr (subtract) {
                    out(row, column) -= value * b(inner, column);
                } else {
                    out(row, column) += value * b(inner, column);
                }
            }
        }
    }
}

#if defined(__x86_64__)

[[nodiscard]] bool has_avx2() noexcept {
    static const bool supported = __builtin_cpu_supports("avx2") != 0;
    return supported;
}

/// The AVX2 vector operations of one scalar type: four doubles or eight floats per register.
template <typename T>
struct simd;

template <>
struct simd<double> {
    using vector = __m256d;
    static constexpr std::size_t width = 4;

    [[nodiscard]] __attribute__((target("avx2,fma"))) static vector load(
        const double* source) noexcept {
        return _mm256_loadu_pd(source);
    }
    __attribute__((target("avx2,fma"))) static void store(double* destination,
                                                          vector value) noexcept {
        _mm256_storeu_pd(destination, value);
    }
    [[nodiscard]] __attribute__((target("avx2,fma"))) static vector broadcast(
        const double* source) noexcept {
        return _mm256_broadcast_sd(source);
    }
    [[nodiscard]] __attribute__((target("avx2,fma"))) static vector multiply_add(
        vector a, vector b, vector c) noexcept {
        return _mm256_fmadd_pd(a, b, c);
    }
    [[nodiscard]] __attribute__((target("avx2,fma"))) static vector multiply_subtract(
        vector a, vector b, vector c) noexcept {
        return _mm256_fnmadd_pd(a, b, c);
    }
};

template <>
struct simd<float> {
    using vector = __m256;
    static constexpr std::size_t width = 8;

    [[nodiscard]] __attribute__((target("avx2,fma"))) static vector load(
        const float* source) noexcept {
        return _mm256_loadu_ps(source);
    }
    __attribute__((target("avx2,fma"))) static void store(float* destination,
                                                          vector value) noexcept {
        _mm256_storeu_ps(destination, value);
    }
    [[nodiscard]] __attribute__((target("avx2,fma"))) static vector broadcast(
        const float* source) noexcept {
        return _mm256_broadcast_ss(source);
    }
    [[nodiscard]] __attribute__((target("avx2,fma"))) static vector multiply_add(
        vector a, vector b, vector c) noexcept {
        return _mm256_fmadd_ps(a, b, c);
    }
    [[nodiscard]] __attribute__((target("avx2,fma"))) static vector multiply_subtract(
        vector a, vector b, vector c) noexcept {
        return _mm256_fnmadd_ps(a, b, c);
    }
};

/// Adds one tile of `ap * bp` into `c`. Full tiles only; the driver handles the edges.
///
/// Both panels are packed in micro-panel order: for each depth step the rows of `ap` and the
/// columns of `bp` are contiguous, so one cache line feeds each side of the multiply instead of
/// one line per element.
template <typename T, bool subtract>
__attribute__((target("avx2,fma"))) void avx2_tile(
    const T* ap, const T* bp, std::size_t ldb, T* c, std::size_t ldc, std::size_t kc) {
    using simd_type = simd<T>;
    using vector = typename simd_type::vector;

    vector accumulator[tile_rows];
    for (std::size_t row = 0; row < tile_rows; ++row) {
        accumulator[row] = simd_type::load(c + row * ldc);
    }

    for (std::size_t p = 0; p < kc; ++p) {
        const T* a = ap + p * tile_rows;
        const vector bv = simd_type::load(bp + p * ldb);
        for (std::size_t row = 0; row < tile_rows; ++row) {
            if constexpr (subtract) {
                accumulator[row] = simd_type::multiply_subtract(
                    simd_type::broadcast(a + row), bv, accumulator[row]);
            } else {
                accumulator[row] =
                    simd_type::multiply_add(simd_type::broadcast(a + row), bv, accumulator[row]);
            }
        }
    }

    for (std::size_t row = 0; row < tile_rows; ++row) {
        simd_type::store(c + row * ldc, accumulator[row]);
    }
}

/// Accumulates part of a tile from the packed panels, for the rows and columns the register
/// tile does not cover.
template <typename T, bool subtract>
void scalar_tile(const T* ap,
                 const T* bp,
                 std::size_t ldb,
                 T* c,
                 std::size_t ldc,
                 std::size_t rows,
                 std::size_t cols,
                 std::size_t kc) {
    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t column = 0; column < cols; ++column) {
            T sum = c[row * ldc + column];
            for (std::size_t p = 0; p < kc; ++p) {
                if constexpr (subtract) {
                    sum -= ap[p * tile_rows + row] * bp[p * ldb + column];
                } else {
                    sum += ap[p * tile_rows + row] * bp[p * ldb + column];
                }
            }
            c[row * ldc + column] = sum;
        }
    }
}

template <typename T, bool subtract>
__attribute__((target("avx2,fma"))) void avx2_matmul(matrix_view<const T> a,
                                                     matrix_view<const T> b,
                                                     matrix_view<T> out,
                                                     options settings) {
    constexpr std::size_t tile_cols = simd<T>::width;
    const std::size_t m = a.rows();
    const std::size_t k = a.cols();
    const std::size_t n = b.cols();

    std::vector<T> packed_b((block_n + tile_cols) * block_k);
    if constexpr (!subtract) {
        out.fill(T{});
    }

    // Never touch the process-wide OpenMP defaults; ask for a thread count per call instead.
    // At most one worker per row block: asking for twelve workers to spread two blocks of work
    // is fork/join plus spin-wait for no gain, and measurably slower than one thread.
    const auto row_blocks = static_cast<int>((m + block_m - 1) / block_m);
    int workers = 1;
#if defined(_OPENMP)
    workers =
        settings.num_threads == 0 ? omp_get_max_threads() : static_cast<int>(settings.num_threads);
    workers = std::min(workers, row_blocks);
#endif
    const bool parallel = workers > 1 && m * k * n >= parallel_threshold;

    for (std::size_t jc = 0; jc < n; jc += block_n) {
        const std::size_t nc = (n - jc < block_n) ? n - jc : block_n;
        for (std::size_t pc = 0; pc < k; pc += block_k) {
            const std::size_t kc = (k - pc < block_k) ? k - pc : block_k;

            // The B panel is packed once and shared read-only by every worker.
            for (std::size_t j0 = 0; j0 < nc; j0 += tile_cols) {
                const std::size_t cols = (nc - j0 < tile_cols) ? nc - j0 : tile_cols;
                T* panel = packed_b.data() + (j0 / tile_cols) * kc * tile_cols;
                for (std::size_t p = 0; p < kc; ++p) {
                    const T* source =
                        b.data() + (pc + p) * b.row_stride() + (jc + j0) * b.col_stride();
                    for (std::size_t column = 0; column < cols; ++column) {
                        panel[p * tile_cols + column] = source[column * b.col_stride()];
                    }
                }
            }

            // Row blocks of the output are independent, so they are the unit of work.
#pragma omp parallel for schedule(dynamic) if (parallel) num_threads(workers)
            for (std::size_t ic = 0; ic < m; ic += block_m) {
                const std::size_t mc = (m - ic < block_m) ? m - ic : block_m;
                std::vector<T> packed_a(block_m * block_k);

                for (std::size_t i0 = 0; i0 < mc; i0 += tile_rows) {
                    const std::size_t rows = (mc - i0 < tile_rows) ? mc - i0 : tile_rows;
                    T* panel = packed_a.data() + (i0 / tile_rows) * kc * tile_rows;
                    for (std::size_t row = 0; row < rows; ++row) {
                        const T* source =
                            a.data() + (ic + i0 + row) * a.row_stride() + pc * a.col_stride();
                        for (std::size_t p = 0; p < kc; ++p) {
                            panel[p * tile_rows + row] = source[p * a.col_stride()];
                        }
                    }
                }

                const std::size_t rows_done = (mc / tile_rows) * tile_rows;
                const std::size_t cols_done = (nc / tile_cols) * tile_cols;

                for (std::size_t i0 = 0; i0 < rows_done; i0 += tile_rows) {
                    for (std::size_t j0 = 0; j0 < cols_done; j0 += tile_cols) {
                        avx2_tile<T, subtract>(
                            packed_a.data() + (i0 / tile_rows) * kc * tile_rows,
                            packed_b.data() + (j0 / tile_cols) * kc * tile_cols,
                            tile_cols,
                            out.data() + (ic + i0) * out.row_stride() + (jc + j0),
                            out.row_stride(),
                            kc);
                    }
                }

                // Rows that do not fill a register tile, over the full column blocks.
                for (std::size_t i0 = rows_done; i0 < mc; ++i0) {
                    const T* panel =
                        packed_a.data() + (i0 / tile_rows) * kc * tile_rows + (i0 % tile_rows);
                    for (std::size_t j0 = 0; j0 < cols_done; j0 += tile_cols) {
                        scalar_tile<T, subtract>(
                            panel,
                            packed_b.data() + (j0 / tile_cols) * kc * tile_cols,
                            tile_cols,
                            out.data() + (ic + i0) * out.row_stride() + (jc + j0),
                            out.row_stride(),
                            1,
                            tile_cols,
                            kc);
                    }
                }

                // The final partial column block, for every row.
                if (cols_done < nc) {
                    for (std::size_t i0 = 0; i0 < mc; ++i0) {
                        const T* panel =
                            packed_a.data() + (i0 / tile_rows) * kc * tile_rows + (i0 % tile_rows);
                        scalar_tile<T, subtract>(
                            panel,
                            packed_b.data() + (cols_done / tile_cols) * kc * tile_cols,
                            tile_cols,
                            out.data() + (ic + i0) * out.row_stride() + (jc + cols_done),
                            out.row_stride(),
                            1,
                            nc - cols_done,
                            kc);
                    }
                }
            }
        }
    }
}

#endif  // x86-64

}  // namespace

template <bool subtract>
void dispatch(matrix_view<const float> a,
              matrix_view<const float> b,
              matrix_view<float> out,
              options settings) {
#if defined(__x86_64__)
    if (has_avx2() && out.col_stride() == 1) {
        avx2_matmul<float, subtract>(a, b, out, settings);
        return;
    }
#endif
    scalar_matmul<float, subtract>(a, b, out);
}

template <bool subtract>
void dispatch(matrix_view<const double> a,
              matrix_view<const double> b,
              matrix_view<double> out,
              options settings) {
#if defined(__x86_64__)
    if (has_avx2() && out.col_stride() == 1) {
        avx2_matmul<double, subtract>(a, b, out, settings);
        return;
    }
#endif
    scalar_matmul<double, subtract>(a, b, out);
}

void matmul(matrix_view<const float> a,
            matrix_view<const float> b,
            matrix_view<float> out,
            options settings) {
    dispatch<false>(a, b, out, settings);
}

void matmul(matrix_view<const double> a,
            matrix_view<const double> b,
            matrix_view<double> out,
            options settings) {
    dispatch<false>(a, b, out, settings);
}

void matmul_subtract(matrix_view<const float> a,
                     matrix_view<const float> b,
                     matrix_view<float> out,
                     options settings) {
    dispatch<true>(a, b, out, settings);
}

void matmul_subtract(matrix_view<const double> a,
                     matrix_view<const double> b,
                     matrix_view<double> out,
                     options settings) {
    dispatch<true>(a, b, out, settings);
}

}  // namespace lazuli::detail::kernel
