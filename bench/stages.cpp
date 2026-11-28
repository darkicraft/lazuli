// Staged speedups: naive -> explicit SIMD -> blocked/one thread -> full/all threads.
//
//   cmake --build build/release --target bench_stages
//   ./build/release/bench/stages
//   ./build/release/bench/stages --csv
//
// The naive matmul is the frozen baseline. The SIMD stage is a bench-only AVX2 i-k-j loop with
// no packing, blocking or threads. The library kernel is the blocked, threaded stage.
// LU and Cholesky follow the same idea: a scalar factorisation, the same factorisation with the
// trailing update/dot products vectorised, then the library's blocked and threaded version.

#include <baseline/matmul.hpp>
#include <lazuli/lazuli.hpp>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <stages/linalg.hpp>
#include <string_view>
#include <vector>

namespace {

using clock_type = std::chrono::steady_clock;
using lazuli::matrix;
using lazuli::options;

struct row {
    const char* operation;
    std::size_t size;
    double naive;
    double simd;
    double blocked;
    double full;
};

template <typename Prepare, typename Work>
double best_of(int repetitions, Prepare prepare, Work work) {
    double best = 1e9;
    for (int run = 0; run <= repetitions; ++run) {
        prepare();
        const auto start = clock_type::now();
        work();
        const auto stop = clock_type::now();
        if (run > 0) {
            best = std::min(best, std::chrono::duration<double>(stop - start).count());
        }
    }
    return best;
}

double gflops(double flops, double seconds) {
    return flops / seconds / 1e9;
}

void print_human(const row& result) {
    std::printf(
        "  n=%-5zu naive=%6.2f  simd=%6.2f (%5.1fx)  blocked/1t=%6.2f (%5.1fx)  "
        "full/all=%6.2f (%5.1fx)\n",
        result.size,
        result.naive,
        result.simd,
        result.simd / result.naive,
        result.blocked,
        result.blocked / result.naive,
        result.full,
        result.full / result.naive);
}

void print_csv(const row& result) {
    std::printf("%s,%zu,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n",
                result.operation,
                result.size,
                result.naive,
                result.simd,
                result.blocked,
                result.full,
                result.simd / result.naive,
                result.blocked / result.naive,
                result.full / result.naive);
}

row time_matmul(std::size_t n) {
    const auto a = matrix<double>::random(n, n, 1, {-1.0, 1.0});
    const auto b = matrix<double>::random(n, n, 2, {-1.0, 1.0});
    matrix<double> out(n, n);
    const double flops =
        2.0 * static_cast<double>(n) * static_cast<double>(n) * static_cast<double>(n);

    const auto zero = [&out] { out.fill(0.0); };
    const double naive = best_of(3, zero, [&] { lazuli::bench::baseline::matmul(a, b, out); });
    const double simd = best_of(3, zero, [&] { lazuli::bench::stages::matmul_simd(a, b, out); });
    const double blocked =
        best_of(3, zero, [&] { lazuli::detail::kernel::matmul(a, b, out, options{1}); });
    const double full =
        best_of(3, zero, [&] { lazuli::detail::kernel::matmul(a, b, out, options{0}); });
    return {"matmul",
            n,
            gflops(flops, naive),
            gflops(flops, simd),
            gflops(flops, blocked),
            gflops(flops, full)};
}

row time_lu(std::size_t n) {
    matrix<double> original = matrix<double>::random(n, n, 3, {-1.0, 1.0});
    for (std::size_t index = 0; index < n; ++index) {
        original(index, index) += static_cast<double>(n);
    }
    matrix<double> work(n, n);
    std::vector<std::size_t> pivots(n);
    const double flops =
        2.0 / 3.0 * static_cast<double>(n) * static_cast<double>(n) * static_cast<double>(n);

    const auto copy = [&] { work = original; };
    const double naive =
        best_of(3, copy, [&] { (void)lazuli::bench::stages::lu_naive(work.view(), pivots); });
    const double simd =
        best_of(3, copy, [&] { (void)lazuli::bench::stages::lu_simd(work.view(), pivots); });
    const double blocked =
        best_of(3, copy, [&] { (void)lazuli::nt::lu_in_place(work.view(), options{1}); });
    const double full =
        best_of(3, copy, [&] { (void)lazuli::nt::lu_in_place(work.view(), options{0}); });
    return {"lu",
            n,
            gflops(flops, naive),
            gflops(flops, simd),
            gflops(flops, blocked),
            gflops(flops, full)};
}

row time_cholesky(std::size_t n) {
    const auto b = matrix<double>::random(n, n, 5, {-1.0, 1.0});
    matrix<double> spd = b.transpose() * b;
    for (std::size_t index = 0; index < n; ++index) {
        spd(index, index) += static_cast<double>(n);
    }
    matrix<double> work(n, n);
    const double flops =
        1.0 / 3.0 * static_cast<double>(n) * static_cast<double>(n) * static_cast<double>(n);

    const auto copy = [&] { work = spd; };
    const auto nothing = [] {};
    const double naive =
        best_of(3, copy, [&] { (void)lazuli::bench::stages::cholesky_naive(work.view()); });
    const double simd =
        best_of(3, copy, [&] { (void)lazuli::bench::stages::cholesky_simd(work.view()); });
    const double blocked =
        best_of(3, nothing, [&] { (void)lazuli::nt::cholesky(spd, options{1}); });
    const double full = best_of(3, nothing, [&] { (void)lazuli::nt::cholesky(spd, options{0}); });
    return {"cholesky",
            n,
            gflops(flops, naive),
            gflops(flops, simd),
            gflops(flops, blocked),
            gflops(flops, full)};
}

}  // namespace

int main(int argc, char** argv) {
    const bool csv = argc > 1 && std::string_view{argv[1]} == "--csv";
    if (!lazuli::bench::stages::avx2_available()) {
        std::fprintf(stderr, "this benchmark needs AVX2\n");
        return 1;
    }

    if (csv) {
        std::printf(
            "operation,n,naive_gflops,simd_gflops,blocked_1t_gflops,full_all_gflops,"
            "simd_speedup,blocked_speedup,full_speedup\n");
    } else {
        std::printf("staged speedup, double, best of 3 (GFLOPS)\n");
    }

    if (!csv) {
        std::printf("Matmul\n");
    }
    for (const std::size_t n : {std::size_t{256}, std::size_t{512}, std::size_t{1024}}) {
        const row result = time_matmul(n);
        if (csv) {
            print_csv(result);
        } else {
            print_human(result);
        }
    }
    if (!csv) {
        std::printf("LU\n");
    }
    for (const std::size_t n : {std::size_t{256}, std::size_t{512}, std::size_t{1024}}) {
        const row result = time_lu(n);
        if (csv) {
            print_csv(result);
        } else {
            print_human(result);
        }
    }
    if (!csv) {
        std::printf("Cholesky\n");
    }
    for (const std::size_t n : {std::size_t{256}, std::size_t{512}, std::size_t{1024}}) {
        const row result = time_cholesky(n);
        if (csv) {
            print_csv(result);
        } else {
            print_human(result);
        }
    }
    return 0;
}
