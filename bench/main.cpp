// Interactive benchmark menu.
//
//   cmake --build build/release --target bench
//   ./build/release/bench/bench
//
// The naive references are compiled with -O0 by request. The SIMD stage uses AVX2/FMA with no
// packing, blocking or threads; the library stages are the blocked, threaded implementation.

#include <baseline/matmul.hpp>
#include <lazuli/lazuli.hpp>

#include <naive/linalg.hpp>
#include <stages/linalg.hpp>

#if defined(LAZULI_HAVE_OPENBLAS)
#include <reference/openblas.hpp>
#endif

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

namespace {

using clock_type = std::chrono::steady_clock;
using lazuli::matrix;
using lazuli::options;

int repetitions = 3;
constexpr std::size_t sizes[] = {std::size_t{256}, std::size_t{512}, std::size_t{1024}};

template <typename Prepare, typename Work>
double best_of(int count, Prepare prepare, Work work) {
    double best = 1e9;
    for (int run = 0; run <= count; ++run) {
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

void print_stage_row(std::size_t n, double naive, double simd, double blocked, double full) {
    std::printf(
        "  n=%-5zu naive=%7.2f  simd=%7.2f (%5.1fx)  blocked=%7.2f (%5.1fx)  "
        "full=%7.2f (%5.1fx)\n",
        n,
        naive,
        simd,
        simd / naive,
        blocked,
        blocked / naive,
        full,
        full / naive);
}

void run_matmul() {
    std::printf("\nmatmul, double, best of %d\n", repetitions);
    for (const std::size_t n : sizes) {
        const auto a = matrix<double>::random(n, n, 1, {-1.0, 1.0});
        const auto b = matrix<double>::random(n, n, 2, {-1.0, 1.0});
        matrix<double> out(n, n);
        const double flops =
            2.0 * static_cast<double>(n) * static_cast<double>(n) * static_cast<double>(n);
        const auto zero = [&out] { out.fill(0.0); };
        const double naive =
            best_of(repetitions, zero, [&] { lazuli::bench::baseline::matmul(a, b, out); });
        const double simd =
            best_of(repetitions, zero, [&] { lazuli::bench::stages::matmul_simd(a, b, out); });
        const double blocked = best_of(
            repetitions, zero, [&] { lazuli::detail::kernel::matmul(a, b, out, options{1}); });
        const double full = best_of(
            repetitions, zero, [&] { lazuli::detail::kernel::matmul(a, b, out, options{0}); });
        print_stage_row(n,
                        gflops(flops, naive),
                        gflops(flops, simd),
                        gflops(flops, blocked),
                        gflops(flops, full));
    }
}

void run_lu() {
    std::printf("\nLU, double, best of %d\n", repetitions);
    for (const std::size_t n : sizes) {
        matrix<double> original = matrix<double>::random(n, n, 3, {-1.0, 1.0});
        for (std::size_t index = 0; index < n; ++index) {
            original(index, index) += static_cast<double>(n);
        }
        matrix<double> work(n, n);
        std::vector<std::size_t> pivots(n);
        const double flops =
            2.0 / 3.0 * static_cast<double>(n) * static_cast<double>(n) * static_cast<double>(n);
        const auto copy = [&] { work = original; };
        const double naive = best_of(
            repetitions, copy, [&] { (void)lazuli::bench::naive::lu(work.view(), pivots); });
        const double simd = best_of(
            repetitions, copy, [&] { (void)lazuli::bench::stages::lu_simd(work.view(), pivots); });
        const double blocked = best_of(
            repetitions, copy, [&] { (void)lazuli::nt::lu_in_place(work.view(), options{1}); });
        const double full = best_of(
            repetitions, copy, [&] { (void)lazuli::nt::lu_in_place(work.view(), options{0}); });
        print_stage_row(n,
                        gflops(flops, naive),
                        gflops(flops, simd),
                        gflops(flops, blocked),
                        gflops(flops, full));
    }
}

void run_cholesky() {
    std::printf("\nCholesky, double, best of %d\n", repetitions);
    std::printf(
        "  (library cholesky allocates its result; naive/SIMD factor a preallocated copy)\n");
    for (const std::size_t n : sizes) {
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
            best_of(repetitions, copy, [&] { (void)lazuli::bench::naive::cholesky(work.view()); });
        const double simd = best_of(
            repetitions, copy, [&] { (void)lazuli::bench::stages::cholesky_simd(work.view()); });
        const double blocked =
            best_of(repetitions, nothing, [&] { (void)lazuli::nt::cholesky(spd, options{1}); });
        const double full =
            best_of(repetitions, nothing, [&] { (void)lazuli::nt::cholesky(spd, options{0}); });
        print_stage_row(n,
                        gflops(flops, naive),
                        gflops(flops, simd),
                        gflops(flops, blocked),
                        gflops(flops, full));
    }
}

void run_thread_sweep() {
    std::printf("\nmatmul thread sweep, double, best of %d\n", repetitions);
    for (const std::size_t n : sizes) {
        const auto a = matrix<double>::random(n, n, 7, {-1.0, 1.0});
        const auto b = matrix<double>::random(n, n, 8, {-1.0, 1.0});
        matrix<double> out(n, n);
        const double flops =
            2.0 * static_cast<double>(n) * static_cast<double>(n) * static_cast<double>(n);
        std::printf("  n=%-5zu", n);
        for (const std::size_t threads : {std::size_t{1},
                                          std::size_t{2},
                                          std::size_t{3},
                                          std::size_t{4},
                                          std::size_t{6},
                                          std::size_t{8},
                                          std::size_t{12}}) {
            const double seconds = best_of(
                repetitions,
                [&] { out.fill(0.0); },
                [&] { lazuli::detail::kernel::matmul(a, b, out, options{threads}); });
            std::printf("  t=%zu:%6.2f", threads, gflops(flops, seconds));
        }
        std::printf("\n");
    }
}

void run_openblas() {
#if defined(LAZULI_HAVE_OPENBLAS)
    std::printf("\nOpenBLAS reference, double, best of %d\n", repetitions);
    const int all = static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
    for (const std::size_t n : sizes) {
        const auto a = matrix<double>::random(n, n, 1, {-1.0, 1.0});
        const auto b = matrix<double>::random(n, n, 2, {-1.0, 1.0});
        matrix<double> out(n, n);
        const double flops =
            2.0 * static_cast<double>(n) * static_cast<double>(n) * static_cast<double>(n);
        const auto one = best_of(
            repetitions,
            [&] {
                out.fill(0.0);
                lazuli::bench::reference::set_threads(1);
            },
            [&] { lazuli::bench::reference::matmul(n, n, n, a.data(), b.data(), out.data()); });
        const auto many = best_of(
            repetitions,
            [&] {
                out.fill(0.0);
                lazuli::bench::reference::set_threads(all);
            },
            [&] { lazuli::bench::reference::matmul(n, n, n, a.data(), b.data(), out.data()); });
        std::printf("  n=%-5zu openblas/1t=%7.2f  openblas/%dt=%7.2f\n",
                    n,
                    gflops(flops, one),
                    all,
                    gflops(flops, many));
    }
#else
    std::printf("\nOpenBLAS was not found at configure time\n");
#endif
}

void set_iterations() {
    std::printf("iterations (1-20): ");
    std::fflush(stdout);
    std::string line;
    if (!std::getline(std::cin, line)) {
        return;
    }
    char* end = nullptr;
    const long value = std::strtol(line.c_str(), &end, 10);
    if (end == line.c_str() || *end != '\0' || value < 1 || value > 20) {
        std::printf("invalid iteration count\n");
        return;
    }
    repetitions = static_cast<int>(value);
}

void print_menu() {
    std::printf(
        "\nlazuli benchmarks\n"
        "  1) matmul       naive -> SIMD -> blocked -> full\n"
        "  2) LU           naive -> SIMD -> blocked -> full\n"
        "  3) Cholesky     naive -> SIMD -> blocked -> full\n"
        "  4) thread sweep matmul, 1, 2, 3, 4, 6, 8 and 12 threads\n"
        "  5) OpenBLAS reference\n"
        "  6) iterations   current: %d\n"
        "  7) exit\n"
        "> ",
        repetitions);
    std::fflush(stdout);
}

}  // namespace

int main() {
    if (!lazuli::bench::stages::avx2_available()) {
        std::fprintf(stderr, "this benchmark needs AVX2 for the SIMD stage\n");
        return 1;
    }
    std::printf("naive stages are compiled with -O0; their speedups are not like-for-like.\n");

    std::string line;
    while (true) {
        print_menu();
        if (!std::getline(std::cin, line)) {
            break;
        }
        if (line == "1") {
            run_matmul();
        } else if (line == "2") {
            run_lu();
        } else if (line == "3") {
            run_cholesky();
        } else if (line == "4") {
            run_thread_sweep();
        } else if (line == "5") {
            run_openblas();
        } else if (line == "6") {
            set_iterations();
        } else if (line == "7") {
            break;
        } else {
            std::printf("unknown selection\n");
        }
    }
    return 0;
}
