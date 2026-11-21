// Times OpenBLAS, as the reference column in the report.
//
//   cmake --build build/release --target bench_openblas_timing
//   ./build/release/bench/openblas_timing
//
// This is deliberately a separate process from `kernel_timing`: OpenMP and OpenBLAS both keep
// worker pools warm, and timing them back to back measures the two runtimes fighting over the
// same cores rather than either library. Run the two programs, then compare the tables.

#include <lazuli/lazuli.hpp>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <reference/openblas.hpp>
#include <thread>

namespace {

using clock_type = std::chrono::steady_clock;

template <typename T>
double time_call(const lazuli::matrix<T>& a,
                 const lazuli::matrix<T>& b,
                 lazuli::matrix<T>& out,
                 int repetitions,
                 int threads) {
    double best = 1e9;
    lazuli::bench::reference::set_threads(threads);
    for (int run = 0; run <= repetitions; ++run) {
        out.fill(T{});
        const auto start = clock_type::now();
        lazuli::bench::reference::matmul(
            a.rows(), a.cols(), b.cols(), a.data(), b.data(), out.data());
        const auto stop = clock_type::now();
        if (run > 0) {
            best = std::min(best, std::chrono::duration<double>(stop - start).count());
        }
    }
    return best;
}

template <typename T>
void report(const char* label, int all) {
    std::printf("%s\n", label);
    for (std::size_t n :
         {std::size_t{128}, std::size_t{256}, std::size_t{512}, std::size_t{1024}}) {
        const auto a = lazuli::matrix<T>::random(n, n, 1, {T{-1}, T{1}});
        const auto b = lazuli::matrix<T>::random(n, n, 2, {T{-1}, T{1}});
        lazuli::matrix<T> out(n, n);

        const double flops =
            2.0 * static_cast<double>(n) * static_cast<double>(n) * static_cast<double>(n);
        const double one_thread = time_call(a, b, out, 9, 1);
        const double all_threads = time_call(a, b, out, 9, all);
        std::printf("  n=%-5zu blas/1t=%6.2f  blas/all=%6.2f (%4.1fx)\n",
                    n,
                    flops / one_thread / 1e9,
                    flops / all_threads / 1e9,
                    one_thread / all_threads);
    }
}

}  // namespace

int main(int argc, char** argv) {
    int all = static_cast<int>(std::max(1U, std::thread::hardware_concurrency()));
    if (argc > 1) {
        all = std::atoi(argv[1]);
    }
    if (all < 1) {
        all = 1;
    }

    std::printf("OpenBLAS, %d processors detected, %d threads requested\n",
                lazuli::bench::reference::core_count(),
                all);
    report<double>("double", all);
    report<float>("float", all);
    return 0;
}
