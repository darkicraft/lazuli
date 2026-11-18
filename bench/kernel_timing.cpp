// Times the baseline and the library kernel in-process, with no file I/O in the loop.
//
//   cmake --build build/release --target bench_kernel_timing
//   ./build/release/bench/kernel_timing
//
// This is where the numbers in bench/README.md come from. It reports the best of several
// repetitions rather than the median, because the interesting question is "how fast can this
// kernel go", not "what does this laptop do on average".

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdio>

#include <lazuli/kernel/matmul.hpp>
#include <lazuli/lazuli.hpp>

#include "baseline/matmul.hpp"

namespace {

using clock_type = std::chrono::steady_clock;

template <typename T>
double time_call(const lazuli::matrix<T>& a,
                 const lazuli::matrix<T>& b,
                 lazuli::matrix<T>& out,
                 int repetitions,
                 bool kernel) {
    double best = 1e9;
    for (int run = 0; run <= repetitions; ++run) {
        out.fill(T{});
        const auto start = clock_type::now();
        if (kernel) {
            lazuli::detail::kernel::matmul(a, b, out);
        } else if constexpr (std::is_same_v<T, double>) {
            lazuli::bench::baseline::matmul(a, b, out);
        } else {
            // The frozen baseline is double only; the float comparison is against the scalar
            // loop below, which is the same algorithm.
            for (std::size_t row = 0; row < a.rows(); ++row) {
                for (std::size_t inner = 0; inner < a.cols(); ++inner) {
                    const T value = a(row, inner);
                    for (std::size_t column = 0; column < b.cols(); ++column) {
                        out(row, column) += value * b(inner, column);
                    }
                }
            }
        }
        const auto stop = clock_type::now();
        if (run > 0) {
            best = std::min(best, std::chrono::duration<double>(stop - start).count());
        }
    }
    return best;
}

template <typename T>
void report(const char* label) {
    std::printf("%s\n", label);
    for (std::size_t n :
         {std::size_t{128}, std::size_t{256}, std::size_t{512}, std::size_t{1024}}) {
        const auto a = lazuli::matrix<T>::random(n, n, 1, {T{-1}, T{1}});
        const auto b = lazuli::matrix<T>::random(n, n, 2, {T{-1}, T{1}});
        lazuli::matrix<T> out(n, n);

        const double flops =
            2.0 * static_cast<double>(n) * static_cast<double>(n) * static_cast<double>(n);
        const double scalar_seconds = time_call(a, b, out, 3, false);
        const double kernel_seconds = time_call(a, b, out, 5, true);
        std::printf("  n=%-5zu baseline=%6.2f gflops  kernel=%6.2f gflops  speedup=%.1fx\n",
                    n,
                    flops / scalar_seconds / 1e9,
                    flops / kernel_seconds / 1e9,
                    scalar_seconds / kernel_seconds);
    }
}

}  // namespace

int main() {
    report<double>("double");
    report<float>("float");
    return 0;
}
