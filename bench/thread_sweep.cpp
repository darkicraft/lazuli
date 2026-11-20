// Thread-count sweep for the library kernel.
//
//   cmake --build build/release --target bench_thread_sweep
//   ./build/release/bench/thread_sweep
//
// Reports GFLOPS per thread count and the speedup over one thread. The machine is a 6-core,
// 12-thread 15 W part, so expect the curve to flatten at 6 and to lose ground past it.

#include <lazuli/kernel/matmul.hpp>
#include <lazuli/lazuli.hpp>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <vector>

namespace {

using clock_type = std::chrono::steady_clock;

double time_once(const lazuli::matrix<double>& a,
                 const lazuli::matrix<double>& b,
                 lazuli::matrix<double>& out,
                 std::size_t threads,
                 int repetitions) {
    double best = 1e9;
    for (int run = 0; run <= repetitions; ++run) {
        out.fill(0.0);
        const auto start = clock_type::now();
        lazuli::detail::kernel::matmul(a, b, out, lazuli::options{threads});
        const auto stop = clock_type::now();
        if (run > 0) {
            best = std::min(best, std::chrono::duration<double>(stop - start).count());
        }
    }
    return best;
}

}  // namespace

int main() {
    const std::vector<std::size_t> thread_counts{1, 2, 3, 4, 6, 8, 12};

    for (std::size_t n : {std::size_t{256}, std::size_t{512}, std::size_t{1024}}) {
        const auto a = lazuli::matrix<double>::random(n, n, 1, {-1.0, 1.0});
        const auto b = lazuli::matrix<double>::random(n, n, 2, {-1.0, 1.0});
        lazuli::matrix<double> out(n, n);
        const double flops =
            2.0 * static_cast<double>(n) * static_cast<double>(n) * static_cast<double>(n);

        double single = 0.0;
        std::printf("n=%zu\n", n);
        for (std::size_t threads : thread_counts) {
            const double seconds = time_once(a, b, out, threads, 5);
            const double gflops = flops / seconds / 1e9;
            if (threads == 1) {
                single = seconds;
            }
            std::printf(
                "  threads=%-2zu gflops=%6.2f speedup=%.2fx\n", threads, gflops, single / seconds);
        }
    }
    return 0;
}
