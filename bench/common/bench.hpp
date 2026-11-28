#pragma once

/// @file
/// @brief Shared timing and reporting helpers for the benchmark menu.

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdio>

namespace lazuli::bench {

using clock_type = std::chrono::steady_clock;

/// Sizes used by every operation in the menu.
inline constexpr std::size_t benchmark_sizes[] = {
    std::size_t{256}, std::size_t{512}, std::size_t{1024}};

/// Runs `work` `count + 1` times, discarding the first run and returning the best time.
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

inline double gflops(double flops, double seconds) {
    return flops / seconds / 1e9;
}

inline void print_stage_row(std::size_t n, double naive, double simd, double blocked, double full) {
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

}  // namespace lazuli::bench
