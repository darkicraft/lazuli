// Shared timing and reporting helpers for the benchmark menu.
//
// The benchmark is a demo, not a library, so it has no internal headers. Every
// other translation unit declares the handful of helpers it uses.

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <functional>
#include <string>
#include <thread>
#include <vector>

namespace lazuli::bench {

using clock_type = std::chrono::steady_clock;

/// Sizes used by every operation in the menu.
extern const std::size_t benchmark_sizes[3] = {
    std::size_t{256}, std::size_t{512}, std::size_t{1024}};

/// Runs `prepare` then `work` `count + 1` times, discarding the first run.
double best_of(int count, const std::function<void()>& prepare, const std::function<void()>& work) {
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

std::size_t hardware_threads() {
    return std::max(1u, std::thread::hardware_concurrency());
}

/// Powers of two up to the hardware count, half the hardware count when it is at least four,
/// and the hardware count itself. On 12 hardware threads this is 1, 2, 4, 6, 8, 12.
std::vector<std::size_t> thread_counts() {
    const std::size_t hardware = hardware_threads();
    std::vector<std::size_t> counts{1};
    for (std::size_t value = 2; value <= hardware; value *= 2) {
        counts.push_back(value);
    }
    const std::size_t half = hardware / 2;
    if (half >= 4) {
        counts.push_back(half);
    }
    if (hardware > 1) {
        counts.push_back(hardware);
    }
    std::sort(counts.begin(), counts.end());
    counts.erase(std::unique(counts.begin(), counts.end()), counts.end());
    return counts;
}

std::string thread_count_label() {
    std::string label;
    for (const std::size_t count : thread_counts()) {
        if (!label.empty()) {
            label += ", ";
        }
        label += std::to_string(count);
    }
    return label;
}

void print_stage_header() {
    std::printf("  %-5s %9s %9s %10s %11s %10s %11s %10s\n",
                "n",
                "naive",
                "SIMD",
                "vs naive",
                "blocked/1t",
                "vs naive",
                "full/all",
                "vs naive");
}

void print_stage_row(std::size_t n, double naive, double simd, double blocked, double full) {
    std::printf("  %-5zu %9.2f %9.2f %9.1fx %11.2f %9.1fx %11.2f %9.1fx\n",
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
