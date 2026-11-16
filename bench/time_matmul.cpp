// Times the frozen baseline matrix multiplication.
//
//   cmake --build build --target bench_time_matmul
//   ./build/bench/time_matmul a.lzm b.lzm --reps 5 --output c.lzm
//
// Operands come from files on purpose: nothing on the timed path may be visible to the compiler
// as a constant.

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <string>
#include <string_view>
#include <vector>

#include <lazuli/lazuli.hpp>

#include "baseline/matmul.hpp"

namespace {

constexpr std::string_view usage =
    "usage: time_matmul <a.lzm> <b.lzm> [--reps N] [--output C.lzm]\n";

[[nodiscard]] bool parse_positive(std::string_view text, int& value) {
    try {
        std::size_t consumed = 0;
        const long parsed = std::stol(std::string{text}, &consumed);
        if (consumed != text.size() || parsed <= 0 || parsed > 1000) {
            return false;
        }
        value = static_cast<int>(parsed);
        return true;
    } catch (...) {
        return false;
    }
}

}  // namespace

int main(int argc, char** argv) {
    std::string a_path;
    std::string b_path;
    std::string output_path;
    int repetitions = 5;

    for (int i = 1; i < argc; ++i) {
        const std::string_view argument{argv[i]};
        if (argument == "--reps" && i + 1 < argc) {
            if (!parse_positive(argv[++i], repetitions)) {
                std::fputs(usage.data(), stderr);
                return 2;
            }
        } else if (argument == "--output" && i + 1 < argc) {
            output_path = argv[++i];
        } else if (a_path.empty()) {
            a_path = argument;
        } else if (b_path.empty()) {
            b_path = argument;
        } else {
            std::fputs(usage.data(), stderr);
            return 2;
        }
    }
    if (a_path.empty() || b_path.empty()) {
        std::fputs(usage.data(), stderr);
        return 2;
    }

    const auto loaded_a = lazuli::io::load<double>(a_path);
    if (!loaded_a) {
        std::fprintf(
            stderr, "cannot load %s: %s\n", a_path.c_str(), loaded_a.error().message.c_str());
        return 1;
    }
    const auto loaded_b = lazuli::io::load<double>(b_path);
    if (!loaded_b) {
        std::fprintf(
            stderr, "cannot load %s: %s\n", b_path.c_str(), loaded_b.error().message.c_str());
        return 1;
    }

    const lazuli::matrix<double>& a = *loaded_a;
    const lazuli::matrix<double>& b = *loaded_b;
    if (a.cols() != b.rows()) {
        std::fprintf(stderr,
                     "shape mismatch: %zu x %zu times %zu x %zu\n",
                     a.rows(),
                     a.cols(),
                     b.rows(),
                     b.cols());
        return 1;
    }

    lazuli::matrix<double> out(a.rows(), b.cols());
    std::vector<double> timings;
    timings.reserve(static_cast<std::size_t>(repetitions) + 1);

    for (int run = 0; run <= repetitions; ++run) {
        out.fill(0.0);
        const auto start = std::chrono::steady_clock::now();
        lazuli::bench::baseline::matmul(a, b, out);
        const auto stop = std::chrono::steady_clock::now();
        // The warm-up run is discarded: this laptop throttles, and a cold run is not the same
        // machine as a warm one.
        if (run > 0) {
            timings.push_back(std::chrono::duration<double>(stop - start).count());
        }
    }

    std::sort(timings.begin(), timings.end());
    const double median = timings[timings.size() / 2];

    double checksum = 0.0;
    for (std::size_t row = 0; row < out.rows(); ++row) {
        for (std::size_t column = 0; column < out.cols(); ++column) {
            checksum += out(row, column);
        }
    }

    const double operations = 2.0 * static_cast<double>(a.rows()) * static_cast<double>(a.cols()) *
                              static_cast<double>(b.cols());
    const double gflops = operations / median / 1e9;

    std::printf("size=%zux%zux%zu reps=%d median_ms=%.3f gflops=%.3f checksum=%.9e\n",
                a.rows(),
                a.cols(),
                b.cols(),
                repetitions,
                median * 1e3,
                gflops,
                checksum);

    if (!output_path.empty()) {
        const auto written = lazuli::io::save(output_path, out);
        if (!written) {
            std::fprintf(stderr,
                         "cannot write %s: %s\n",
                         output_path.c_str(),
                         written.error().message.c_str());
            return 1;
        }
    }
    return 0;
}
