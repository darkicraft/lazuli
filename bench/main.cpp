// Interactive benchmark menu.
//
//   cmake --build build/release --target bench
//   ./build/release/bench/bench
//
// Each operation lives in operations/<name>.cpp; the reference implementations live under
// references/. This file only owns the menu and the iteration setting.

#include <common/bench.hpp>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <operations/cholesky.hpp>
#include <operations/lu.hpp>
#include <operations/matmul.hpp>
#include <operations/openblas.hpp>
#include <operations/threads.hpp>
#include <references/simd/matmul.hpp>
#include <string>

namespace {

int repetitions = 3;

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
    const std::string threads = lazuli::bench::thread_count_label();
    std::printf("\nlazuli benchmark | AVX2 | %zu hardware threads | %d iterations\n",
                lazuli::bench::hardware_threads(),
                repetitions);
    std::printf("  1) matmul       naive -> SIMD -> blocked -> full\n");
    std::printf("  2) LU           naive -> SIMD -> blocked -> full\n");
    std::printf("  3) Cholesky     naive -> SIMD -> blocked -> full\n");
    std::printf("  4) thread sweep matmul at %s threads\n", threads.c_str());
    std::printf("  5) OpenBLAS reference\n");
    std::printf("  6) iterations   current: %d\n", repetitions);
    std::printf("  7) exit\n> ");
    std::fflush(stdout);
}

}  // namespace

int main() {
    if (!lazuli::bench::simd::available()) {
        std::fprintf(stderr, "this benchmark needs AVX2 for the SIMD stage\n");
        return 1;
    }

    std::string line;
    while (true) {
        print_menu();
        if (!std::getline(std::cin, line)) {
            break;
        }
        if (line == "1") {
            lazuli::bench::run_matmul(repetitions);
        } else if (line == "2") {
            lazuli::bench::run_lu(repetitions);
        } else if (line == "3") {
            lazuli::bench::run_cholesky(repetitions);
        } else if (line == "4") {
            lazuli::bench::run_threads(repetitions);
        } else if (line == "5") {
            lazuli::bench::run_openblas(repetitions);
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
