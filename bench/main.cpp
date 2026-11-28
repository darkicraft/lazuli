// Interactive benchmark menu.
//
//   cmake --build build/release --target bench
//   ./build/release/bench/bench
//
// Each operation lives in operations/<name>.cpp; the reference implementations live under
// references/. This file only owns the menu and the iteration setting.

#include <cctype>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>

namespace lazuli::bench {

std::size_t hardware_threads();
std::string thread_count_label();
void run_cholesky(int repetitions);
void run_lu(int repetitions);
void run_matmul(int repetitions);
void run_openblas(int repetitions);
void run_threads(int repetitions);

namespace simd {
bool available() noexcept;
}

}  // namespace lazuli::bench

namespace {

int repetitions = 3;

std::string normalize(const std::string& line) {
    const auto first = line.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) {
        return {};
    }
    const auto last = line.find_last_not_of(" \t\r\n");
    std::string text = line.substr(first, last - first + 1);
    for (char& character : text) {
        character = static_cast<char>(std::tolower(static_cast<unsigned char>(character)));
    }
    return text;
}

void set_iterations() {
    std::printf("iterations (1-20): ");
    std::fflush(stdout);
    std::string line;
    if (!std::getline(std::cin, line)) {
        return;
    }
    const std::string text = normalize(line);
    char* end = nullptr;
    const long value = std::strtol(text.c_str(), &end, 10);
    if (end == text.c_str() || *end != '\0' || value < 1 || value > 20) {
        std::printf("invalid iteration count\n");
        return;
    }
    repetitions = static_cast<int>(value);
    std::printf("iterations = %d\n", repetitions);
}

void print_full_menu() {
    const std::string threads = lazuli::bench::thread_count_label();
    std::printf("\nlazuli benchmark | AVX2 | %zu hardware threads | %d iterations\n\n",
                lazuli::bench::hardware_threads(),
                repetitions);
    std::printf("  m  matmul       naive / SIMD / blocked 1t / full all\n");
    std::printf("  l  LU           naive / SIMD / blocked 1t / full all\n");
    std::printf("  c  Cholesky     naive / SIMD / blocked 1t / full all\n");
    std::printf("  t  thread sweep matmul at %s threads\n", threads.c_str());
    std::printf("  o  OpenBLAS     reference, if built\n");
    std::printf("  a  all staged   matmul + LU + Cholesky + thread sweep\n");
    std::printf("  i  iterations   set repetitions (current: %d)\n", repetitions);
    std::printf("  ?  help\n");
    std::printf("  q  quit\n");
}

void print_footer() {
    std::printf(
        "\n"
        "[m]atmul [l]u [c]holesky [t]hreads [o]penblas [a]ll [i]terations "
        "[?]help [q]uit\n");
}

void print_prompt() {
    std::printf("\nbench [%d]> ", repetitions);
    std::fflush(stdout);
}

}  // namespace

int main() {
    if (!lazuli::bench::simd::available()) {
        std::fprintf(stderr, "this benchmark needs AVX2 for the SIMD stage\n");
        return 1;
    }

    print_full_menu();

    std::string line;
    while (true) {
        print_prompt();
        if (!std::getline(std::cin, line)) {
            std::printf("\n");
            break;
        }
        const std::string command = normalize(line);
        if (command.empty()) {
            continue;
        }

        bool ran = false;
        if (command == "m" || command == "matmul" || command == "1") {
            lazuli::bench::run_matmul(repetitions);
            ran = true;
        } else if (command == "l" || command == "lu" || command == "2") {
            lazuli::bench::run_lu(repetitions);
            ran = true;
        } else if (command == "c" || command == "cholesky" || command == "3") {
            lazuli::bench::run_cholesky(repetitions);
            ran = true;
        } else if (command == "t" || command == "threads" || command == "thread" ||
                   command == "sweep" || command == "4") {
            lazuli::bench::run_threads(repetitions);
            ran = true;
        } else if (command == "o" || command == "openblas" || command == "5") {
            lazuli::bench::run_openblas(repetitions);
            ran = true;
        } else if (command == "a" || command == "all") {
            lazuli::bench::run_matmul(repetitions);
            lazuli::bench::run_lu(repetitions);
            lazuli::bench::run_cholesky(repetitions);
            lazuli::bench::run_threads(repetitions);
            ran = true;
        } else if (command == "i" || command == "iterations" || command == "iter" ||
                   command == "6") {
            set_iterations();
            ran = true;
        } else if (command == "?" || command == "h" || command == "help") {
            print_full_menu();
        } else if (command == "q" || command == "quit" || command == "exit" || command == "7") {
            std::printf("bye\n");
            break;
        } else {
            std::printf("unknown selection; ? for help\n");
        }
        if (ran) {
            print_footer();
        }
    }
    return 0;
}
