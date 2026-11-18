// What the library throws, and what it reports without throwing.
//
//   cmake --build build --target errors_handling
//   ./build/examples/errors/handling

#include <lazuli/lazuli.hpp>

#include <cstdio>
#include <exception>
#include <stdexcept>

namespace {

void print_spelling(lazuli::error_code code) {
    const auto spelling = lazuli::to_string(code);
    std::printf("  %.*s\n", static_cast<int>(spelling.size()), spelling.data());
}

}  // namespace

int main() {
    // A shape disagreement is caught where the offending expression is written.
    try {
        lazuli::matrix<double> ragged{{1.0, 2.0}, {3.0}};
        (void)ragged;
    } catch (const lazuli::dimension_mismatch& failure) {
        std::printf("dimension_mismatch: %s\n", failure.what());
    }

    // at() validates; operator() does not.
    lazuli::matrix<double> matrix(2, 2);
    try {
        (void)matrix.at(2, 0);
    } catch (const lazuli::out_of_bounds& failure) {
        std::printf("out_of_bounds: %s\n", failure.what());
    }

    // Every failure type derives from a standard exception, so one handler can
    // cover the whole library.
    try {
        (void)matrix.at(0, 9);
    } catch (const std::out_of_range& failure) {
        std::printf("as std::out_of_range: %s\n", failure.what());
    } catch (const std::exception&) {
        std::printf("as std::exception\n");
    }

    // Codes are stable, so a caller can switch on them instead of parsing text.
    std::printf("error codes:\n");
    print_spelling(lazuli::error_code::dimension_mismatch);
    print_spelling(lazuli::error_code::singular);
    print_spelling(lazuli::error_code::not_converged);

    // The same information as a value, for code paths that must not throw.
    lazuli::error report{lazuli::error_code::singular, "matrix is singular"};
    if (report) {
        std::printf("report: %s\n", report.message.c_str());
    }

    return 0;
}
