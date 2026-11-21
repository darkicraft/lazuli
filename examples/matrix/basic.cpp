// Constructing matrices and touching their elements.
//
//   cmake --build build --target matrix_basic
//   ./build/examples/matrix/basic

#include <lazuli/lazuli.hpp>

#include <cstdio>
#include <format>
#include <string>

namespace {

void print(const char* label, const lazuli::matrix<double>& matrix) {
    std::printf("%s (%zu x %zu)\n", label, matrix.rows(), matrix.cols());
    for (std::size_t row = 0; row < matrix.rows(); ++row) {
        for (std::size_t column = 0; column < matrix.cols(); ++column) {
            std::printf(" %7.2f", matrix(row, column));
        }
        std::printf("\n");
    }
}

}  // namespace

int main() {
    // The ordinary constructor zero-fills.
    lazuli::matrix<double> zeros(2, 3);
    print("zeros", zeros);

    // Row-major initialiser lists, one inner list per row.
    lazuli::matrix<double> values{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};
    print("values", values);

    // operator() is unchecked; at() validates and throws.
    values(0, 0) = 10.0;
    values.at(1, 2) = 60.0;
    print("after writes", values);

    // Copies own their storage, so writing to the copy leaves the original alone.
    lazuli::matrix<double> copy{values};
    copy(0, 1) = -1.0;
    std::printf("original[0,1] = %.2f, copy[0,1] = %.2f\n", values(0, 1), copy(0, 1));

    // uninitialized() skips the zero fill when every element is about to be
    // overwritten anyway: the hot-path constructor.
    auto scratch = lazuli::matrix<double>::uninitialized(2, 2);
    for (std::size_t row = 0; row < scratch.rows(); ++row) {
        for (std::size_t column = 0; column < scratch.cols(); ++column) {
            scratch(row, column) = static_cast<double>(row * scratch.cols() + column);
        }
    }
    print("scratch", scratch);

    // Factories and comparisons.
    const auto identity = lazuli::matrix<double>::identity(3);
    print("identity(3)", identity);
    std::printf("norm of identity(3): %.4f\n", identity.norm());
    std::printf("values approx_equal itself: %s\n",
                values.approx_equal(values, 0.0) ? "yes" : "no");
    std::printf("identity approx_equal values: %s\n",
                identity.approx_equal(values, 0.0) ? "yes" : "no");

    // std::format works without a stream manipulator in sight. The default is
    // compact; `#` puts one row per line and the rest of the spec is handed to
    // the scalar formatter.
    std::printf("compact:   %s\n", std::format("{}", values).c_str());
    std::printf("one row:\n%s\n", std::format("{:#}", values).c_str());
    std::printf("precision: %s\n", std::format("{:.1f}", values).c_str());
    std::printf("empty:     %s\n", std::format("{}", lazuli::matrix<double>(0, 3)).c_str());

    // Storage is row-major and contiguous, so the strides are plain numbers.
    std::printf("row stride %zu, column stride %zu, %zu elements\n",
                values.row_stride(),
                values.col_stride(),
                values.size());

    return 0;
}
