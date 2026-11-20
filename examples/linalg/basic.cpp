// Factorisations: LU, solve, determinant, inverse and Cholesky.
//
//   cmake --build build --target linalg_basic
//   ./build/examples/linalg/basic

#include <lazuli/lazuli.hpp>

#include <cstddef>
#include <cstdio>

namespace {

void print(const char* label, const lazuli::matrix<double>& matrix) {
    std::printf("%s (%zu x %zu)\n", label, matrix.rows(), matrix.cols());
    for (std::size_t row = 0; row < matrix.rows(); ++row) {
        for (std::size_t column = 0; column < matrix.cols(); ++column) {
            std::printf(" %8.4f", matrix(row, column));
        }
        std::printf("\n");
    }
}

}  // namespace

int main() {
    const lazuli::matrix<double> a{{4.0, 3.0}, {6.0, 3.0}};

    // LU with partial pivoting: P A = L U. The pivot list says which row of A ended up where.
    const auto factors = lazuli::lu(a);
    if (!factors) {
        std::printf("lu failed: %s\n", factors.error().message.c_str());
        return 1;
    }
    print("L", factors->l);
    print("U", factors->u);

    // Solve A x = b through the factors.
    const lazuli::vector<double> b{1.0, 1.0};
    const auto x = lazuli::solve(a, b);
    std::printf("solution: %.4f %.4f\n", (*x)[0], (*x)[1]);

    // Determinant and inverse come from the same factorisation.
    std::printf("det: %.4f\n", lazuli::det(a));
    print("inverse", *lazuli::inverse(a));

    // Cholesky needs a symmetric positive-definite matrix and returns the lower factor.
    const lazuli::matrix<double> spd{{4.0, 2.0}, {2.0, 5.0}};
    const auto lower = lazuli::cholesky(spd);
    if (!lower) {
        std::printf("cholesky failed: %s\n", lower.error().message.c_str());
        return 1;
    }
    print("cholesky L", *lower);
    print("L L^T", (*lower) * lower->transpose());

    // Failures are values, not exceptions: a singular matrix says so.
    const lazuli::matrix<double> singular{{1.0, 2.0}, {2.0, 4.0}};
    const auto failed = lazuli::lu(singular);
    std::printf("singular LU: %s\n",
                failed ? "unexpectedly succeeded" : failed.error().message.c_str());

    return 0;
}
