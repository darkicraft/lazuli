// Cholesky factorisation of a symmetric positive-definite matrix.
//
//   cmake --build build --target decomp_cholesky
//   ./build/examples/decomp/cholesky

#include <lazuli/lazuli.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>

int main() {
    // Build an SPD matrix: B^T B plus a multiple of the identity.
    const auto b = lazuli::matrix<double>::random(8, 8, 13, {-1.0, 1.0});
    lazuli::matrix<double> spd = b.transpose() * b;
    for (std::size_t index = 0; index < 8; ++index) {
        spd(index, index) += 8.0;
    }

    const auto lower = lazuli::cholesky(spd);
    if (!lower) {
        std::printf("cholesky failed: %s\n", lower.error().message.c_str());
        return 1;
    }
    std::printf("L[0,0] = %.4f, L[7,7] = %.4f\n", (*lower)(0, 0), (*lower)(7, 7));

    // L L^T must rebuild the matrix.
    const lazuli::matrix<double> rebuilt = (*lower) * lower->transpose();
    double worst = 0.0;
    for (std::size_t row = 0; row < 8; ++row) {
        for (std::size_t column = 0; column < 8; ++column) {
            worst = std::max(worst, std::abs(rebuilt(row, column) - spd(row, column)));
        }
    }
    std::printf("max |L L^T - A|: %.3e\n", worst);

    // A matrix that is not positive definite is an error value, not an exception.
    const lazuli::matrix<double> indefinite{{1.0, 2.0}, {2.0, 1.0}};
    const auto failed = lazuli::cholesky(indefinite);
    std::printf("indefinite: %s\n",
                failed ? "unexpectedly succeeded" : failed.error().message.c_str());
    return 0;
}
