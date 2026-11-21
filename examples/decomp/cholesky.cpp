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

    const lazuli::matrix<double> lower = lazuli::cholesky(spd);
    std::printf("L[0,0] = %.4f, L[7,7] = %.4f\n", lower(0, 0), lower(7, 7));

    // L L^T must rebuild the matrix.
    const lazuli::matrix<double> rebuilt = lower * lower.transpose();
    double worst = 0.0;
    for (std::size_t row = 0; row < 8; ++row) {
        for (std::size_t column = 0; column < 8; ++column) {
            worst = std::max(worst, std::abs(rebuilt(row, column) - spd(row, column)));
        }
    }
    std::printf("max |L L^T - A|: %.3e\n", worst);

    // A matrix that is not positive definite throws by default, and comes back
    // as a value through the no-throw facade.
    const lazuli::matrix<double> indefinite{{1.0, 2.0}, {2.0, 1.0}};
    try {
        (void)lazuli::cholesky(indefinite);
    } catch (const lazuli::not_positive_definite& failure) {
        std::printf("throws: %s\n", failure.what());
    }
    const auto failed = lazuli::nt::cholesky(indefinite);
    std::printf("nt: %s\n", failed ? "unexpectedly succeeded" : failed.error().message.c_str());
    return 0;
}
