// Householder QR: A = Q R, with Q orthogonal and R upper triangular.
//
//   cmake --build build --target decomp_qr
//   ./build/examples/decomp/qr

#include <lazuli/lazuli.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>

int main() {
    const lazuli::matrix<double> a{{4.0, 3.0}, {6.0, 3.0}};

    const auto factors = lazuli::qr(a);
    if (!factors) {
        std::printf("qr failed: %s\n", factors.error().message.c_str());
        return 1;
    }
    std::printf("Q is %zu x %zu, R is %zu x %zu\n",
                factors->q.rows(),
                factors->q.cols(),
                factors->r.rows(),
                factors->r.cols());
    std::printf("R[1,0] = %.3e (upper triangular means this is zero)\n", factors->r(1, 0));

    // Orthogonality, and reconstruction. The signs of Q and R are free, so only magnitudes and
    // products are checked.
    lazuli::matrix<double> identity(2, 2);
    identity(0, 0) = 1.0;
    identity(1, 1) = 1.0;
    const lazuli::matrix<double> orthogonality = factors->q.transpose() * factors->q;
    const lazuli::matrix<double> rebuilt = factors->q * factors->r;

    double worst_orthogonality = 0.0;
    double worst_reconstruction = 0.0;
    for (std::size_t row = 0; row < 2; ++row) {
        for (std::size_t column = 0; column < 2; ++column) {
            worst_orthogonality = std::max(
                worst_orthogonality, std::abs(orthogonality(row, column) - identity(row, column)));
            worst_reconstruction =
                std::max(worst_reconstruction, std::abs(rebuilt(row, column) - a(row, column)));
        }
    }
    std::printf(
        "max |Q^T Q - I|: %.3e, max |Q R - A|: %.3e\n", worst_orthogonality, worst_reconstruction);
    return 0;
}
