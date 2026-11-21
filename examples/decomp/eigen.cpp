// Eigenvalues of a symmetric matrix by Jacobi rotations.
//
// Jacobi is O(n^3) per sweep and needs several sweeps, so it is meant for modest sizes.
//
//   cmake --build build --target decomp_eigen
//   ./build/examples/decomp/eigen

#include <lazuli/lazuli.hpp>

#include <cmath>
#include <cstddef>
#include <cstdio>

int main() {
    const lazuli::matrix<double> a{{2.0, 1.0, 0.0}, {1.0, 2.0, 0.0}, {0.0, 0.0, 3.0}};

    const lazuli::vector<double> values = lazuli::eigenvalues(a);
    std::printf("eigenvalues: %.4f %.4f %.4f (ascending)\n", values[0], values[1], values[2]);

    // The sum of the eigenvalues is the trace, which needs no reference implementation to check.
    double trace = 0.0;
    for (std::size_t index = 0; index < 3; ++index) {
        trace += a(index, index);
    }
    double total = 0.0;
    for (std::size_t index = 0; index < values.size(); ++index) {
        total += values[index];
    }
    std::printf("sum: %.4f, trace: %.4f\n", total, trace);

    // Symmetry is enforced: by default the failure is a typed exception, and
    // through nt it is a value.
    const lazuli::matrix<double> asymmetric{{1.0, 2.0}, {0.0, 1.0}};
    try {
        (void)lazuli::eigenvalues(asymmetric);
    } catch (const lazuli::not_symmetric& failure) {
        std::printf("throws: %s\n", failure.what());
    }
    const auto failed = lazuli::nt::eigenvalues(asymmetric);
    std::printf("nt: %s\n", failed ? "unexpectedly succeeded" : failed.error().message.c_str());
    return 0;
}
