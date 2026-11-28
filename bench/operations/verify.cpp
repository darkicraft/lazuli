#include <lazuli/lazuli.hpp>

#include <algorithm>
#include <cmath>
#include <common/bench.hpp>
#include <cstddef>
#include <cstdio>
#include <operations/verify.hpp>
#include <utility>

namespace lazuli::bench {

using lazuli::detail::matrix;
using lazuli::detail::vector;

namespace {

/// Jacobi is O(n^3) per sweep and QR is scalar, so verification stops well
/// below the staged matmul sizes. At 256 the pair takes roughly a second.
inline constexpr std::size_t verify_sizes[] = {std::size_t{64}, std::size_t{128}, std::size_t{256}};

double max_abs_difference(const matrix<double>& left, const matrix<double>& right) {
    double worst = 0.0;
    for (std::size_t row = 0; row < left.rows(); ++row) {
        for (std::size_t column = 0; column < left.cols(); ++column) {
            worst = std::max(worst, std::abs(left(row, column) - right(row, column)));
        }
    }
    return worst;
}

double squared_frobenius(const matrix<double>& a) {
    double total = 0.0;
    for (std::size_t row = 0; row < a.rows(); ++row) {
        for (std::size_t column = 0; column < a.cols(); ++column) {
            total += a(row, column) * a(row, column);
        }
    }
    return total;
}

}  // namespace

void run_verify(int repetitions) {
    std::printf("\nverify, double, best of %d\n", repetitions);

    std::printf("\n  QR, A = Q R with Q orthogonal\n");
    std::printf("  %-5s %9s %9s %12s %12s\n", "n", "seconds", "GFLOPS", "|Q^T Q-I|", "|Q R-A|");
    for (const std::size_t n : verify_sizes) {
        const matrix<double> a = matrix<double>::random(n, n, 71, {-1.0, 1.0});
        lazuli::detail::decomp::qr_result<double> factors;
        const double seconds = best_of(
            repetitions,
            [] {},
            [&] {
                auto result = lazuli::nt::qr(a);
                if (result) {
                    factors = std::move(*result);
                }
            });

        const matrix<double> identity = matrix<double>::identity(n);
        const matrix<double> gram = factors.q.transpose() * factors.q;
        const matrix<double> rebuilt = factors.q * factors.r;
        const double flops =
            4.0 / 3.0 * static_cast<double>(n) * static_cast<double>(n) * static_cast<double>(n);
        std::printf("  %-5zu %9.3f %9.2f %12.3e %12.3e\n",
                    n,
                    seconds,
                    gflops(flops, seconds),
                    max_abs_difference(gram, identity),
                    max_abs_difference(rebuilt, a));
    }

    std::printf("\n  Jacobi eigenvalues of a symmetric matrix\n");
    std::printf("  %-5s %9s %12s %14s\n", "n", "seconds", "|sum-tr A|", "|sum^2-||A||_F^2|");
    for (const std::size_t n : verify_sizes) {
        const matrix<double> random = matrix<double>::random(n, n, 73, {-1.0, 1.0});
        const matrix<double> a = random + random.transpose();
        vector<double> values;
        const double seconds = best_of(
            repetitions,
            [] {},
            [&] {
                auto result = lazuli::nt::eigenvalues(a);
                if (result) {
                    values = std::move(*result);
                }
            });

        double trace = 0.0;
        for (std::size_t index = 0; index < n; ++index) {
            trace += a(index, index);
        }
        double value_sum = 0.0;
        double value_squares = 0.0;
        for (std::size_t index = 0; index < values.size(); ++index) {
            value_sum += values[index];
            value_squares += values[index] * values[index];
        }
        std::printf("  %-5zu %9.3f %12.3e %14.3e\n",
                    n,
                    seconds,
                    std::abs(value_sum - trace),
                    std::abs(value_squares - squared_frobenius(a)));
    }
}

}  // namespace lazuli::bench
