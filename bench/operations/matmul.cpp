#include <lazuli/lazuli.hpp>

#include <common/bench.hpp>
#include <cstddef>
#include <operations/matmul.hpp>
#include <references/naive/matmul.hpp>
#include <references/simd/matmul.hpp>

namespace lazuli::bench {

using lazuli::detail::matrix;
using lazuli::detail::options;

void run_matmul(int repetitions) {
    std::printf("\nmatmul, double, best of %d\n", repetitions);
    print_stage_header();
    for (const std::size_t n : benchmark_sizes) {
        const auto a = matrix<double>::random(n, n, 1, {-1.0, 1.0});
        const auto b = matrix<double>::random(n, n, 2, {-1.0, 1.0});
        matrix<double> out(n, n);
        const double flops =
            2.0 * static_cast<double>(n) * static_cast<double>(n) * static_cast<double>(n);
        const auto zero = [&out] { out.fill(0.0); };
        const double naive = best_of(repetitions, zero, [&] { naive::matmul(a, b, out); });
        const double simd = best_of(repetitions, zero, [&] { simd::matmul(a, b, out); });
        const double blocked = best_of(
            repetitions, zero, [&] { lazuli::detail::kernel::matmul(a, b, out, options{1}); });
        const double full = best_of(
            repetitions, zero, [&] { lazuli::detail::kernel::matmul(a, b, out, options{0}); });
        print_stage_row(n,
                        gflops(flops, naive),
                        gflops(flops, simd),
                        gflops(flops, blocked),
                        gflops(flops, full));
    }
}

}  // namespace lazuli::bench
