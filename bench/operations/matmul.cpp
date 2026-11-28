#include <lazuli/lazuli.hpp>

#include <cstddef>
#include <functional>

namespace lazuli::bench {

extern const std::size_t benchmark_sizes[3];
double best_of(int count, const std::function<void()>& prepare, const std::function<void()>& work);
double gflops(double flops, double seconds);
void print_stage_header();
void print_stage_row(std::size_t n, double naive, double simd, double blocked, double full);

namespace naive {
void matmul(const lazuli::detail::matrix_view<const double>& a,
            const lazuli::detail::matrix_view<const double>& b,
            lazuli::detail::matrix_view<double> out);
}

namespace simd {
void matmul(lazuli::detail::matrix_view<const double> a,
            lazuli::detail::matrix_view<const double> b,
            lazuli::detail::matrix_view<double> out);
}

}  // namespace lazuli::bench

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
