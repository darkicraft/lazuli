#include <lazuli/lazuli.hpp>

#include <cstddef>
#include <functional>
#include <vector>

namespace lazuli::bench {

extern const std::size_t benchmark_sizes[3];
double best_of(int count, const std::function<void()>& prepare, const std::function<void()>& work);
double gflops(double flops, double seconds);
void print_stage_header();
void print_stage_row(std::size_t n, double naive, double simd, double blocked, double full);

namespace naive {
bool lu(lazuli::detail::matrix_view<double> a, std::vector<std::size_t>& pivots);
}

namespace simd {
bool lu(lazuli::detail::matrix_view<double> a, std::vector<std::size_t>& pivots);
}

}  // namespace lazuli::bench

namespace lazuli::bench {

using lazuli::detail::matrix;
using lazuli::detail::options;

void run_lu(int repetitions) {
    std::printf("\nLU, double, best of %d\n", repetitions);
    print_stage_header();
    for (const std::size_t n : benchmark_sizes) {
        matrix<double> original = matrix<double>::random(n, n, 3, {-1.0, 1.0});
        for (std::size_t index = 0; index < n; ++index) {
            original(index, index) += static_cast<double>(n);
        }
        matrix<double> work(n, n);
        std::vector<std::size_t> pivots(n);
        const double flops =
            2.0 / 3.0 * static_cast<double>(n) * static_cast<double>(n) * static_cast<double>(n);
        const auto copy = [&] { work = original; };
        const double naive =
            best_of(repetitions, copy, [&] { (void)naive::lu(work.view(), pivots); });
        const double simd =
            best_of(repetitions, copy, [&] { (void)simd::lu(work.view(), pivots); });
        const double blocked = best_of(
            repetitions, copy, [&] { (void)lazuli::nt::lu_in_place(work.view(), options{1}); });
        const double full = best_of(
            repetitions, copy, [&] { (void)lazuli::nt::lu_in_place(work.view(), options{0}); });
        print_stage_row(n,
                        gflops(flops, naive),
                        gflops(flops, simd),
                        gflops(flops, blocked),
                        gflops(flops, full));
    }
}

}  // namespace lazuli::bench
