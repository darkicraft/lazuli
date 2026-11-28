#include <lazuli/lazuli.hpp>

#include <common/bench.hpp>
#include <cstddef>
#include <operations/cholesky.hpp>
#include <references/naive/lu_cholesky.hpp>
#include <references/simd/lu_cholesky.hpp>

namespace lazuli::bench {

using lazuli::detail::matrix;
using lazuli::detail::options;

void run_cholesky(int repetitions) {
    std::printf("\nCholesky, double, best of %d\n", repetitions);
    std::printf(
        "  (library cholesky allocates its result; naive/SIMD factor a preallocated copy)\n");
    print_stage_header();
    for (const std::size_t n : benchmark_sizes) {
        const auto b = matrix<double>::random(n, n, 5, {-1.0, 1.0});
        matrix<double> spd = b.transpose() * b;
        for (std::size_t index = 0; index < n; ++index) {
            spd(index, index) += static_cast<double>(n);
        }
        matrix<double> work(n, n);
        const double flops =
            1.0 / 3.0 * static_cast<double>(n) * static_cast<double>(n) * static_cast<double>(n);
        const auto copy = [&] { work = spd; };
        const auto nothing = [] {};
        const double naive =
            best_of(repetitions, copy, [&] { (void)naive::cholesky(work.view()); });
        const double simd = best_of(repetitions, copy, [&] { (void)simd::cholesky(work.view()); });
        const double blocked =
            best_of(repetitions, nothing, [&] { (void)lazuli::nt::cholesky(spd, options{1}); });
        const double full =
            best_of(repetitions, nothing, [&] { (void)lazuli::nt::cholesky(spd, options{0}); });
        print_stage_row(n,
                        gflops(flops, naive),
                        gflops(flops, simd),
                        gflops(flops, blocked),
                        gflops(flops, full));
    }
}

}  // namespace lazuli::bench
