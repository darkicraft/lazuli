#include <lazuli/lazuli.hpp>

#include <common/bench.hpp>
#include <cstddef>
#include <operations/threads.hpp>

namespace lazuli::bench {

using lazuli::detail::matrix;
using lazuli::detail::options;

void run_threads(int repetitions) {
    std::printf("\nmatmul thread sweep, double, best of %d\n", repetitions);
    for (const std::size_t n : benchmark_sizes) {
        const auto a = matrix<double>::random(n, n, 7, {-1.0, 1.0});
        const auto b = matrix<double>::random(n, n, 8, {-1.0, 1.0});
        matrix<double> out(n, n);
        const double flops =
            2.0 * static_cast<double>(n) * static_cast<double>(n) * static_cast<double>(n);
        std::printf("  n=%-5zu", n);
        for (const std::size_t threads : {std::size_t{1},
                                          std::size_t{2},
                                          std::size_t{3},
                                          std::size_t{4},
                                          std::size_t{6},
                                          std::size_t{8},
                                          std::size_t{12}}) {
            const double seconds = best_of(
                repetitions,
                [&] { out.fill(0.0); },
                [&] { lazuli::detail::kernel::matmul(a, b, out, options{threads}); });
            std::printf("  t=%zu:%6.2f", threads, gflops(flops, seconds));
        }
        std::printf("\n");
    }
}

}  // namespace lazuli::bench
