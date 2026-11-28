#include <lazuli/lazuli.hpp>

#include <cstddef>
#include <functional>
#include <vector>

namespace lazuli::bench {

extern const std::size_t benchmark_sizes[3];
double best_of(int count, const std::function<void()>& prepare, const std::function<void()>& work);
double gflops(double flops, double seconds);
std::vector<std::size_t> thread_counts();

}  // namespace lazuli::bench

namespace lazuli::bench {

using lazuli::detail::matrix;
using lazuli::detail::options;

void run_threads(int repetitions) {
    std::printf("\nmatmul thread sweep, double, best of %d\n", repetitions);
    const auto counts = thread_counts();
    std::printf("  %-5s", "n");
    for (const std::size_t threads : counts) {
        std::printf(" %6zu", threads);
    }
    std::printf("\n");
    for (const std::size_t n : benchmark_sizes) {
        const auto a = matrix<double>::random(n, n, 7, {-1.0, 1.0});
        const auto b = matrix<double>::random(n, n, 8, {-1.0, 1.0});
        matrix<double> out(n, n);
        const double flops =
            2.0 * static_cast<double>(n) * static_cast<double>(n) * static_cast<double>(n);
        std::printf("  %-5zu", n);
        for (const std::size_t threads : counts) {
            const double seconds = best_of(
                repetitions,
                [&] { out.fill(0.0); },
                [&] { lazuli::detail::kernel::matmul(a, b, out, options{threads}); });
            std::printf(" %6.2f", gflops(flops, seconds));
        }
        std::printf("\n");
    }
}

}  // namespace lazuli::bench
