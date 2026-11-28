#include <lazuli/lazuli.hpp>

#include <algorithm>
#include <cstddef>
#include <functional>
#include <thread>

namespace lazuli::bench {

extern const std::size_t benchmark_sizes[3];
double best_of(int count, const std::function<void()>& prepare, const std::function<void()>& work);
double gflops(double flops, double seconds);

#if defined(LAZULI_HAVE_OPENBLAS)
namespace openblas {
void set_threads(int threads);
void matmul(std::size_t rows,
            std::size_t inner,
            std::size_t cols,
            const double* a,
            const double* b,
            double* c);
}  // namespace openblas
#endif

}  // namespace lazuli::bench

namespace lazuli::bench {

using lazuli::detail::matrix;

void run_openblas(int repetitions) {
#if defined(LAZULI_HAVE_OPENBLAS)
    std::printf("\nOpenBLAS reference, double, best of %d\n", repetitions);
    std::printf("  %-5s %9s %9s\n", "n", "1t", "all");
    const int all = static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
    for (const std::size_t n : benchmark_sizes) {
        const auto a = matrix<double>::random(n, n, 1, {-1.0, 1.0});
        const auto b = matrix<double>::random(n, n, 2, {-1.0, 1.0});
        matrix<double> out(n, n);
        const double flops =
            2.0 * static_cast<double>(n) * static_cast<double>(n) * static_cast<double>(n);
        const auto one = best_of(
            repetitions,
            [&] {
                out.fill(0.0);
                openblas::set_threads(1);
            },
            [&] { openblas::matmul(n, n, n, a.data(), b.data(), out.data()); });
        const auto many = best_of(
            repetitions,
            [&] {
                out.fill(0.0);
                openblas::set_threads(all);
            },
            [&] { openblas::matmul(n, n, n, a.data(), b.data(), out.data()); });
        std::printf("  %-5zu %9.2f %9.2f\n", n, gflops(flops, one), gflops(flops, many));
    }
#else
    (void)repetitions;
    std::printf("\nOpenBLAS was not found at configure time\n");
#endif
}

}  // namespace lazuli::bench
