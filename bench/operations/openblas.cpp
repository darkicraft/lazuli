#include <lazuli/lazuli.hpp>

#include <common/bench.hpp>
#include <operations/openblas.hpp>

#if defined(LAZULI_HAVE_OPENBLAS)
#include <references/openblas.hpp>
#endif

#include <algorithm>
#include <cstddef>
#include <thread>

namespace lazuli::bench {

using lazuli::detail::matrix;

void run_openblas(int repetitions) {
#if defined(LAZULI_HAVE_OPENBLAS)
    std::printf("\nOpenBLAS reference, double, best of %d\n", repetitions);
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
        std::printf("  n=%-5zu openblas/1t=%7.2f  openblas/%dt=%7.2f\n",
                    n,
                    gflops(flops, one),
                    all,
                    gflops(flops, many));
    }
#else
    (void)repetitions;
    std::printf("\nOpenBLAS was not found at configure time\n");
#endif
}

}  // namespace lazuli::bench
