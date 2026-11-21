// Timings for the factorisations.
//
//   cmake --build build/release --target bench_decomp
//   ./build/release/bench/decomp
//
// LU costs about 2/3 n^3 flops and Cholesky about 1/3 n^3, which is what the GFLOPS column
// divides by. The trailing updates go through the matmul kernel, so the thread setting reaches
// them; the panel factorisation itself is serial.

#include <lazuli/lazuli.hpp>

#include <chrono>
#include <cstddef>
#include <cstdio>

namespace {

using clock_type = std::chrono::steady_clock;

template <typename Work>
double best_of(int repetitions, Work work) {
    double best = 1e9;
    for (int run = 0; run <= repetitions; ++run) {
        const auto start = clock_type::now();
        work();
        const auto stop = clock_type::now();
        if (run > 0) {
            best = std::min(best, std::chrono::duration<double>(stop - start).count());
        }
    }
    return best;
}

}  // namespace

int main() {
    for (std::size_t n :
         {std::size_t{128}, std::size_t{256}, std::size_t{512}, std::size_t{1024}}) {
        const auto a = lazuli::matrix<double>::random(n, n, 1, {-1.0, 1.0});
        // A symmetric positive-definite matrix for Cholesky.
        lazuli::matrix<double> spd = a.transpose() * a;
        for (std::size_t index = 0; index < n; ++index) {
            spd(index, index) += static_cast<double>(n);
        }

        const double size = static_cast<double>(n);
        const double lu_flops = 2.0 / 3.0 * size * size * size;
        const double chol_flops = 1.0 / 3.0 * size * size * size;

        const double lu_one = best_of(4, [&] { (void)lazuli::nt::lu(a, lazuli::options{1}); });
        const double lu_all = best_of(4, [&] { (void)lazuli::nt::lu(a); });
        const double chol_one =
            best_of(4, [&] { (void)lazuli::nt::cholesky(spd, lazuli::options{1}); });
        const double chol_all = best_of(4, [&] { (void)lazuli::nt::cholesky(spd); });

        std::printf(
            "n=%-5zu lu/1t=%6.2f lu/all=%6.2f (%4.2fx)  chol/1t=%6.2f "
            "chol/all=%6.2f (%4.2fx)\n",
            n,
            lu_flops / lu_one / 1e9,
            lu_flops / lu_all / 1e9,
            lu_one / lu_all,
            chol_flops / chol_one / 1e9,
            chol_flops / chol_all / 1e9,
            chol_one / chol_all);
    }
    return 0;
}
