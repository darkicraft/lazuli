// Per-call thread control: the library never touches the process-wide OpenMP defaults.
//
//   cmake --build build --target expr_threads
//   ./build/examples/expr/threads

#include <lazuli/lazuli.hpp>

#include <cstdio>

int main() {
    const auto a = lazuli::matrix<double>::random(256, 256, 1, {-1.0, 1.0});
    const auto b = lazuli::matrix<double>::random(256, 256, 2, {-1.0, 1.0});

    // 0 means "whatever the runtime considers the default", so OMP_NUM_THREADS is respected.
    const auto automatic = (a * b).eval();
    const auto single = (a * b).eval(lazuli::options{1});
    const auto four = (a * b).eval(lazuli::options{4});

    std::printf("default threads: %.6f\n", automatic(0, 0));
    std::printf("one thread     : %.6f\n", single(0, 0));
    std::printf("four threads   : %.6f\n", four(0, 0));

    // Same arithmetic, different partitioning: the results match to rounding.
    const double spread = single(0, 0) - four(0, 0);
    std::printf("difference between one and four threads: %.3e\n", spread < 0 ? -spread : spread);

    // Plain assignment uses the default; pass settings through eval() when you care.
    lazuli::matrix<double> assigned(256, 256);
    assigned = a * b;
    std::printf("assigned[0,0]: %.6f\n", assigned(0, 0));

    return 0;
}
