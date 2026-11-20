// Solving A x = b, with a per-call thread setting.
//
//   cmake --build build --target decomp_solve
//   ./build/examples/decomp/solve

#include <lazuli/lazuli.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>

namespace {

double max_residual(const lazuli::matrix<double>& a,
                    const lazuli::vector<double>& x,
                    const lazuli::vector<double>& b) {
    double worst = 0.0;
    for (std::size_t row = 0; row < a.rows(); ++row) {
        double sum = 0.0;
        for (std::size_t column = 0; column < a.cols(); ++column) {
            sum += a(row, column) * x[column];
        }
        worst = std::max(worst, std::abs(sum - b[row]));
    }
    return worst;
}

}  // namespace

int main() {
    const auto a = lazuli::matrix<double>::random(64, 64, 7, {-1.0, 1.0});
    const auto b = lazuli::vector<double>::random(64, 11, {-1.0, 1.0});

    // The default uses whatever the runtime considers the default thread count.
    const auto automatic = lazuli::solve(a, b);
    const auto single = lazuli::solve(a, b, lazuli::options{1});
    if (!automatic || !single) {
        std::printf("solve failed: %s\n",
                    automatic ? single.error().message.c_str() : automatic.error().message.c_str());
        return 1;
    }

    std::printf("first entries: %.6f (default threads), %.6f (one thread)\n",
                (*automatic)[0],
                (*single)[0]);
    std::printf("residual: %.3e\n", max_residual(a, *automatic, b));
    return 0;
}
