// Reproducible random matrices, and filling an existing one.
//
//   cmake --build build --target matrix_random
//   ./build/examples/matrix/random

#include <lazuli/lazuli.hpp>

#include <cstdio>

namespace {

void print(const char* label, const lazuli::matrix<double>& matrix) {
    std::printf("%s (%zu x %zu)\n", label, matrix.rows(), matrix.cols());
    for (std::size_t row = 0; row < matrix.rows(); ++row) {
        for (std::size_t column = 0; column < matrix.cols(); ++column) {
            std::printf(" %7.4f", matrix(row, column));
        }
        std::printf("\n");
    }
}

}  // namespace

int main() {
    // random(rows, cols, seed, bounds): the same seed gives the same matrix everywhere,
    // because the sampling is SplitMix64 rather than a standard distribution.
    const auto first = lazuli::matrix<double>::random(3, 4, 42, {-1.0, 1.0});
    const auto second = lazuli::matrix<double>::random(3, 4, 42, {-1.0, 1.0});
    print("random(seed 42)", first);
    std::printf("same seed reproduces it: %s\n", first(1, 2) == second(1, 2) ? "yes" : "no");

    // A fill value for the whole matrix.
    lazuli::matrix<double> plate(2, 3);
    plate.fill(2.5);
    print("fill(2.5)", plate);

    // A generator receives the coordinates, so structure is easy to express.
    lazuli::matrix<double> identity(3, 3);
    identity.fill([](std::size_t row, std::size_t column) { return row == column ? 1.0 : 0.0; });
    print("fill(generator)", identity);

    // Both work on a window: only the block changes.
    lazuli::matrix<double> padded(4, 4);
    padded.block(1, 1, 2, 2).fill(-1.0);
    print("block(1, 1, 2, 2).fill(-1.0)", padded);

    return 0;
}
