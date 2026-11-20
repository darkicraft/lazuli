// One-dimensional storage, and handing it to matrix-shaped code.
//
//   cmake --build build --target vector_basic
//   ./build/examples/vector/basic

#include <lazuli/lazuli.hpp>

#include <cstdio>

namespace {

template <typename Grid>
void print(const char* label, const Grid& grid) {
    std::printf("%s (%zu x %zu)\n", label, grid.rows(), grid.cols());
    for (std::size_t row = 0; row < grid.rows(); ++row) {
        for (std::size_t column = 0; column < grid.cols(); ++column) {
            std::printf(" %6.2f", grid(row, column));
        }
        std::printf("\n");
    }
}

}  // namespace

int main() {
    // Zero-filled by size, or filled from a list.
    lazuli::vector<double> zeros(4);
    std::printf("zeros: %zu elements, first %.1f\n", zeros.size(), zeros[0]);

    lazuli::vector<double> values{1.0, 2.0, 3.0};
    std::printf("values: %.1f %.1f %.1f\n", values[0], values[1], values[2]);

    // operator[] is unchecked; at() validates.
    values[0] = 10.0;
    values.at(2) = 30.0;

    // A seeded random vector, and filling one in place.
    lazuli::vector<double> noise = lazuli::vector<double>::random(4, 21, {-1.0, 1.0});
    std::printf("random: %.3f %.3f %.3f %.3f\n", noise[0], noise[1], noise[2], noise[3]);
    noise.fill(0.5);
    std::printf("after fill(0.5): %.3f %.3f\n", noise[0], noise[3]);

    // Iterators are plain pointers, so range-for and algorithms work directly.
    double total = 0.0;
    for (const double value : values) {
        total += value;
    }
    std::printf("sum: %.1f\n", total);

    // as_column() is a view, not a copy: matrix-shaped code can take the vector
    // as an n x 1 grid, and writes reach the vector.
    auto column = values.as_column();
    print("as_column()", column);
    column(1, 0) = 20.0;
    std::printf("after writing through the column, values[1] = %.1f\n", values[1]);

    return 0;
}
