// Non-owning views: blocks, rows, columns and transposes.
//
// Every view is a window onto the original storage, so writes travel back to the
// matrix and nothing is copied.
//
//   cmake --build build --target matrix_views
//   ./build/examples/matrix/views

#include <lazuli/lazuli.hpp>

#include <cstdio>

namespace {

template <typename Grid>
void print(const char* label, const Grid& grid) {
    std::printf("%s (%zu x %zu)\n", label, grid.rows(), grid.cols());
    for (std::size_t row = 0; row < grid.rows(); ++row) {
        for (std::size_t column = 0; column < grid.cols(); ++column) {
            std::printf(" %5.1f", grid(row, column));
        }
        std::printf("\n");
    }
}

lazuli::matrix<double> numbers() {
    return {{1.0, 2.0, 3.0, 4.0}, {5.0, 6.0, 7.0, 8.0}, {9.0, 10.0, 11.0, 12.0}};
}

}  // namespace

int main() {
    lazuli::matrix<double> matrix = numbers();
    print("matrix", matrix);

    // block(row, col, rows, cols): a sub-window, written through.
    auto block = matrix.block(1, 1, 2, 2);
    print("block(1, 1, 2, 2)", block);
    block(0, 0) = 60.0;
    std::printf("after writing through the block, matrix(1,1) = %.1f\n", matrix(1, 1));

    // A row is contiguous; a column is strided but still free to create.
    auto row = matrix.row(0);
    print("row(0)", row);
    auto column = matrix.col(3);
    print("col(3)", column);

    // transpose() swaps shape and strides instead of moving data.
    auto transposed = matrix.transpose();
    print("transpose()", transposed);
    transposed(0, 2) = 90.0;
    std::printf("writing through the transpose set matrix(2,0) = %.1f\n", matrix(2, 0));

    // A const matrix yields a read-only view, and the conversion is implicit.
    const auto& read_only = matrix;
    lazuli::matrix_view<const double> view = read_only;
    print("read-only view", view);

    return 0;
}
