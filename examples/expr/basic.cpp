// Lazy expressions: elementwise chains fuse, products cost at most one temporary.
//
//   cmake --build build --target expr_basic
//   ./build/examples/expr/basic

#include <lazuli/lazuli.hpp>

#include <cstdio>

namespace {

void print(const char* label, const lazuli::matrix<double>& matrix) {
    std::printf("%s (%zu x %zu)\n", label, matrix.rows(), matrix.cols());
    for (std::size_t row = 0; row < matrix.rows(); ++row) {
        for (std::size_t column = 0; column < matrix.cols(); ++column) {
            std::printf(" %7.2f", matrix(row, column));
        }
        std::printf("\n");
    }
}

}  // namespace

int main() {
    const lazuli::matrix<double> a{{1.0, 2.0}, {3.0, 4.0}};
    const lazuli::matrix<double> b{{5.0, 6.0}, {7.0, 8.0}};

    // Nothing is computed until the assignment; the chain then runs in one pass and allocates
    // nothing, because a and b already exist and c is already the right shape.
    lazuli::matrix<double> c(2, 2);
    c = 2.0 * a + b - a / 2.0;
    print("2a + b - a/2", c);

    // A product is the one thing that cannot fuse: as the whole right-hand side it writes
    // straight into c, with no intermediate.
    c = a * b;
    print("a * b", c);

    // Inside a larger expression the product materialises once, then the elementwise tail runs
    // over it.
    c = a * b + a;
    print("a * b + a", c);

    // noalias() promises the destination cannot overlap any operand, which skips the overlap
    // check and the temporary it would otherwise create.
    c.noalias() = a * b;
    print("noalias: a * b", c);

    // A view is an operand like any other: transposing costs nothing.
    lazuli::matrix<double> transposed = a.transpose();
    print("a.transpose()", transposed);

    // eval() materialises an expression when you want to keep the result.
    const auto kept = (a + b).eval();
    print("(a + b).eval()", kept);

    // Vectors are single-column operands.
    const lazuli::vector<double> x{1.0, 2.0};
    const lazuli::vector<double> y = a * x;
    std::printf("a * x = %.1f %.1f\n", y[0], y[1]);

    return 0;
}
