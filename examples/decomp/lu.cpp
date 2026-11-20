// LU with partial pivoting, and the determinant and inverse derived from it.
//
//   cmake --build build --target decomp_lu
//   ./build/examples/decomp/lu

#include <lazuli/lazuli.hpp>

#include <cstddef>
#include <cstdio>

namespace {

void print(const char* label, const lazuli::matrix<double>& matrix) {
    std::printf("%s (%zu x %zu)\n", label, matrix.rows(), matrix.cols());
    for (std::size_t row = 0; row < matrix.rows(); ++row) {
        for (std::size_t column = 0; column < matrix.cols(); ++column) {
            std::printf(" %8.4f", matrix(row, column));
        }
        std::printf("\n");
    }
}

}  // namespace

int main() {
    // A pivot is needed here: the first column's largest entry is in the second row.
    const lazuli::matrix<double> a{{0.0, 2.0}, {3.0, 4.0}};

    const auto factors = lazuli::lu(a);
    if (!factors) {
        std::printf("lu failed: %s\n", factors.error().message.c_str());
        return 1;
    }
    print("L", factors->l);
    print("U", factors->u);
    std::printf("row permutation: %zu %zu, %zu interchange(s)\n",
                factors->pivots[0],
                factors->pivots[1],
                factors->swaps);

    // Determinant and inverse come from the same factors.
    std::printf("det: %.4f\n", lazuli::det(a));
    const auto inverse = lazuli::inverse(a);
    if (!inverse) {
        std::printf("inverse failed: %s\n", inverse.error().message.c_str());
        return 1;
    }
    print("inverse", *inverse);

    // The allocation-free path: factor a copy in place and keep the permutation.
    lazuli::matrix<double> working = a;
    const auto pivots = lazuli::lu_in_place(working.view());
    if (!pivots) {
        std::printf("lu_in_place failed: %s\n", pivots.error().message.c_str());
        return 1;
    }
    std::printf("in place: U[0,0] = %.4f, permutation %zu %zu\n",
                working(0, 0),
                (*pivots)[0],
                (*pivots)[1]);

    return 0;
}
