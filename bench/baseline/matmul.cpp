#include "baseline/matmul.hpp"

#include <cstddef>

namespace lazuli::bench::baseline {

void matmul(const lazuli::detail::matrix_view<const double>& a,
            const lazuli::detail::matrix_view<const double>& b,
            lazuli::detail::matrix_view<double> out) {
    for (std::size_t row = 0; row < a.rows(); ++row) {
        for (std::size_t inner = 0; inner < a.cols(); ++inner) {
            const double value = a(row, inner);
            for (std::size_t column = 0; column < b.cols(); ++column) {
                out(row, column) += value * b(inner, column);
            }
        }
    }
}

}  // namespace lazuli::bench::baseline
