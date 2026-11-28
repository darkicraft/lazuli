#include <lazuli/matrix/view.hpp>

#include <cmath>
#include <cstddef>
#include <utility>
#include <vector>

namespace lazuli::bench::naive {

using lazuli::detail::matrix_view;

namespace {

void swap_rows(matrix_view<double> a, std::size_t first, std::size_t second) {
    for (std::size_t column = 0; column < a.cols(); ++column) {
        std::swap(a(first, column), a(second, column));
    }
}

}  // namespace

bool lu(matrix_view<double> a, std::vector<std::size_t>& pivots) {
    const std::size_t size = a.rows();
    for (std::size_t index = 0; index < size; ++index) {
        pivots[index] = index;
    }

    for (std::size_t step = 0; step < size; ++step) {
        std::size_t pivot = step;
        double best = std::abs(a(step, step));
        for (std::size_t row = step + 1; row < size; ++row) {
            const double candidate = std::abs(a(row, step));
            if (candidate > best) {
                best = candidate;
                pivot = row;
            }
        }
        if (best == 0.0) {
            return false;
        }
        if (pivot != step) {
            swap_rows(a, step, pivot);
            std::swap(pivots[step], pivots[pivot]);
        }

        const double diagonal = a(step, step);
        for (std::size_t row = step + 1; row < size; ++row) {
            a(row, step) /= diagonal;
        }
        for (std::size_t row = step + 1; row < size; ++row) {
            const double factor = a(row, step);
            for (std::size_t column = step + 1; column < size; ++column) {
                a(row, column) -= factor * a(step, column);
            }
        }
    }
    return true;
}

bool cholesky(matrix_view<double> a) {
    const std::size_t size = a.rows();
    for (std::size_t step = 0; step < size; ++step) {
        double diagonal = a(step, step);
        for (std::size_t previous = 0; previous < step; ++previous) {
            diagonal -= a(step, previous) * a(step, previous);
        }
        if (diagonal <= 0.0) {
            return false;
        }
        const double pivot = std::sqrt(diagonal);
        a(step, step) = pivot;
        for (std::size_t row = step + 1; row < size; ++row) {
            double sum = a(row, step);
            for (std::size_t previous = 0; previous < step; ++previous) {
                sum -= a(row, previous) * a(step, previous);
            }
            a(row, step) = sum / pivot;
        }
    }
    return true;
}

}  // namespace lazuli::bench::naive
