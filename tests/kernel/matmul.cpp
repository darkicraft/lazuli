#include <lazuli/kernel/matmul.hpp>
#include <lazuli/lazuli.hpp>

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <utility>

namespace lazuli::bench::naive {

void matmul(const lazuli::detail::matrix_view<const double>& a,
            const lazuli::detail::matrix_view<const double>& b,
            lazuli::detail::matrix_view<double> out);

}  // namespace lazuli::bench::naive

namespace {

using lazuli::detail::kernel::matmul;

lazuli::matrix<double> reference(const lazuli::matrix<double>& a, const lazuli::matrix<double>& b) {
    lazuli::matrix<double> out(a.rows(), b.cols());
    lazuli::bench::naive::matmul(a, b, out);
    return out;
}

template <typename T>
double largest_difference(const lazuli::matrix<T>& left, const lazuli::matrix<T>& right) {
    REQUIRE(left.rows() == right.rows());
    REQUIRE(left.cols() == right.cols());
    double worst = 0.0;
    for (std::size_t row = 0; row < left.rows(); ++row) {
        for (std::size_t column = 0; column < left.cols(); ++column) {
            worst = std::max(worst,
                             static_cast<double>(std::abs(left(row, column) - right(row, column))));
        }
    }
    return worst;
}

void check_shape(std::size_t m, std::size_t k, std::size_t n) {
    const auto a = lazuli::matrix<double>::random(m, k, 17 + m, {-1.0, 1.0});
    const auto b = lazuli::matrix<double>::random(k, n, 29 + n, {-1.0, 1.0});
    const auto expected = reference(a, b);

    lazuli::matrix<double> actual(m, n);
    matmul(a, b, actual);

    // Both sides sum in a different order, so the comparison is a tight tolerance rather than
    // bit equality; the magnitude of each result is at most k, hence the 1e-12 * k allowance.
    const double tolerance = 1e-12 * static_cast<double>(k > 0 ? k : 1);
    REQUIRE(largest_difference(actual, expected) <= tolerance);
}

}  // namespace

TEST_CASE("kernel: shapes that fit one register tile") {
    for (std::size_t m : {std::size_t{1}, std::size_t{2}, std::size_t{3}}) {
        for (std::size_t k : {std::size_t{1}, std::size_t{2}, std::size_t{3}}) {
            for (std::size_t n : {std::size_t{1}, std::size_t{2}, std::size_t{3}}) {
                check_shape(m, k, n);
            }
        }
    }
}

TEST_CASE("kernel: shapes around the register tile and block sizes") {
    for (std::size_t size : {std::size_t{7},
                             std::size_t{8},
                             std::size_t{9},
                             std::size_t{11},
                             std::size_t{12},
                             std::size_t{13},
                             std::size_t{31},
                             std::size_t{63},
                             std::size_t{64},
                             std::size_t{65},
                             std::size_t{100}}) {
        check_shape(size, size, size);
    }
}

TEST_CASE("kernel: rectangular shapes") {
    check_shape(1, 100, 1);
    check_shape(13, 1, 17);
    check_shape(64, 129, 31);
    check_shape(129, 64, 71);
    check_shape(200, 3, 400);
}

TEST_CASE("kernel: a transposed operand") {
    const auto a = lazuli::matrix<double>::random(40, 25, 3, {-1.0, 1.0});
    const auto b = lazuli::matrix<double>::random(25, 40, 4, {-1.0, 1.0});
    const auto expected = reference(a, b);

    lazuli::matrix<double> actual(40, 40);
    matmul(a.transpose().transpose(), b, actual);
    REQUIRE(largest_difference(actual, expected) <= 1e-12 * 25.0);
}

TEST_CASE("kernel: a strided destination") {
    const auto a = lazuli::matrix<double>::random(20, 20, 5, {-1.0, 1.0});
    const auto b = lazuli::matrix<double>::random(20, 20, 6, {-1.0, 1.0});
    const auto expected = reference(a, b);

    // A block of a bigger matrix: rows are strided, columns are contiguous.
    lazuli::matrix<double> big(40, 40);
    auto destination = big.block(5, 7, 20, 20);
    matmul(a, b, destination);

    for (std::size_t row = 0; row < 20; ++row) {
        for (std::size_t column = 0; column < 20; ++column) {
            REQUIRE(std::abs(big(5 + row, 7 + column) - expected(row, column)) <= 1e-10);
        }
    }
}

TEST_CASE("kernel: a destination whose columns are not contiguous") {
    const auto a = lazuli::matrix<double>::random(15, 15, 7, {-1.0, 1.0});
    const auto b = lazuli::matrix<double>::random(15, 15, 8, {-1.0, 1.0});
    const auto expected = reference(a, b);

    // The transpose has column stride 15, so the vector stores cannot be used.
    lazuli::matrix<double> big(15, 15);
    matmul(a, b, big.transpose());

    for (std::size_t row = 0; row < 15; ++row) {
        for (std::size_t column = 0; column < 15; ++column) {
            REQUIRE(std::abs(big(column, row) - expected(row, column)) <= 1e-10);
        }
    }
}

namespace {

lazuli::matrix<float> scalar_float_product(const lazuli::matrix<float>& a,
                                           const lazuli::matrix<float>& b) {
    lazuli::matrix<float> out(a.rows(), b.cols());
    for (std::size_t row = 0; row < a.rows(); ++row) {
        for (std::size_t inner = 0; inner < a.cols(); ++inner) {
            const float value = a(row, inner);
            for (std::size_t column = 0; column < b.cols(); ++column) {
                out(row, column) += value * b(inner, column);
            }
        }
    }
    return out;
}

}  // namespace

TEST_CASE("kernel: float matches the scalar loop") {
    for (std::size_t size : {std::size_t{1},
                             std::size_t{7},
                             std::size_t{8},
                             std::size_t{9},
                             std::size_t{15},
                             std::size_t{16},
                             std::size_t{17},
                             std::size_t{33},
                             std::size_t{64},
                             std::size_t{100}}) {
        const auto a = lazuli::matrix<float>::random(size, size, 11 + size, {-1.0F, 1.0F});
        const auto b = lazuli::matrix<float>::random(size, size, 23 + size, {-1.0F, 1.0F});
        const auto expected = scalar_float_product(a, b);

        lazuli::matrix<float> actual(size, size);
        lazuli::detail::kernel::matmul(a, b, actual);

        const double tolerance = 1e-5 * static_cast<double>(size > 0 ? size : 1);
        REQUIRE(largest_difference(actual, expected) <= tolerance);
    }
}

TEST_CASE("kernel: the worker count does not change the result") {
    const auto a = lazuli::matrix<double>::random(200, 300, 31, {-1.0, 1.0});
    const auto b = lazuli::matrix<double>::random(300, 150, 32, {-1.0, 1.0});
    const auto expected = reference(a, b);

    for (std::size_t threads : {std::size_t{1}, std::size_t{2}, std::size_t{4}}) {
        lazuli::matrix<double> actual(200, 150);
        lazuli::detail::kernel::matmul(a, b, actual, lazuli::options{threads});
        REQUIRE(largest_difference(actual, expected) <= 1e-12 * 300.0);
    }
}

TEST_CASE("expr: eval accepts thread settings") {
    const auto a = lazuli::matrix<double>::random(150, 150, 41, {-1.0, 1.0});
    const auto b = lazuli::matrix<double>::random(150, 150, 42, {-1.0, 1.0});

    const auto single = (a * b).eval(lazuli::options{1});
    const auto parallel = (a * b).eval(lazuli::options{4});
    REQUIRE(largest_difference(single, parallel) <= 1e-12 * 150.0);
}
