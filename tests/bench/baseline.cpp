#include <lazuli/lazuli.hpp>

#include <catch2/catch_test_macros.hpp>

#include <cstddef>

namespace lazuli::bench::naive {

void matmul(const lazuli::detail::matrix_view<const double>& a,
            const lazuli::detail::matrix_view<const double>& b,
            lazuli::detail::matrix_view<double> out);

}  // namespace lazuli::bench::naive

namespace {

lazuli::matrix<double> multiply(const lazuli::matrix<double>& a, const lazuli::matrix<double>& b) {
    lazuli::matrix<double> out(a.rows(), b.cols());
    lazuli::bench::naive::matmul(a, b, out);
    return out;
}

}  // namespace

TEST_CASE("baseline: a 2x3 by 3x2 product") {
    const lazuli::matrix<double> a{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};
    const lazuli::matrix<double> b{{7.0, 8.0}, {9.0, 10.0}, {11.0, 12.0}};
    const auto c = multiply(a, b);
    REQUIRE(c.rows() == 2);
    REQUIRE(c.cols() == 2);
    REQUIRE(c(0, 0) == 58.0);
    REQUIRE(c(0, 1) == 64.0);
    REQUIRE(c(1, 0) == 139.0);
    REQUIRE(c(1, 1) == 154.0);
}

TEST_CASE("baseline: identity on both sides") {
    const lazuli::matrix<double> identity{{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}};
    const lazuli::matrix<double> a{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {7.0, 8.0, 9.0}};
    const auto right = multiply(a, identity);
    const auto left = multiply(identity, a);
    for (std::size_t row = 0; row < 3; ++row) {
        for (std::size_t column = 0; column < 3; ++column) {
            REQUIRE(right(row, column) == a(row, column));
            REQUIRE(left(row, column) == a(row, column));
        }
    }
}

TEST_CASE("baseline: a 1x1 product") {
    const lazuli::matrix<double> a{{3.0}};
    const lazuli::matrix<double> b{{4.0}};
    const auto c = multiply(a, b);
    REQUIRE(c.rows() == 1);
    REQUIRE(c.cols() == 1);
    REQUIRE(c(0, 0) == 12.0);
}

TEST_CASE("baseline: a row times a column is a scalar") {
    const lazuli::matrix<double> row{{1.0, 2.0, 3.0}};
    const lazuli::matrix<double> column{{4.0}, {5.0}, {6.0}};
    const auto c = multiply(row, column);
    REQUIRE(c.rows() == 1);
    REQUIRE(c.cols() == 1);
    REQUIRE(c(0, 0) == 32.0);
}

TEST_CASE("baseline: a column times a row is a full matrix") {
    const lazuli::matrix<double> column{{1.0}, {2.0}};
    const lazuli::matrix<double> row{{3.0, 4.0}};
    const auto c = multiply(column, row);
    REQUIRE(c.rows() == 2);
    REQUIRE(c.cols() == 2);
    REQUIRE(c(0, 0) == 3.0);
    REQUIRE(c(0, 1) == 4.0);
    REQUIRE(c(1, 0) == 6.0);
    REQUIRE(c(1, 1) == 8.0);
}

TEST_CASE("baseline: a zero matrix produces zeros") {
    const lazuli::matrix<double> zero(2, 2);
    const lazuli::matrix<double> a{{1.0, 2.0}, {3.0, 4.0}};
    const auto c = multiply(zero, a);
    REQUIRE(c(0, 0) == 0.0);
    REQUIRE(c(1, 1) == 0.0);
}
