#include <catch2/catch_test_macros.hpp>

#include <lazuli/core/allocation.hpp>
#include <lazuli/lazuli.hpp>

namespace {

lazuli::matrix<double> sample() {
    return {{1.0, 2.0}, {3.0, 4.0}};
}

lazuli::matrix<double> other() {
    return {{10.0, 20.0}, {30.0, 40.0}};
}

}  // namespace

TEST_CASE("expr: addition") {
    const auto a = sample();
    const auto b = other();
    lazuli::matrix<double> c(2, 2);
    c = a + b;
    REQUIRE(c(0, 0) == 11.0);
    REQUIRE(c(0, 1) == 22.0);
    REQUIRE(c(1, 0) == 33.0);
    REQUIRE(c(1, 1) == 44.0);
}

TEST_CASE("expr: subtraction") {
    const auto a = sample();
    const auto b = other();
    lazuli::matrix<double> c(2, 2);
    c = b - a;
    REQUIRE(c(0, 0) == 9.0);
    REQUIRE(c(1, 1) == 36.0);
}

TEST_CASE("expr: unary minus") {
    const auto a = sample();
    lazuli::matrix<double> c(2, 2);
    c = -a;
    REQUIRE(c(0, 0) == -1.0);
    REQUIRE(c(1, 1) == -4.0);
}

TEST_CASE("expr: scalar multiplication and division") {
    const auto a = sample();
    lazuli::matrix<double> c(2, 2);
    c = 3.0 * a;
    REQUIRE(c(1, 0) == 9.0);
    c = a / 2.0;
    REQUIRE(c(1, 0) == 1.5);
    c = a * 2.0;
    REQUIRE(c(0, 1) == 4.0);
}

TEST_CASE("expr: chains fuse") {
    const auto a = sample();
    const auto b = other();
    lazuli::matrix<double> c(2, 2);
    c = 2.0 * a + b - a / 2.0;
    REQUIRE(c(0, 0) == 11.5);
    REQUIRE(c(1, 1) == 46.0);
}

TEST_CASE("expr: elementwise assignment allocates nothing") {
    const auto a = sample();
    const auto b = other();
    lazuli::matrix<double> c(2, 2);

    lazuli::detail::reset_allocation_count();
    c = 2.0 * a + b - a / 2.0;
    REQUIRE(lazuli::detail::allocation_count() == 0);
}

TEST_CASE("expr: shape mismatch is caught where the expression is written") {
    const auto a = sample();
    const lazuli::matrix<double> wide(2, 3);
    REQUIRE_THROWS_AS(a + wide, lazuli::dimension_mismatch);
}

TEST_CASE("expr: eval materialises") {
    const auto a = sample();
    const auto b = other();
    const auto c = (2.0 * a + b).eval();
    REQUIRE(c.rows() == 2);
    REQUIRE(c.cols() == 2);
    REQUIRE(c(0, 1) == 24.0);
}

TEST_CASE("expr: assignment resizes an empty destination") {
    const auto a = sample();
    lazuli::matrix<double> c;
    c = a + a;
    REQUIRE(c.rows() == 2);
    REQUIRE(c.cols() == 2);
    REQUIRE(c(0, 0) == 2.0);
}

TEST_CASE("expr: assignment to a sized destination keeps its shape") {
    const auto a = sample();
    lazuli::matrix<double> c(3, 3);
    REQUIRE_THROWS_AS(c = a + a, lazuli::dimension_mismatch);
}

TEST_CASE("expr: views participate directly") {
    const auto a = sample();
    const auto b = other();
    lazuli::matrix<double> c(2, 2);
    c = a.transpose() + b;
    REQUIRE(c(0, 0) == 11.0);
    REQUIRE(c(0, 1) == 23.0);
    REQUIRE(c(1, 0) == 32.0);
    REQUIRE(c(1, 1) == 44.0);
}

TEST_CASE("expr: a block takes part as a window") {
    lazuli::matrix<double> a{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {7.0, 8.0, 9.0}};
    lazuli::matrix<double> c(2, 2);
    c = a.block(0, 1, 2, 2) + a.block(1, 0, 2, 2);
    REQUIRE(c(0, 0) == 6.0);
    REQUIRE(c(1, 1) == 14.0);
}
