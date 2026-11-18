#include <catch2/catch_test_macros.hpp>

#include <lazuli/core/allocation.hpp>
#include <lazuli/lazuli.hpp>

TEST_CASE("expr: a matrix-vector product") {
    const lazuli::matrix<double> a{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};
    const lazuli::vector<double> x{1.0, 1.0, 1.0};

    lazuli::vector<double> y = a * x;
    REQUIRE(y.size() == 2);
    REQUIRE(y[0] == 6.0);
    REQUIRE(y[1] == 15.0);
}

TEST_CASE("expr: a vector-vector elementwise expression") {
    const lazuli::vector<double> x{1.0, 2.0};
    const lazuli::vector<double> z{10.0, 20.0};

    lazuli::vector<double> y = 2.0 * x + z;
    REQUIRE(y[0] == 12.0);
    REQUIRE(y[1] == 24.0);
}

TEST_CASE("expr: vectors stay single-column inside a matrix expression") {
    const lazuli::vector<double> x{1.0, 2.0, 3.0};
    lazuli::matrix<double> c(3, 1);
    c = x + x;
    REQUIRE(c.rows() == 3);
    REQUIRE(c.cols() == 1);
    REQUIRE(c(2, 0) == 6.0);
}

TEST_CASE("expr: assigning into a sized vector keeps its shape") {
    const lazuli::vector<double> x{1.0, 2.0, 3.0};
    lazuli::vector<double> y(2);
    REQUIRE_THROWS_AS(y = x + x, lazuli::dimension_mismatch);
}

TEST_CASE("expr: a multi-column result cannot become a vector") {
    const lazuli::matrix<double> a{{1.0, 0.0}, {0.0, 1.0}};
    const lazuli::matrix<double> b{{1.0, 2.0}, {3.0, 4.0}};
    REQUIRE_THROWS_AS(lazuli::vector<double>{a * b}, lazuli::dimension_mismatch);
}

TEST_CASE("expr: vector expressions allocate nothing") {
    const lazuli::vector<double> x{1.0, 2.0, 3.0};
    const lazuli::vector<double> z{4.0, 5.0, 6.0};
    lazuli::vector<double> y(3);

    lazuli::detail::reset_allocation_count();
    y = 2.0 * x - z;
    REQUIRE(lazuli::detail::allocation_count() == 0);
}
