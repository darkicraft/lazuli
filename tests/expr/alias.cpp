#include <lazuli/core/allocation.hpp>
#include <lazuli/lazuli.hpp>

#include <catch2/catch_test_macros.hpp>

TEST_CASE("alias: a self-referencing sum stays correct") {
    lazuli::matrix<double> a{{1.0, 2.0}, {3.0, 4.0}};
    const lazuli::matrix<double> b{{10.0, 20.0}, {30.0, 40.0}};
    a = a + b;
    REQUIRE(a(0, 0) == 11.0);
    REQUIRE(a(0, 1) == 22.0);
    REQUIRE(a(1, 0) == 33.0);
    REQUIRE(a(1, 1) == 44.0);
}

TEST_CASE("alias: a self-referencing scale stays correct") {
    lazuli::matrix<double> a{{1.0, 2.0}, {3.0, 4.0}};
    a = 2.0 * a;
    REQUIRE(a(0, 0) == 2.0);
    REQUIRE(a(1, 1) == 8.0);
}

TEST_CASE("alias: a self product stays correct") {
    lazuli::matrix<double> a{{1.0, 2.0}, {3.0, 4.0}};
    a = a * a;
    REQUIRE(a(0, 0) == 7.0);
    REQUIRE(a(0, 1) == 10.0);
    REQUIRE(a(1, 0) == 15.0);
    REQUIRE(a(1, 1) == 22.0);
}

TEST_CASE("alias: a block of the destination is detected") {
    lazuli::matrix<double> a{{1.0, 2.0}, {3.0, 4.0}};
    const lazuli::matrix<double> b{{10.0, 20.0}, {30.0, 40.0}};

    lazuli::detail::reset_allocation_count();
    a = a.block(0, 0, 2, 2) + b;
    REQUIRE(lazuli::detail::allocation_count() == 1);
    REQUIRE(a(0, 0) == 11.0);
    REQUIRE(a(1, 1) == 44.0);
}

TEST_CASE("alias: one temporary when the destination is an operand") {
    lazuli::matrix<double> a{{1.0, 2.0}, {3.0, 4.0}};
    const lazuli::matrix<double> b{{10.0, 20.0}, {30.0, 40.0}};

    lazuli::detail::reset_allocation_count();
    a = a + b;
    REQUIRE(lazuli::detail::allocation_count() == 1);
}

TEST_CASE("alias: noalias skips the temporary") {
    lazuli::matrix<double> c(2, 2);
    const lazuli::matrix<double> a{{1.0, 2.0}, {3.0, 4.0}};
    const lazuli::matrix<double> b{{10.0, 20.0}, {30.0, 40.0}};

    lazuli::detail::reset_allocation_count();
    c.noalias() = a + b;
    REQUIRE(lazuli::detail::allocation_count() == 0);
    REQUIRE(c(0, 0) == 11.0);
}

TEST_CASE("alias: distinct buffers need no temporary") {
    const lazuli::matrix<double> a{{1.0, 2.0}, {3.0, 4.0}};
    const lazuli::matrix<double> b{{10.0, 20.0}, {30.0, 40.0}};
    lazuli::matrix<double> c(2, 2);

    lazuli::detail::reset_allocation_count();
    c = 2.0 * a + b;
    REQUIRE(lazuli::detail::allocation_count() == 0);
}
