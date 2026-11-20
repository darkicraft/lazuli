#include <lazuli/core/allocation.hpp>
#include <lazuli/lazuli.hpp>

#include <catch2/catch_test_macros.hpp>

namespace {

lazuli::matrix<double> small_a() {
    return {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};
}

lazuli::matrix<double> small_b() {
    return {{7.0, 8.0}, {9.0, 10.0}, {11.0, 12.0}};
}

}  // namespace

TEST_CASE("expr: a matrix product") {
    const auto a = small_a();
    const auto b = small_b();
    lazuli::matrix<double> c(2, 2);
    c = a * b;
    REQUIRE(c(0, 0) == 58.0);
    REQUIRE(c(0, 1) == 64.0);
    REQUIRE(c(1, 0) == 139.0);
    REQUIRE(c(1, 1) == 154.0);
}

TEST_CASE("expr: a product with an elementwise tail") {
    const auto a = small_a();
    const auto b = small_b();
    lazuli::matrix<double> ones(2, 2);
    ones.fill(1.0);
    lazuli::matrix<double> c(2, 2);
    c = 2.0 * a * b + ones;
    REQUIRE(c(0, 0) == 117.0);
    REQUIRE(c(1, 1) == 309.0);
}

TEST_CASE("expr: a root product allocates nothing") {
    const auto a = small_a();
    const auto b = small_b();
    lazuli::matrix<double> c(2, 2);

    lazuli::detail::reset_allocation_count();
    c = a * b;
    REQUIRE(lazuli::detail::allocation_count() == 0);
}

TEST_CASE("expr: a nested product costs exactly one temporary") {
    const auto a = small_a();
    const auto b = small_b();
    lazuli::matrix<double> ones(2, 2);
    ones.fill(1.0);
    lazuli::matrix<double> c(2, 2);

    lazuli::detail::reset_allocation_count();
    c = a * b + ones;
    REQUIRE(lazuli::detail::allocation_count() == 1);
}

TEST_CASE("expr: chained products cost one temporary") {
    const lazuli::matrix<double> a{{1.0, 2.0}, {3.0, 4.0}};
    const lazuli::matrix<double> b{{1.0, 0.0}, {0.0, 1.0}};
    lazuli::matrix<double> c(2, 2);

    lazuli::detail::reset_allocation_count();
    c = a * b * a;
    REQUIRE(lazuli::detail::allocation_count() == 1);
    REQUIRE(c(0, 0) == 7.0);
    REQUIRE(c(1, 1) == 22.0);
}

TEST_CASE("expr: a transposed view multiplies") {
    const lazuli::matrix<double> a{{1.0, 4.0}, {2.0, 5.0}, {3.0, 6.0}};
    const lazuli::matrix<double> b{{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}};
    lazuli::matrix<double> c(2, 3);
    c = a.transpose() * b;
    REQUIRE(c(0, 0) == 1.0);
    REQUIRE(c(0, 1) == 2.0);
    REQUIRE(c(0, 2) == 3.0);
    REQUIRE(c(1, 0) == 4.0);
}

TEST_CASE("expr: a product of sums") {
    const lazuli::matrix<double> a{{1.0, 1.0}, {0.0, 1.0}};
    const lazuli::matrix<double> b{{2.0, 0.0}, {0.0, 2.0}};
    lazuli::matrix<double> c(2, 2);
    c = (a + b) * (a - b);
    REQUIRE(c(0, 0) == -3.0);
    REQUIRE(c(0, 1) == 2.0);
    REQUIRE(c(1, 0) == 0.0);
    REQUIRE(c(1, 1) == -3.0);
}

TEST_CASE("expr: mismatched products are caught at construction") {
    const lazuli::matrix<double> a(2, 3);
    const lazuli::matrix<double> b(2, 3);
    REQUIRE_THROWS_AS(a * b, lazuli::dimension_mismatch);
}
