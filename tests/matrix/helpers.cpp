#include <lazuli/lazuli.hpp>

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstddef>
#include <format>
#include <string>

TEST_CASE("matrix: identity has ones on the diagonal") {
    const auto identity = lazuli::matrix<double>::identity(3);
    REQUIRE(identity.rows() == 3);
    REQUIRE(identity.cols() == 3);
    for (std::size_t row = 0; row < 3; ++row) {
        for (std::size_t column = 0; column < 3; ++column) {
            REQUIRE(identity(row, column) == (row == column ? 1.0 : 0.0));
        }
    }
    REQUIRE(lazuli::matrix<double>::identity(0).empty());
}

TEST_CASE("matrix: the Frobenius norm") {
    const lazuli::matrix<double> matrix{{3.0, 0.0}, {0.0, 4.0}};
    REQUIRE(std::abs(matrix.norm() - 5.0) <= 1e-12);

    const lazuli::vector<double> values{1.0, 2.0, 2.0};
    REQUIRE(std::abs(values.norm() - 3.0) <= 1e-12);

    // A view reports the norm of the window, not of the matrix behind it.
    const lazuli::matrix<double> bigger{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};
    REQUIRE(std::abs(bigger.block(0, 0, 1, 2).norm() - std::sqrt(5.0)) <= 1e-12);
}

TEST_CASE("matrix: approx_equal") {
    const lazuli::matrix<double> left{{1.0, 2.0}, {3.0, 4.0}};
    const lazuli::matrix<double> close{{1.0, 2.0}, {3.0, 4.0 + 1e-13}};
    const lazuli::matrix<double> far{{1.0, 2.0}, {3.0, 4.1}};
    const lazuli::matrix<double> wider(2, 3);

    REQUIRE(left.approx_equal(left, 0.0));
    REQUIRE(left.approx_equal(close, 1e-12));
    REQUIRE(!left.approx_equal(far, 1e-12));
    REQUIRE(!left.approx_equal(wider, 1e-12));
    REQUIRE(left.approx_equal(lazuli::matrix<double>::identity(2) * left, 1e-12));
}

TEST_CASE("format: matrices and vectors") {
    const lazuli::matrix<double> matrix{{1.0, 2.0}, {3.0, 4.0}};
    REQUIRE(std::format("{}", matrix) == "[[1 2]; [3 4]]");

    const lazuli::vector<double> values{1.0, 2.5};
    REQUIRE(std::format("{}", values) == "[1 2.5]");

    REQUIRE(std::format("{}", lazuli::matrix<double>()) == "[]");
}
