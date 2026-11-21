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

TEST_CASE("format: matrices are compact and rows are bracketed") {
    const lazuli::matrix<double> matrix{{1.0, 2.0}, {3.0, 4.0}};
    REQUIRE(std::format("{}", matrix) == "[[1 2] [3 4]]");

    // The spec is consumed, so text may follow the replacement field.
    REQUIRE(std::format("{}!", matrix) == "[[1 2] [3 4]]!");

    const lazuli::vector<double> values{1.0, 2.5};
    REQUIRE(std::format("{}", values) == "[1 2.5]");
}

TEST_CASE("format: the alternate form puts every row on its own line") {
    const lazuli::matrix<double> matrix{{1.0, 2.0}, {3.0, 4.0}};
    REQUIRE(std::format("{:#}", matrix) == "[[1 2]\n [3 4]]");

    // A row has no meaning for a vector, so the flag changes nothing.
    const lazuli::vector<double> values{1.0, 2.5};
    REQUIRE(std::format("{:#}", values) == "[1 2.5]");
}

TEST_CASE("format: the rest of the spec reaches each element") {
    const lazuli::matrix<double> matrix{{1.0, 2.0}, {3.0, 4.125}};
    REQUIRE(std::format("{:.2f}", matrix) == "[[1.00 2.00] [3.00 4.12]]");
    REQUIRE(std::format("{:+.1f}", matrix) == "[[+1.0 +2.0] [+3.0 +4.1]]");
    REQUIRE(std::format("{:#.1f}", matrix) == "[[1.0 2.0]\n [3.0 4.1]]");

    // Width and alignment come from the scalar formatter, not from a reimplementation.
    REQUIRE(std::format("{:6.2f}", matrix) == "[[  1.00   2.00] [  3.00   4.12]]");

    const lazuli::vector<double> values{1.0, 2.5};
    REQUIRE(std::format("{:.3f}", values) == "[1.000 2.500]");
}

TEST_CASE("format: degenerate shapes print as an empty grid") {
    REQUIRE(std::format("{}", lazuli::matrix<double>()) == "[]");
    REQUIRE(std::format("{}", lazuli::matrix<double>(0, 5)) == "[]");
    REQUIRE(std::format("{}", lazuli::matrix<double>(5, 0)) == "[]");
    REQUIRE(std::format("{:#}", lazuli::matrix<double>(0, 5)) == "[]");
    REQUIRE(std::format("{}", lazuli::vector<double>()) == "[]");
}

TEST_CASE("format: a large matrix is shortened with an ellipsis") {
    lazuli::matrix<double> grid(40, 40);
    for (std::size_t row = 0; row < grid.rows(); ++row) {
        for (std::size_t column = 0; column < grid.cols(); ++column) {
            grid(row, column) = static_cast<double>(row * grid.cols() + column);
        }
    }

    const std::string shown = std::format("{}", grid);
    REQUIRE(shown ==
            "[[0 1 2 ... 37 38 39] [40 41 42 ... 77 78 79] [80 81 82 ... 117 118 119] "
            "... [1480 1481 1482 ... 1517 1518 1519] [1520 1521 1522 ... 1557 1558 1559] "
            "[1560 1561 1562 ... 1597 1598 1599]]");

    // Exactly at the threshold nothing is hidden; one more element starts shortening.
    lazuli::matrix<double> fits(40, 25);
    REQUIRE(std::format("{}", fits).find("...") == std::string::npos);
    lazuli::matrix<double> over(41, 25);
    REQUIRE(std::format("{}", over).find("...") != std::string::npos);
}

TEST_CASE("format: a long vector is shortened the same way") {
    lazuli::vector<double> values(1001);
    for (std::size_t index = 0; index < values.size(); ++index) {
        values[index] = static_cast<double>(index);
    }
    REQUIRE(std::format("{}", values) == "[0 1 2 ... 998 999 1000]");

    lazuli::vector<double> fits(1000);
    REQUIRE(std::format("{}", fits).find("...") == std::string::npos);
}
