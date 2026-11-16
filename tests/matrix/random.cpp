#include <catch2/catch_test_macros.hpp>

#include <complex>
#include <cstddef>
#include <cstdint>

#include <lazuli/lazuli.hpp>

TEST_CASE("random: the same seed reproduces the matrix") {
    const auto first = lazuli::matrix<double>::random(16, 16, 12345, {-1.0, 1.0});
    const auto second = lazuli::matrix<double>::random(16, 16, 12345, {-1.0, 1.0});
    REQUIRE(first.rows() == 16);
    REQUIRE(first.cols() == 16);
    for (std::size_t row = 0; row < 16; ++row) {
        for (std::size_t column = 0; column < 16; ++column) {
            REQUIRE(first(row, column) == second(row, column));
        }
    }
}

TEST_CASE("random: different seeds produce different matrices") {
    const auto first = lazuli::matrix<double>::random(8, 8, 1, {-1.0, 1.0});
    const auto second = lazuli::matrix<double>::random(8, 8, 2, {-1.0, 1.0});
    bool differ = false;
    for (std::size_t row = 0; row < 8 && !differ; ++row) {
        for (std::size_t column = 0; column < 8; ++column) {
            if (first(row, column) != second(row, column)) {
                differ = true;
                break;
            }
        }
    }
    REQUIRE(differ);
}

TEST_CASE("random: values stay inside the bounds") {
    const auto values = lazuli::matrix<double>::random(32, 32, 7, {-2.5, 3.5});
    for (std::size_t row = 0; row < values.rows(); ++row) {
        for (std::size_t column = 0; column < values.cols(); ++column) {
            REQUIRE(values(row, column) >= -2.5);
            REQUIRE(values(row, column) <= 3.5);
        }
    }
}

TEST_CASE("random: float bounds are respected") {
    const auto values = lazuli::matrix<float>::random(16, 16, 9, {0.0F, 1.0F});
    for (std::size_t row = 0; row < values.rows(); ++row) {
        for (std::size_t column = 0; column < values.cols(); ++column) {
            REQUIRE(values(row, column) >= 0.0F);
            REQUIRE(values(row, column) <= 1.0F);
        }
    }
}

TEST_CASE("random: complex values sample both parts") {
    const auto values = lazuli::matrix<std::complex<double>>::random(16, 16, 11, {-1.0, 1.0});
    bool any_imaginary = false;
    for (std::size_t row = 0; row < values.rows(); ++row) {
        for (std::size_t column = 0; column < values.cols(); ++column) {
            const auto value = values(row, column);
            REQUIRE(value.real() >= -1.0);
            REQUIRE(value.real() <= 1.0);
            REQUIRE(value.imag() >= -1.0);
            REQUIRE(value.imag() <= 1.0);
            if (value.imag() != 0.0) {
                any_imaginary = true;
            }
        }
    }
    REQUIRE(any_imaginary);
}

TEST_CASE("random: degenerate shapes stay empty") {
    const auto none = lazuli::matrix<double>::random(0, 0, 1, {-1.0, 1.0});
    const auto no_columns = lazuli::matrix<double>::random(4, 0, 1, {-1.0, 1.0});
    const auto no_rows = lazuli::matrix<double>::random(0, 4, 1, {-1.0, 1.0});
    REQUIRE(none.empty());
    REQUIRE(no_columns.empty());
    REQUIRE(no_rows.empty());
    REQUIRE(no_columns.rows() == 4);
    REQUIRE(no_rows.cols() == 4);
}
