#include <catch2/catch_test_macros.hpp>

#include <cstddef>

#include <lazuli/lazuli.hpp>

namespace {

bool all_equal(const lazuli::matrix<double>& matrix, double value) {
    for (std::size_t row = 0; row < matrix.rows(); ++row) {
        for (std::size_t column = 0; column < matrix.cols(); ++column) {
            if (matrix(row, column) != value) {
                return false;
            }
        }
    }
    return true;
}

}  // namespace

TEST_CASE("fill: a value fills every element") {
    lazuli::matrix<double> matrix(3, 4);
    matrix.fill(2.5);
    REQUIRE(all_equal(matrix, 2.5));
}

TEST_CASE("fill: the generator sees the row and the column") {
    lazuli::matrix<double> matrix(3, 4);
    matrix.fill(
        [](std::size_t row, std::size_t column) { return static_cast<double>(row * 10 + column); });
    for (std::size_t row = 0; row < matrix.rows(); ++row) {
        for (std::size_t column = 0; column < matrix.cols(); ++column) {
            REQUIRE(matrix(row, column) == static_cast<double>(row * 10 + column));
        }
    }
}

TEST_CASE("fill: a block only touches the block") {
    lazuli::matrix<double> matrix(4, 4);
    matrix.block(1, 1, 2, 2).fill(7.0);
    REQUIRE(matrix(0, 0) == 0.0);
    REQUIRE(matrix(1, 1) == 7.0);
    REQUIRE(matrix(2, 2) == 7.0);
    REQUIRE(matrix(3, 3) == 0.0);
}

TEST_CASE("fill: a transposed view fills the original transposed") {
    lazuli::matrix<double> matrix(2, 3);
    matrix.transpose().fill(9.0);
    REQUIRE(all_equal(matrix, 9.0));
}

TEST_CASE("fill: a column view fills only that column") {
    lazuli::matrix<double> matrix{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};
    matrix.col(1).fill(0.0);
    REQUIRE(matrix(0, 0) == 1.0);
    REQUIRE(matrix(0, 1) == 0.0);
    REQUIRE(matrix(1, 1) == 0.0);
    REQUIRE(matrix(1, 2) == 6.0);
}

TEST_CASE("fill: empty matrices are a no-op") {
    lazuli::matrix<double> none(0, 0);
    lazuli::matrix<double> no_columns(3, 0);
    lazuli::matrix<double> no_rows(0, 3);
    none.fill(1.0);
    no_columns.fill(1.0);
    no_rows.fill(1.0);
    REQUIRE(none.empty());
    REQUIRE(no_columns.empty());
    REQUIRE(no_rows.empty());
}

TEST_CASE("fill: vectors fill by value and by generator") {
    lazuli::vector<double> values(4);
    values.fill(1.5);
    for (std::size_t index = 0; index < values.size(); ++index) {
        REQUIRE(values[index] == 1.5);
    }

    values.fill([](std::size_t index) { return static_cast<double>(index); });
    for (std::size_t index = 0; index < values.size(); ++index) {
        REQUIRE(values[index] == static_cast<double>(index));
    }
}

TEST_CASE("fill: mutable views are fillable") {
    // `matrix_view<const T>::fill` is excluded by a constraint, so the read-only spelling
    // cannot be written to; both GCC and Clang reject a requires-expression that probes the
    // negative case, so only the positive side is asserted here.
    static_assert(requires(lazuli::matrix_view<double> view) { view.fill(0.0); });
    REQUIRE(true);
}
