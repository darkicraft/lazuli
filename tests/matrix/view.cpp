#include <cstddef>
#include <type_traits>

#include <lazuli/lazuli.hpp>

#include <catch2/catch_message.hpp>
#include <catch2/catch_test_macros.hpp>

namespace {

constexpr std::size_t row_count = 3;
constexpr std::size_t col_count = 4;

struct grid {
    double values[row_count * col_count] = {
        0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0};
};

lazuli::matrix_view<double> mutable_view(grid& g) {
    return lazuli::matrix_view<double>{g.values, row_count, col_count, col_count, 1};
}

lazuli::matrix_view<const double> const_view(const grid& g) {
    return lazuli::matrix_view<const double>{g.values, row_count, col_count, col_count, 1};
}

TEST_CASE("view: default is empty") {
    const lazuli::matrix_view<double> view;
    REQUIRE(view.rows() == std::size_t{0});
    REQUIRE(view.cols() == std::size_t{0});
    REQUIRE(view.size() == std::size_t{0});
    INFO("must be empty");
    REQUIRE(view.empty());
    INFO("must not point anywhere");
    REQUIRE(view.data() == nullptr);
    REQUIRE(view.row_stride() == std::size_t{0});
    REQUIRE(view.col_stride() == std::size_t{0});
}
TEST_CASE("view: geometry from constructor") {
    grid g;
    const auto view = mutable_view(g);
    REQUIRE(view.rows() == row_count);
    REQUIRE(view.cols() == col_count);
    REQUIRE(view.size() == row_count * col_count);
    INFO("must not be empty");
    REQUIRE(!view.empty());
    INFO("must point at the caller's buffer");
    REQUIRE(view.data() == g.values);
    REQUIRE(view.row_stride() == col_count);
    REQUIRE(view.col_stride() == std::size_t{1});
}
TEST_CASE("view: unchecked access reaches corners") {
    grid g;
    const auto view = const_view(g);
    REQUIRE(view(0, 0) == 0.0);
    REQUIRE(view(0, col_count - 1) == 3.0);
    REQUIRE(view(row_count - 1, 0) == 8.0);
    REQUIRE(view(row_count - 1, col_count - 1) == 11.0);
    REQUIRE(view(1, 2) == 6.0);
}
TEST_CASE("view: checked access accepts boundary") {
    grid g;
    const auto view = const_view(g);
    REQUIRE(view.at(0, 0) == 0.0);
    REQUIRE(view.at(row_count - 1, col_count - 1) == 11.0);
}
TEST_CASE("view: checked access rejects past end") {
    grid g;
    const auto view = const_view(g);
    REQUIRE_THROWS_AS(view.at(row_count, 0), lazuli::out_of_bounds);
    REQUIRE_THROWS_AS(view.at(0, col_count), lazuli::out_of_bounds);
    REQUIRE_THROWS_AS(view.at(row_count, col_count), lazuli::out_of_bounds);
}
TEST_CASE("view: empty view rejects indices") {
    const lazuli::matrix_view<double> view;
    REQUIRE_THROWS_AS(view.at(0, 0), lazuli::out_of_bounds);
    REQUIRE_THROWS_AS(view.row(0), lazuli::out_of_bounds);
    REQUIRE_THROWS_AS(view.col(0), lazuli::out_of_bounds);
}
TEST_CASE("view: full-size block is the view") {
    grid g;
    const auto view = const_view(g);
    const auto block = view.block(0, 0, row_count, col_count);
    REQUIRE(block.rows() == row_count);
    REQUIRE(block.cols() == col_count);
    for (std::size_t r = 0; r < row_count; ++r) {
        for (std::size_t c = 0; c < col_count; ++c) {
            REQUIRE(block(r, c) == view(r, c));
        }
    }
}
TEST_CASE("view: block maps a window") {
    grid g;
    const auto view = const_view(g);
    const auto block = view.block(1, 1, 2, 2);
    REQUIRE(block.rows() == std::size_t{2});
    REQUIRE(block.cols() == std::size_t{2});
    REQUIRE(block(0, 0) == 5.0);
    REQUIRE(block(1, 1) == 10.0);
    REQUIRE(block.row_stride() == col_count);
    REQUIRE(block.col_stride() == std::size_t{1});
}
TEST_CASE("view: single row, column and cell blocks") {
    grid g;
    const auto view = const_view(g);
    REQUIRE(view.block(0, 0, 1, col_count).size() == col_count);
    REQUIRE(view.block(0, 0, row_count, 1).size() == row_count);
    REQUIRE(view.block(2, 3, 1, 1)(0, 0) == 11.0);
}
TEST_CASE("view: empty block at the boundary") {
    grid g;
    const auto view = const_view(g);
    const auto empty = view.block(row_count, col_count, 0, 0);
    REQUIRE(empty.size() == std::size_t{0});
    INFO("must be empty");
    REQUIRE(empty.empty());
}
TEST_CASE("view: blocks cannot overhang") {
    grid g;
    const auto view = const_view(g);
    REQUIRE_THROWS_AS(view.block(1, 0, row_count, 1), lazuli::out_of_bounds);
    REQUIRE_THROWS_AS(view.block(0, 1, 1, col_count), lazuli::out_of_bounds);
    REQUIRE_THROWS_AS(view.block(row_count, 0, 1, 1), lazuli::out_of_bounds);
    REQUIRE_THROWS_AS(view.block(0, col_count, 1, 1), lazuli::out_of_bounds);
}
TEST_CASE("view: nested blocks compose") {
    grid g;
    const auto view = mutable_view(g);
    const auto outer = view.block(1, 1, 2, 3);
    const auto inner = outer.block(1, 1, 1, 2);
    REQUIRE(inner(0, 0) == 10.0);
    REQUIRE(inner(0, 1) == 11.0);
}
TEST_CASE("view: rows map one row each") {
    grid g;
    const auto view = const_view(g);
    REQUIRE(view.row(0).rows() == std::size_t{1});
    REQUIRE(view.row(0).cols() == col_count);
    REQUIRE(view.row(2)(0, 0) == 8.0);
    REQUIRE(view.row(2)(0, col_count - 1) == 11.0);
    REQUIRE_THROWS_AS(view.row(row_count), lazuli::out_of_bounds);
}
TEST_CASE("view: columns map one column each") {
    grid g;
    const auto view = const_view(g);
    REQUIRE(view.col(0).rows() == row_count);
    REQUIRE(view.col(0).cols() == std::size_t{1});
    REQUIRE(view.col(0)(0, 0) == 0.0);
    REQUIRE(view.col(3)(row_count - 1, 0) == 11.0);
    REQUIRE_THROWS_AS(view.col(col_count), lazuli::out_of_bounds);
}
TEST_CASE("view: transpose swaps shape and strides") {
    grid g;
    const auto view = const_view(g);
    const auto transposed = view.transpose();
    REQUIRE(transposed.rows() == col_count);
    REQUIRE(transposed.cols() == row_count);
    REQUIRE(transposed.row_stride() == std::size_t{1});
    REQUIRE(transposed.col_stride() == col_count);
}
TEST_CASE("view: transpose reads every element") {
    grid g;
    const auto view = const_view(g);
    const auto transposed = view.transpose();
    for (std::size_t r = 0; r < row_count; ++r) {
        for (std::size_t c = 0; c < col_count; ++c) {
            REQUIRE(transposed(c, r) == view(r, c));
        }
    }
}
TEST_CASE("view: transpose of transpose is original") {
    grid g;
    const auto view = const_view(g);
    const auto twice = view.transpose().transpose();
    REQUIRE(twice.rows() == row_count);
    REQUIRE(twice.cols() == col_count);
    REQUIRE(twice(2, 3) == 11.0);
}
TEST_CASE("view: transpose of a block") {
    grid g;
    const auto view = const_view(g);
    const auto block = view.block(0, 1, 2, 2).transpose();
    REQUIRE(block.rows() == std::size_t{2});
    REQUIRE(block(0, 0) == 1.0);
    REQUIRE(block(1, 1) == 6.0);
}
TEST_CASE("view: writing reaches the buffer") {
    grid g;
    auto view = mutable_view(g);
    view(1, 2) = 42.0;
    REQUIRE(g.values[1 * col_count + 2] == 42.0);
    REQUIRE(view.at(1, 2) == 42.0);
}
TEST_CASE("view: writing through a block") {
    grid g;
    auto view = mutable_view(g);
    view.block(1, 1, 2, 2)(1, 1) = 99.0;
    REQUIRE(g.values[2 * col_count + 2] == 99.0);
}
TEST_CASE("view: writing through a row") {
    grid g;
    auto view = mutable_view(g);
    view.row(2)(0, 3) = 77.0;
    REQUIRE(g.values[2 * col_count + 3] == 77.0);
}
TEST_CASE("view: writing through a column") {
    grid g;
    auto view = mutable_view(g);
    view.col(3)(2, 0) = 55.0;
    REQUIRE(g.values[2 * col_count + 3] == 55.0);
}
TEST_CASE("view: writing through a transpose") {
    grid g;
    auto view = mutable_view(g);
    view.transpose()(3, 2) = 66.0;
    REQUIRE(g.values[2 * col_count + 3] == 66.0);
}
TEST_CASE("view: const views expose const elements") {
    grid g;
    const auto view = const_view(g);
    static_assert(std::is_const_v<std::remove_reference_t<decltype(view(0, 0))>>);
    static_assert(std::is_const_v<std::remove_reference_t<decltype(view.at(0, 0))>>);
    static_assert(std::is_same_v<decltype(view.data()), const double*>);
    REQUIRE(view(0, 0) == 0.0);
}
TEST_CASE("view: empty axes report zero size") {
    double value = 5.0;
    const lazuli::matrix_view<double> no_columns{&value, 0, 4, 4, 1};
    const lazuli::matrix_view<double> no_rows{&value, 4, 0, 0, 1};
    REQUIRE(no_columns.size() == std::size_t{0});
    REQUIRE(no_rows.size() == std::size_t{0});
    INFO("must be empty");
    REQUIRE(no_columns.empty());
    INFO("must be empty");
    REQUIRE(no_rows.empty());
}

}  // namespace
