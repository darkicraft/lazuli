#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <utility>

#include <lazuli/lazuli.hpp>

#include <catch2/catch_message.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <initializer_list>

namespace {

constexpr std::uintptr_t cache_line = 64;

template <typename T>
void sized_matrix_is_zero_filled() {
    const lazuli::matrix<T> matrix(3, 4);
    REQUIRE(matrix.rows() == std::size_t{3});
    REQUIRE(matrix.cols() == std::size_t{4});
    REQUIRE(matrix.size() == std::size_t{12});
    for (std::size_t r = 0; r < matrix.rows(); ++r) {
        for (std::size_t c = 0; c < matrix.cols(); ++c) {
            INFO("must be zero filled");
            REQUIRE(matrix(r, c) == T{});
        }
    }
}

template <typename T>
void scalar_type_round_trips() {
    lazuli::matrix<T> matrix(2, 3);
    matrix(0, 0) = T{1};
    matrix(1, 2) = T{6};
    INFO("element write");
    REQUIRE(matrix(0, 0) == T{1});
    INFO("checked read");
    REQUIRE(matrix.at(1, 2) == T{6});
    const auto block = matrix.block(1, 0, 1, 3);
    INFO("block read");
    REQUIRE(block(0, 2) == T{6});
    INFO("block size");
    REQUIRE(block.size() == std::size_t{3});
}

TEST_CASE("matrix<float>: zero filled") {
    sized_matrix_is_zero_filled<float>();
}
TEST_CASE("matrix<double>: zero filled") {
    sized_matrix_is_zero_filled<double>();
}
TEST_CASE("matrix: default is empty") {
    const lazuli::matrix<double> matrix;
    REQUIRE(matrix.rows() == std::size_t{0});
    REQUIRE(matrix.cols() == std::size_t{0});
    REQUIRE(matrix.size() == std::size_t{0});
    INFO("must be empty");
    REQUIRE(matrix.empty());
    INFO("must not own storage");
    REQUIRE(matrix.data() == nullptr);
    REQUIRE(matrix.row_stride() == std::size_t{0});
    REQUIRE(matrix.col_stride() == std::size_t{1});
}
TEST_CASE("matrix: degenerate shapes are empty") {
    for (const auto& shape : {std::pair<std::size_t, std::size_t>{0, 0},
                              std::pair<std::size_t, std::size_t>{0, 5},
                              std::pair<std::size_t, std::size_t>{5, 0}}) {
        const lazuli::matrix<double> matrix(shape.first, shape.second);
        REQUIRE(matrix.rows() == shape.first);
        REQUIRE(matrix.cols() == shape.second);
        REQUIRE(matrix.size() == std::size_t{0});
        INFO("must be empty");
        REQUIRE(matrix.empty());
        INFO("must not own storage");
        REQUIRE(matrix.data() == nullptr);
    }
}
TEST_CASE("matrix: uninitialized has shape") {
    const auto matrix = lazuli::matrix<double>::uninitialized(2, 5);
    REQUIRE(matrix.rows() == std::size_t{2});
    REQUIRE(matrix.cols() == std::size_t{5});
    INFO("must own storage");
    REQUIRE(matrix.data() != nullptr);
    INFO("must not be empty");
    REQUIRE(!matrix.empty());
    INFO("empty uninitialized");
    REQUIRE(lazuli::matrix<double>::uninitialized(0, 5).empty());
}
TEST_CASE("matrix: initializer list fills row major") {
    const lazuli::matrix<double> matrix{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};
    REQUIRE(matrix.rows() == std::size_t{2});
    REQUIRE(matrix.cols() == std::size_t{3});
    REQUIRE(matrix(0, 0) == 1.0);
    REQUIRE(matrix(0, 2) == 3.0);
    REQUIRE(matrix(1, 0) == 4.0);
    REQUIRE(matrix(1, 2) == 6.0);
}
TEST_CASE("matrix: single cell, row and column lists") {
    const lazuli::matrix<double> cell{{7.0}};
    REQUIRE(cell.rows() == std::size_t{1});
    REQUIRE(cell.cols() == std::size_t{1});
    REQUIRE(cell(0, 0) == 7.0);

    const lazuli::matrix<double> row{{1.0, 2.0, 3.0}};
    REQUIRE(row.rows() == std::size_t{1});
    REQUIRE(row.cols() == std::size_t{3});

    const lazuli::matrix<double> column{{1.0}, {2.0}, {3.0}};
    REQUIRE(column.rows() == std::size_t{3});
    REQUIRE(column.cols() == std::size_t{1});
}
TEST_CASE("matrix: absent initializer list") {
    const lazuli::matrix<double> matrix(std::initializer_list<std::initializer_list<double>>{});
    REQUIRE(matrix.rows() == std::size_t{0});
    REQUIRE(matrix.cols() == std::size_t{0});
    INFO("must be empty");
    REQUIRE(matrix.empty());
}
TEST_CASE("matrix: ragged list is rejected") {
    REQUIRE_THROWS_WITH(([] {
                            const lazuli::matrix<double> matrix{{1.0, 2.0}, {3.0}};
                            (void)matrix;
                        })(),
                        Catch::Matchers::ContainsSubstring("ragged"));
}
TEST_CASE("matrix: access reaches every corner") {
    lazuli::matrix<double> matrix(3, 4);
    matrix(0, 0) = 1.0;
    matrix(0, 3) = 2.0;
    matrix(2, 0) = 3.0;
    matrix(2, 3) = 4.0;
    matrix(1, 2) = 5.0;
    REQUIRE(matrix(0, 0) == 1.0);
    REQUIRE(matrix(0, 3) == 2.0);
    REQUIRE(matrix(2, 0) == 3.0);
    REQUIRE(matrix(2, 3) == 4.0);
    REQUIRE(matrix(1, 2) == 5.0);

    const auto& read_only = matrix;
    REQUIRE(read_only(1, 2) == 5.0);
}
TEST_CASE("matrix: checked access accepts boundary") {
    const lazuli::matrix<double> matrix(3, 4);
    REQUIRE(matrix.at(0, 0) == 0.0);
    REQUIRE(matrix.at(2, 3) == 0.0);
}
TEST_CASE("matrix: checked access rejects past end") {
    const lazuli::matrix<double> matrix(3, 4);
    REQUIRE_THROWS_AS(matrix.at(3, 0), lazuli::out_of_bounds);
    REQUIRE_THROWS_AS(matrix.at(0, 4), lazuli::out_of_bounds);
    REQUIRE_THROWS_WITH(matrix.at(9, 0), Catch::Matchers::ContainsSubstring("9"));
}
TEST_CASE("matrix: strides describe layout") {
    const lazuli::matrix<double> matrix(3, 4);
    REQUIRE(matrix.row_stride() == std::size_t{4});
    REQUIRE(matrix.col_stride() == std::size_t{1});
    const lazuli::matrix<double> tall(5, 2);
    REQUIRE(tall.row_stride() == std::size_t{2});
}
TEST_CASE("matrix: copy construction is deep") {
    const lazuli::matrix<double> original{{1.0, 2.0}, {3.0, 4.0}};
    lazuli::matrix<double> copy{original};
    REQUIRE(copy.rows() == original.rows());
    REQUIRE(copy(1, 1) == 4.0);
    copy(0, 0) = 9.0;
    REQUIRE(original(0, 0) == 1.0);
    INFO("must not share storage");
    REQUIRE(copy.data() != original.data());
}
TEST_CASE("matrix: copying empty stays empty") {
    const lazuli::matrix<double> empty;
    const lazuli::matrix<double> copy{empty};
    INFO("must stay empty");
    REQUIRE(copy.empty());
    INFO("must not own storage");
    REQUIRE(copy.data() == nullptr);
}
TEST_CASE("matrix: copy assignment is deep") {
    const lazuli::matrix<double> original{{1.0, 2.0}, {3.0, 4.0}};
    lazuli::matrix<double> target(3, 3);
    target = original;
    REQUIRE(target.rows() == std::size_t{2});
    REQUIRE(target.cols() == std::size_t{2});
    REQUIRE(target(1, 1) == 4.0);
    target(0, 0) = 9.0;
    REQUIRE(original(0, 0) == 1.0);
}
TEST_CASE("matrix: self copy assignment") {
    lazuli::matrix<double> matrix{{1.0, 2.0}};
    lazuli::matrix<double>& alias = matrix;
    matrix = alias;
    REQUIRE(matrix.rows() == std::size_t{1});
    REQUIRE(matrix(0, 1) == 2.0);
}
TEST_CASE("matrix: move construction empties source") {
    lazuli::matrix<double> source{{1.0, 2.0}};
    double* const address = source.data();
    lazuli::matrix<double> moved{std::move(source)};
    INFO("storage must move");
    REQUIRE(moved.data() == address);
    REQUIRE(moved(0, 1) == 2.0);
    INFO("source must be empty");
    REQUIRE(source.empty());
    REQUIRE(source.data() == nullptr);
}
TEST_CASE("matrix: move assignment empties source") {
    lazuli::matrix<double> source{{1.0, 2.0}};
    lazuli::matrix<double> target(3, 3);
    target = std::move(source);
    REQUIRE(target.rows() == std::size_t{1});
    REQUIRE(target(0, 1) == 2.0);
    INFO("source must be empty");
    REQUIRE(source.empty());
}
TEST_CASE("matrix: self move assignment") {
    lazuli::matrix<double> matrix{{1.0, 2.0}};
    lazuli::matrix<double>& alias = matrix;
    matrix = std::move(alias);
    REQUIRE(matrix.rows() == std::size_t{1});
    REQUIRE(matrix(0, 1) == 2.0);
}
TEST_CASE("matrix: swap exchanges contents") {
    lazuli::matrix<double> left{{1.0}};
    lazuli::matrix<double> right{{2.0, 3.0}};
    left.swap(right);
    REQUIRE(left.rows() == std::size_t{1});
    REQUIRE(left.cols() == std::size_t{2});
    REQUIRE(left(0, 0) == 2.0);
    REQUIRE(right(0, 0) == 1.0);

    lazuli::matrix<double> empty;
    left.swap(empty);
    INFO("left must take the empty state");
    REQUIRE(left.empty());
    REQUIRE(empty.cols() == std::size_t{2});
}
TEST_CASE("matrix: view exposes the matrix") {
    lazuli::matrix<double> matrix(2, 3);
    auto view = matrix.view();
    REQUIRE(view.rows() == matrix.rows());
    REQUIRE(view.cols() == matrix.cols());
    INFO("must share storage");
    REQUIRE(view.data() == matrix.data());
    REQUIRE(view.row_stride() == matrix.row_stride());
    view(1, 2) = 5.0;
    REQUIRE(matrix(1, 2) == 5.0);
}
TEST_CASE("matrix: const matrix yields const view") {
    const lazuli::matrix<double> matrix{{1.0}};
    const auto view = matrix.view();
    static_assert(std::is_const_v<std::remove_reference_t<decltype(view(0, 0))>>);
    REQUIRE(view(0, 0) == 1.0);
}
TEST_CASE("matrix: implicit conversions") {
    static_assert(std::is_convertible_v<lazuli::matrix<double>&, lazuli::matrix_view<double>>);
    static_assert(
        std::is_convertible_v<const lazuli::matrix<double>&, lazuli::matrix_view<const double>>);
    static_assert(
        !std::is_convertible_v<const lazuli::matrix<double>&, lazuli::matrix_view<double>>);

    lazuli::matrix<double> matrix{{1.0, 2.0}};
    lazuli::matrix_view<double> mutable_view = matrix;
    mutable_view(0, 0) = 3.0;
    REQUIRE(matrix(0, 0) == 3.0);

    const lazuli::matrix<double>& read_only = matrix;
    lazuli::matrix_view<const double> const_view = read_only;
    REQUIRE(const_view(0, 1) == 2.0);
}
TEST_CASE("matrix: windows are views") {
    lazuli::matrix<double> matrix{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};
    auto block = matrix.block(0, 1, 2, 2);
    REQUIRE(block.rows() == std::size_t{2});
    REQUIRE(block(0, 0) == 2.0);
    REQUIRE(block(1, 1) == 6.0);

    auto row = matrix.row(1);
    REQUIRE(row.cols() == std::size_t{3});
    REQUIRE(row(0, 2) == 6.0);

    auto column = matrix.col(2);
    REQUIRE(column.rows() == std::size_t{2});
    REQUIRE(column(0, 0) == 3.0);

    auto transposed = matrix.transpose();
    REQUIRE(transposed.rows() == std::size_t{3});
    REQUIRE(transposed(2, 1) == 6.0);
    transposed(2, 0) = 9.0;
    REQUIRE(matrix(0, 2) == 9.0);
}
TEST_CASE("matrix: window bounds are checked") {
    lazuli::matrix<double> matrix(2, 2);
    REQUIRE_THROWS_AS(matrix.block(1, 1, 2, 2), lazuli::out_of_bounds);
    REQUIRE_THROWS_AS(matrix.row(2), lazuli::out_of_bounds);
    REQUIRE_THROWS_AS(matrix.col(2), lazuli::out_of_bounds);
}
TEST_CASE("matrix<float>: scalar round trip") {
    scalar_type_round_trips<float>();
}
TEST_CASE("matrix<double>: scalar round trip") {
    scalar_type_round_trips<double>();
}

TEST_CASE("matrix: storage is cache line aligned") {
    for (const auto& shape : {std::pair<std::size_t, std::size_t>{1, 1},
                              std::pair<std::size_t, std::size_t>{3, 3},
                              std::pair<std::size_t, std::size_t>{5, 7}}) {
        const lazuli::matrix<float> narrow(shape.first, shape.second);
        const lazuli::matrix<double> wide(shape.first, shape.second);
        INFO("float");
        REQUIRE(reinterpret_cast<std::uintptr_t>(narrow.data()) % cache_line == 0);
        INFO("double");
        REQUIRE(reinterpret_cast<std::uintptr_t>(wide.data()) % cache_line == 0);
    }
}

}  // namespace
