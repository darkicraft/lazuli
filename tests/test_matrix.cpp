#include <complex>
#include <cstdint>
#include <iterator>
#include <string_view>
#include <type_traits>
#include <utility>

#include <lazuli/lazuli.hpp>

#include "harness.hpp"

namespace {

using lazuli_test::require;
using lazuli_test::require_equal;
using lazuli_test::throws;

void default_matrix_is_empty() {
    lazuli::matrix<double> m;
    require_equal(m.rows(), std::size_t{0}, "rows");
    require_equal(m.cols(), std::size_t{0}, "cols");
    require(m.empty(), "must be empty");
    require(m.data() == nullptr, "must not own storage");
}

void sized_matrix_is_zero_filled() {
    lazuli::matrix<double> m(3, 4);
    require_equal(m.rows(), std::size_t{3}, "rows");
    require_equal(m.cols(), std::size_t{4}, "cols");
    require_equal(m.size(), std::size_t{12}, "size");
    for (std::size_t r = 0; r < m.rows(); ++r) {
        for (std::size_t c = 0; c < m.cols(); ++c) {
            require_equal(m(r, c), 0.0, "zero fill");
        }
    }
}

void uninitialized_matrix_has_shape() {
    auto m = lazuli::matrix<double>::uninitialized(2, 5);
    require_equal(m.rows(), std::size_t{2}, "rows");
    require_equal(m.cols(), std::size_t{5}, "cols");
    require(m.data() != nullptr, "must own storage");
}

void initializer_list_builds_shape_and_values() {
    lazuli::matrix<double> m{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};
    require_equal(m.rows(), std::size_t{2}, "rows");
    require_equal(m.cols(), std::size_t{3}, "cols");
    require_equal(m(0, 0), 1.0, "first");
    require_equal(m(1, 2), 6.0, "last");
}

void ragged_list_is_rejected() {
    const bool rejected = throws<lazuli::dimension_mismatch>([] {
        lazuli::matrix<double> m{{1.0, 2.0}, {3.0}};
        (void)m;
    });
    require(rejected, "ragged initialiser list must throw dimension_mismatch");
}

void element_access_round_trips() {
    lazuli::matrix<double> m(2, 2);
    m(0, 1) = 4.5;
    require_equal(m(0, 1), 4.5, "value");
    require_equal(m.row_stride(), std::size_t{2}, "row stride");
    require_equal(m.col_stride(), std::size_t{1}, "column stride");
}

void checked_access_is_checked() {
    lazuli::matrix<double> m(2, 2);
    require_equal(m.at(1, 1), 0.0, "in range");
    require(throws<lazuli::out_of_bounds>([&m] { (void)m.at(2, 0); }), "row out of range");
    require(throws<lazuli::out_of_bounds>([&m] { (void)m.at(0, 2); }), "column out of range");
}

void copy_is_deep() {
    lazuli::matrix<double> original{{1.0}};
    lazuli::matrix<double> copy{original};
    copy(0, 0) = 9.0;
    require_equal(original(0, 0), 1.0, "original must be untouched");
}

void move_empties_the_source() {
    lazuli::matrix<double> source{{1.0, 2.0}};
    lazuli::matrix<double> moved{std::move(source)};
    require_equal(moved(0, 1), 2.0, "value must move");
    require(source.empty(), "source must become empty");
    require_equal(source.rows(), std::size_t{0}, "source rows");
}

void assignment_works() {
    lazuli::matrix<double> a{{1.0, 2.0}};
    lazuli::matrix<double> b;
    b = a;
    b(0, 0) = 5.0;
    require_equal(a(0, 0), 1.0, "copy assignment must be deep");

    lazuli::matrix<double> c;
    c = std::move(b);
    require_equal(c(0, 0), 5.0, "move assignment must transfer values");
    require(b.empty(), "move assignment must empty the source");
}

void storage_is_cache_line_aligned() {
    lazuli::matrix<double> real(3, 3);
    lazuli::matrix<std::complex<float>> complex(3, 3);
    const auto alignment = std::uintptr_t{64};
    require(reinterpret_cast<std::uintptr_t>(real.data()) % alignment == 0, "double alignment");
    require(reinterpret_cast<std::uintptr_t>(complex.data()) % alignment == 0, "complex alignment");
}

void block_is_a_window() {
    lazuli::matrix<double> m(4, 4);
    auto block = m.block(1, 1, 2, 2);
    require_equal(block.rows(), std::size_t{2}, "block rows");
    require_equal(block.cols(), std::size_t{2}, "block cols");
    block(0, 0) = 7.0;
    require_equal(m(1, 1), 7.0, "write must be visible through the matrix");
    require_equal(m(0, 0), 0.0, "elements outside the block must be untouched");
}

void block_bounds_are_checked() {
    lazuli::matrix<double> m(4, 4);
    require(throws<lazuli::out_of_bounds>([&m] { (void)m.block(3, 3, 2, 2); }), "block bounds");
    require(throws<lazuli::out_of_bounds>([&m] { (void)m.row(4); }), "row bounds");
    require(throws<lazuli::out_of_bounds>([&m] { (void)m.col(4); }), "column bounds");
}

void transpose_is_a_view() {
    lazuli::matrix<double> m{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};
    auto transposed = m.transpose();
    require_equal(transposed.rows(), std::size_t{3}, "transposed rows");
    require_equal(transposed.cols(), std::size_t{2}, "transposed cols");
    require_equal(transposed(0, 1), 4.0, "transposed read");
    transposed(2, 0) = 9.0;
    require_equal(m(0, 2), 9.0, "write must reach the original element");
}

void row_and_column_views() {
    lazuli::matrix<double> m{{1.0, 2.0}, {3.0, 4.0}};
    auto row = m.row(1);
    require_equal(row.rows(), std::size_t{1}, "row rows");
    require_equal(row.cols(), std::size_t{2}, "row cols");
    require_equal(row(0, 1), 4.0, "row value");

    auto column = m.col(0);
    require_equal(column.rows(), std::size_t{2}, "column rows");
    require_equal(column.cols(), std::size_t{1}, "column cols");
    require_equal(column(1, 0), 3.0, "column value");
}

void const_view_is_read_only() {
    const lazuli::matrix<double> m{{1.0, 2.0}};
    auto view = m.view();
    require_equal(view(0, 1), 2.0, "value");
    static_assert(std::is_const_v<std::remove_reference_t<decltype(view(0, 0))>>);
}

void vector_basics() {
    lazuli::vector<double> v{1.0, 2.0, 3.0};
    require_equal(v.size(), std::size_t{3}, "size");
    require_equal(v[2], 3.0, "subscript");
    require_equal(v.at(0), 1.0, "checked subscript");
    require(throws<lazuli::out_of_bounds>([&v] { (void)v.at(3); }), "vector bounds");
    require_equal(std::distance(v.begin(), v.end()), std::ptrdiff_t{3}, "iterators");

    auto column = v.as_column();
    require_equal(column.rows(), std::size_t{3}, "column rows");
    require_equal(column.cols(), std::size_t{1}, "column cols");
    require_equal(column(2, 0), 3.0, "column value");
}

template <typename T>
void scalar_type_round_trips() {
    lazuli::matrix<T> m(2, 2);
    m(0, 0) = T{2};
    m(1, 1) = T{3};
    require(m(0, 0) == T{2}, "element write");
    require(m.block(0, 0, 2, 2)(1, 1) == T{3}, "block read");

    lazuli::vector<T> v(2);
    v[1] = T{5};
    require(v[1] == T{5}, "vector write");
}

void all_scalar_types_work() {
    scalar_type_round_trips<float>();
    scalar_type_round_trips<double>();
    scalar_type_round_trips<std::complex<float>>();
    scalar_type_round_trips<std::complex<double>>();
}

void error_codes_have_names() {
    require_equal(lazuli::to_string(lazuli::error_code::singular),
                  std::string_view{"singular"},
                  "code spelling");
    require_equal(lazuli::to_string(lazuli::error_code::out_of_bounds),
                  std::string_view{"out_of_bounds"},
                  "code spelling");
}

}  // namespace

int main() {
    std::printf("lazuli unit tests\n");

    int failures = 0;
    failures += lazuli_test::run("default matrix is empty", default_matrix_is_empty);
    failures += lazuli_test::run("sized matrix is zero filled", sized_matrix_is_zero_filled);
    failures += lazuli_test::run("uninitialized matrix has shape", uninitialized_matrix_has_shape);
    failures += lazuli_test::run("initializer list builds shape and values",
                                 initializer_list_builds_shape_and_values);
    failures += lazuli_test::run("ragged list is rejected", ragged_list_is_rejected);
    failures += lazuli_test::run("element access round trips", element_access_round_trips);
    failures += lazuli_test::run("checked access is checked", checked_access_is_checked);
    failures += lazuli_test::run("copy is deep", copy_is_deep);
    failures += lazuli_test::run("move empties the source", move_empties_the_source);
    failures += lazuli_test::run("assignment works", assignment_works);
    failures += lazuli_test::run("storage is cache line aligned", storage_is_cache_line_aligned);
    failures += lazuli_test::run("block is a window", block_is_a_window);
    failures += lazuli_test::run("block bounds are checked", block_bounds_are_checked);
    failures += lazuli_test::run("transpose is a view", transpose_is_a_view);
    failures += lazuli_test::run("row and column views", row_and_column_views);
    failures += lazuli_test::run("const view is read only", const_view_is_read_only);
    failures += lazuli_test::run("vector basics", vector_basics);
    failures += lazuli_test::run("all scalar types work", all_scalar_types_work);
    failures += lazuli_test::run("error codes have names", error_codes_have_names);

    if (failures == 0) {
        std::printf("all tests passed\n");
        return 0;
    }
    std::printf("%d test(s) failed\n", failures);
    return 1;
}
