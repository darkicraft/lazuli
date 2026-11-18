#include <cstddef>
#include <cstdint>
#include <iterator>
#include <type_traits>
#include <utility>

#include <lazuli/lazuli.hpp>

#include <catch2/catch_message.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

namespace {

constexpr std::uintptr_t cache_line = 64;

template <typename T>
void sized_vector_is_zero_filled() {
    const lazuli::vector<T> values(3);
    REQUIRE(values.size() == std::size_t{3});
    INFO("must not be empty");
    REQUIRE(!values.empty());
    for (std::size_t i = 0; i < values.size(); ++i) {
        INFO("must be zero filled");
        REQUIRE(values[i] == T{});
    }
}

template <typename T>
void scalar_type_round_trips() {
    lazuli::vector<T> values(2);
    values[0] = T{1};
    values[1] = T{2};
    INFO("write");
    REQUIRE(values[0] == T{1});
    INFO("checked read");
    REQUIRE(values.at(1) == T{2});
    INFO("column read");
    REQUIRE(values.as_column()(1, 0) == T{2});
}

TEST_CASE("vector<float>: zero filled") {
    sized_vector_is_zero_filled<float>();
}
TEST_CASE("vector<double>: zero filled") {
    sized_vector_is_zero_filled<double>();
}
TEST_CASE("vector: default is empty") {
    const lazuli::vector<double> values;
    REQUIRE(values.size() == std::size_t{0});
    INFO("must be empty");
    REQUIRE(values.empty());
    INFO("must not own storage");
    REQUIRE(values.data() == nullptr);
    INFO("iterators must coincide");
    REQUIRE(values.begin() == values.end());
}
TEST_CASE("vector: zero sized has no storage") {
    const lazuli::vector<double> values(0);
    REQUIRE(values.size() == std::size_t{0});
    INFO("must be empty");
    REQUIRE(values.empty());
    INFO("must not own storage");
    REQUIRE(values.data() == nullptr);
}
TEST_CASE("vector: uninitialized has size") {
    const auto values = lazuli::vector<double>::uninitialized(4);
    REQUIRE(values.size() == std::size_t{4});
    INFO("must own storage");
    REQUIRE(values.data() != nullptr);
    INFO("empty uninitialized");
    REQUIRE(lazuli::vector<double>::uninitialized(0).empty());
}
TEST_CASE("vector: initializer list fills values") {
    const lazuli::vector<double> values{1.0, 2.0, 3.0};
    REQUIRE(values.size() == std::size_t{3});
    REQUIRE(values[0] == 1.0);
    REQUIRE(values[2] == 3.0);
}
TEST_CASE("vector: single and absent initializer lists") {
    const lazuli::vector<double> single{7.0};
    REQUIRE(single.size() == std::size_t{1});
    REQUIRE(single[0] == 7.0);

    const lazuli::vector<double> absent{};
    REQUIRE(absent.size() == std::size_t{0});
    INFO("absent must be empty");
    REQUIRE(absent.empty());
}
TEST_CASE("vector: copy construction is deep") {
    const lazuli::vector<double> original{1.0, 2.0};
    lazuli::vector<double> copy{original};
    REQUIRE(copy.size() == std::size_t{2});
    INFO("must not share storage");
    REQUIRE(copy.data() != original.data());
    copy[0] = 9.0;
    REQUIRE(original[0] == 1.0);
}
TEST_CASE("vector: copy assignment is deep") {
    const lazuli::vector<double> original{1.0, 2.0, 3.0};
    lazuli::vector<double> target(1);
    target = original;
    REQUIRE(target.size() == std::size_t{3});
    REQUIRE(target[2] == 3.0);
    target[0] = 9.0;
    REQUIRE(original[0] == 1.0);
}
TEST_CASE("vector: self copy assignment") {
    lazuli::vector<double> values{1.0, 2.0};
    lazuli::vector<double>& alias = values;
    values = alias;
    REQUIRE(values.size() == std::size_t{2});
    REQUIRE(values[1] == 2.0);
}
TEST_CASE("vector: move construction empties source") {
    lazuli::vector<double> source{1.0, 2.0};
    double* const address = source.data();
    lazuli::vector<double> moved{std::move(source)};
    INFO("storage must move");
    REQUIRE(moved.data() == address);
    REQUIRE(moved[1] == 2.0);
    INFO("source must be empty");
    REQUIRE(source.empty());
    REQUIRE(source.data() == nullptr);
}
TEST_CASE("vector: move assignment empties source") {
    lazuli::vector<double> source{1.0, 2.0};
    lazuli::vector<double> target(5);
    target = std::move(source);
    REQUIRE(target.size() == std::size_t{2});
    REQUIRE(target[1] == 2.0);
    INFO("source must be empty");
    REQUIRE(source.empty());
}
TEST_CASE("vector: self move assignment") {
    lazuli::vector<double> values{1.0, 2.0};
    lazuli::vector<double>& alias = values;
    values = std::move(alias);
    REQUIRE(values.size() == std::size_t{2});
    REQUIRE(values[1] == 2.0);
}
TEST_CASE("vector: swap exchanges contents") {
    lazuli::vector<double> left{1.0};
    lazuli::vector<double> right{2.0, 3.0};
    left.swap(right);
    REQUIRE(left.size() == std::size_t{2});
    REQUIRE(left[0] == 2.0);
    REQUIRE(right.size() == std::size_t{1});
    REQUIRE(right[0] == 1.0);
}
TEST_CASE("vector: subscript reaches both ends") {
    lazuli::vector<double> values{1.0, 2.0, 3.0};
    REQUIRE(values[0] == 1.0);
    REQUIRE(values[2] == 3.0);
    values[0] = 4.0;
    values[2] = 5.0;
    REQUIRE(values[0] == 4.0);
    REQUIRE(values[2] == 5.0);

    const auto& read_only = values;
    REQUIRE(read_only[1] == 2.0);
}
TEST_CASE("vector: checked access accepts boundary") {
    const lazuli::vector<double> values{1.0, 2.0, 3.0};
    REQUIRE(values.at(0) == 1.0);
    REQUIRE(values.at(2) == 3.0);
}
TEST_CASE("vector: checked access rejects past end") {
    const lazuli::vector<double> values{1.0, 2.0, 3.0};
    REQUIRE_THROWS_AS(values.at(3), lazuli::out_of_bounds);
    REQUIRE_THROWS_WITH(values.at(8), Catch::Matchers::ContainsSubstring("8"));
    const lazuli::vector<double> empty;
    REQUIRE_THROWS_AS(empty.at(0), lazuli::out_of_bounds);
}
TEST_CASE("vector: iterators cover every element") {
    const lazuli::vector<double> values{1.0, 2.0, 3.0};
    REQUIRE(std::distance(values.begin(), values.end()) == std::ptrdiff_t{3});
    double total = 0.0;
    for (const double value : values) {
        total += value;
    }
    REQUIRE(total == 6.0);

    lazuli::vector<double> writable{0.0, 0.0};
    for (double& value : writable) {
        value = 1.0;
    }
    REQUIRE(writable[0] == 1.0);
    REQUIRE(writable[1] == 1.0);
}
TEST_CASE("vector: column view maps the vector") {
    lazuli::vector<double> values{1.0, 2.0, 3.0};
    auto column = values.as_column();
    REQUIRE(column.rows() == std::size_t{3});
    REQUIRE(column.cols() == std::size_t{1});
    REQUIRE(column.row_stride() == std::size_t{1});
    INFO("must share storage");
    REQUIRE(column.data() == values.data());
    REQUIRE(column(2, 0) == 3.0);
    column(1, 0) = 9.0;
    REQUIRE(values[1] == 9.0);
}
TEST_CASE("vector: const column view is const") {
    const lazuli::vector<double> values{1.0};
    const auto column = values.as_column();
    static_assert(std::is_const_v<std::remove_reference_t<decltype(column(0, 0))>>);
    REQUIRE(column(0, 0) == 1.0);
}
TEST_CASE("vector<float>: scalar round trip") {
    scalar_type_round_trips<float>();
}
TEST_CASE("vector<double>: scalar round trip") {
    scalar_type_round_trips<double>();
}

TEST_CASE("vector: storage is cache line aligned") {
    const lazuli::vector<float> narrow{1.0F};
    const lazuli::vector<double> wide{1.0};
    INFO("float");
    REQUIRE(reinterpret_cast<std::uintptr_t>(narrow.data()) % cache_line == 0);
    INFO("double");
    REQUIRE(reinterpret_cast<std::uintptr_t>(wide.data()) % cache_line == 0);
}

}  // namespace
