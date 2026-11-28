#include <lazuli/lazuli.hpp>

#include <catch2/catch_test_macros.hpp>

#include <memory>
#include <string>
#include <utility>

namespace {

struct no_default {
    explicit no_default(int input)
        : value(input) {}
    int value;
};

}  // namespace

TEST_CASE("expected: a value and an error are distinguishable") {
    lazuli::expected<int, int> value{7};
    REQUIRE(value.has_value());
    REQUIRE(static_cast<bool>(value));
    REQUIRE(*value == 7);
    REQUIRE(value.value() == 7);
    REQUIRE(value.value_or(9) == 7);

    lazuli::expected<int, int> error{lazuli::unexpected{42}};
    REQUIRE(!error.has_value());
    REQUIRE(!static_cast<bool>(error));
    REQUIRE(error.error() == 42);
    REQUIRE(error.value_or(9) == 9);
}

TEST_CASE("expected: value throws bad_expected_access on an error") {
    lazuli::expected<int, int> error{lazuli::unexpected{5}};
    REQUIRE_THROWS_AS(error.value(), lazuli::bad_expected_access<int>);
    try {
        (void)error.value();
        FAIL("value() must throw");
    } catch (const lazuli::bad_expected_access<int>& failure) {
        REQUIRE(failure.error() == 5);
        REQUIRE(std::string{failure.what()}.size() > 0);
    }
}

TEST_CASE("expected: operator-> forwards to the value") {
    lazuli::expected<std::string, int> value{std::string{"lazuli"}};
    REQUIRE(value->size() == 6);
    REQUIRE(value->at(0) == 'l');
}

TEST_CASE("expected: copy, move and assignment") {
    lazuli::expected<int, int> value{3};
    lazuli::expected<int, int> copy{value};
    REQUIRE(*copy == 3);

    lazuli::expected<int, int> moved{std::move(copy)};
    REQUIRE(*moved == 3);

    lazuli::expected<int, int> error{lazuli::unexpected{8}};
    moved = error;
    REQUIRE(!moved);
    REQUIRE(moved.error() == 8);

    moved = lazuli::expected<int, int>{11};
    REQUIRE(moved.has_value());
    REQUIRE(*moved == 11);

    lazuli::expected<int, int> assigned{lazuli::unexpected{1}};
    assigned = lazuli::expected<int, int>{lazuli::unexpected{2}};
    REQUIRE(!assigned);
    REQUIRE(assigned.error() == 2);
}

TEST_CASE("expected: works with a move-only value") {
    lazuli::expected<std::unique_ptr<int>, int> value{std::make_unique<int>(7)};
    REQUIRE(value.has_value());
    REQUIRE(**value == 7);

    lazuli::expected<std::unique_ptr<int>, int> moved{std::move(value)};
    REQUIRE(moved.has_value());
    REQUIRE(**moved == 7);
}

TEST_CASE("expected: does not require a default-constructible value") {
    lazuli::expected<no_default, int> value{no_default{13}};
    REQUIRE(value.has_value());
    REQUIRE(value->value == 13);
}

TEST_CASE("expected: the void specialization") {
    lazuli::expected<void, int> success;
    REQUIRE(success.has_value());
    success.value();
    *success;

    lazuli::expected<void, int> failure{lazuli::unexpected{9}};
    REQUIRE(!failure.has_value());
    REQUIRE(failure.error() == 9);
    REQUIRE_THROWS_AS(failure.value(), lazuli::bad_expected_access<int>);

    lazuli::expected<void, int> assigned{lazuli::unexpected{1}};
    assigned = lazuli::expected<void, int>{};
    REQUIRE(assigned.has_value());
    assigned = lazuli::expected<void, int>{lazuli::unexpected{4}};
    REQUIRE(!assigned);
    REQUIRE(assigned.error() == 4);
}

TEST_CASE("expected: works with the library error type") {
    lazuli::expected<int, lazuli::error> failure{
        lazuli::unexpected{lazuli::error{lazuli::error_code::singular, "zero pivot"}}};
    REQUIRE(!failure);
    REQUIRE(failure.error().code == lazuli::error_code::singular);
    REQUIRE(failure.error().message == std::string{"zero pivot"});
}
