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

TEST_CASE("expected: in-place construction, emplace and assignment") {
    lazuli::expected<std::string, int> value{std::in_place, std::size_t{3}, 'x'};
    REQUIRE(*value == "xxx");

    value.emplace(std::size_t{2}, 'y');
    REQUIRE(*value == "yy");

    lazuli::expected<std::string, int> error{lazuli::unexpect, 5};
    REQUIRE(!error);
    REQUIRE(error.error() == 5);

    value = lazuli::unexpected{7};
    REQUIRE(!value);
    REQUIRE(value.error() == 7);

    value = std::string{"z"};
    REQUIRE(value.has_value());
    REQUIRE(*value == "z");
}

TEST_CASE("expected: comparisons") {
    const lazuli::expected<int, int> value{3};
    const lazuli::expected<int, int> same{3};
    const lazuli::expected<int, int> other{4};
    const lazuli::expected<int, int> error{lazuli::unexpected{7}};
    const lazuli::expected<int, int> same_error{lazuli::unexpected{7}};

    REQUIRE(value == same);
    REQUIRE(value != other);
    REQUIRE(value != error);
    REQUIRE(error == same_error);
    REQUIRE(error == lazuli::unexpected{7});
    REQUIRE(lazuli::unexpected{7} == error);
    REQUIRE(error != lazuli::unexpected{8});
    REQUIRE(lazuli::unexpected{7} == lazuli::unexpected{7});
}

TEST_CASE("expected: swap") {
    lazuli::expected<int, int> left{1};
    lazuli::expected<int, int> right{lazuli::unexpected{2}};

    left.swap(right);
    REQUIRE(!left);
    REQUIRE(left.error() == 2);
    REQUIRE(right.has_value());
    REQUIRE(*right == 1);

    lazuli::expected<int, int> other{3};
    using std::swap;
    swap(left, other);
    REQUIRE(left.has_value());
    REQUIRE(*left == 3);
    REQUIRE(!other);
    REQUIRE(other.error() == 2);

    lazuli::unexpected<int> first{4};
    lazuli::unexpected<int> second{5};
    first.swap(second);
    REQUIRE(first.error() == 5);
    REQUIRE(second.error() == 4);
}

TEST_CASE("expected: transform and and_then") {
    const lazuli::expected<int, int> value{3};
    const lazuli::expected<int, int> error{lazuli::unexpected{5}};

    const auto doubled = value.transform([](int input) { return input * 2; });
    REQUIRE(doubled.has_value());
    REQUIRE(*doubled == 6);

    const auto failed = error.transform([](int input) { return input * 2; });
    REQUIRE(!failed);
    REQUIRE(failed.error() == 5);

    const auto chained =
        value.and_then([](int input) { return lazuli::expected<int, int>{input + 1}; });
    REQUIRE(chained.has_value());
    REQUIRE(*chained == 4);

    const auto chained_error =
        error.and_then([](int input) { return lazuli::expected<int, int>{input + 1}; });
    REQUIRE(!chained_error);
    REQUIRE(chained_error.error() == 5);

    int observed = 0;
    const auto ignored = value.transform([&observed](int input) { observed = input; });
    REQUIRE(ignored.has_value());
    REQUIRE(observed == 3);
}

TEST_CASE("expected: or_else and transform_error") {
    const lazuli::expected<int, int> value{3};
    const lazuli::expected<int, int> error{lazuli::unexpected{5}};

    const auto recovered =
        error.or_else([](int reason) { return lazuli::expected<int, int>{reason * 10}; });
    REQUIRE(recovered.has_value());
    REQUIRE(*recovered == 50);

    const auto kept =
        value.or_else([](int reason) { return lazuli::expected<int, int>{reason * 10}; });
    REQUIRE(kept.has_value());
    REQUIRE(*kept == 3);

    const auto changed = error.transform_error([](int reason) { return reason + 1; });
    REQUIRE(!changed);
    REQUIRE(changed.error() == 6);

    const auto unchanged = value.transform_error([](int reason) { return reason + 1; });
    REQUIRE(unchanged.has_value());
    REQUIRE(*unchanged == 3);
}

TEST_CASE("expected: void monadic operations") {
    const lazuli::expected<void, int> success;
    const lazuli::expected<void, int> failure{lazuli::unexpected{4}};

    const auto mapped = success.transform([] { return 7; });
    REQUIRE(mapped.has_value());
    REQUIRE(*mapped == 7);

    const auto mapped_failure = failure.transform([] { return 7; });
    REQUIRE(!mapped_failure);
    REQUIRE(mapped_failure.error() == 4);

    const auto chained = success.and_then([] { return lazuli::expected<int, int>{8}; });
    REQUIRE(chained.has_value());
    REQUIRE(*chained == 8);

    const auto recovered_failure =
        failure.or_else([](int) { return lazuli::expected<void, int>{}; });
    REQUIRE(recovered_failure.has_value());

    const auto changed_failure = failure.transform_error([](int reason) { return reason + 1; });
    REQUIRE(!changed_failure);
    REQUIRE(changed_failure.error() == 5);

    lazuli::expected<void, int> emplaced{lazuli::unexpect, 3};
    REQUIRE(emplaced.error() == 3);
    emplaced.emplace();
    REQUIRE(emplaced.has_value());

    REQUIRE(success == lazuli::expected<void, int>{});
    REQUIRE(failure == lazuli::unexpected{4});
}
