#include <cstdint>
#include <iterator>
#include <new>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>

#include <lazuli/lazuli.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

namespace {

struct spelling {
    lazuli::error_code code;
    std::string_view text;
};

constexpr spelling spellings[] = {
    {lazuli::error_code::none, "none"},
    {lazuli::error_code::dimension_mismatch, "dimension_mismatch"},
    {lazuli::error_code::not_square, "not_square"},
    {lazuli::error_code::not_positive_definite, "not_positive_definite"},
    {lazuli::error_code::singular, "singular"},
    {lazuli::error_code::not_converged, "not_converged"},
    {lazuli::error_code::io_parse_error, "io_parse_error"},
    {lazuli::error_code::out_of_bounds, "out_of_bounds"},
};

TEST_CASE("error: every code has a spelling") {
    for (const auto& entry : spellings) {
        REQUIRE(lazuli::to_string(entry.code) == entry.text);
    }
}
TEST_CASE("error: codes are distinct") {
    for (std::size_t i = 0; i < std::size(spellings); ++i) {
        for (std::size_t j = i + 1; j < std::size(spellings); ++j) {
            INFO("codes must be distinct");
            REQUIRE(spellings[i].code != spellings[j].code);
        }
    }
}
TEST_CASE("error: code type is fixed") {
    static_assert(std::is_same_v<std::underlying_type_t<lazuli::error_code>, std::int32_t>);
    static_assert(!std::is_convertible_v<lazuli::error_code, int>);
}
TEST_CASE("error: default is empty") {
    lazuli::error report;
    REQUIRE(report.code == lazuli::error_code::none);
    INFO("message must be empty");
    REQUIRE(report.message.empty());
    INFO("must be falsy");
    REQUIRE(!static_cast<bool>(report));
}
TEST_CASE("error: keeps code and message") {
    lazuli::error report{lazuli::error_code::singular, "matrix is singular"};
    REQUIRE(report.code == lazuli::error_code::singular);
    REQUIRE(report.message == std::string{"matrix is singular"});
    INFO("must be truthy");
    REQUIRE(static_cast<bool>(report));
}
TEST_CASE("error: dimension mismatch type") {
    const lazuli::dimension_mismatch failure{"ragged"};
    const std::invalid_argument& base = failure;
    REQUIRE(std::string_view{base.what()} == std::string_view{"ragged"});
    REQUIRE_THROWS_AS(throw failure, std::exception);
}
TEST_CASE("error: singular matrix type") {
    const lazuli::singular_matrix failure{"zero pivot"};
    const std::runtime_error& base = failure;
    REQUIRE(std::string_view{base.what()} == std::string_view{"zero pivot"});
}
TEST_CASE("error: out of bounds type") {
    const lazuli::out_of_bounds failure{"index 9"};
    const std::out_of_range& base = failure;
    REQUIRE(std::string_view{base.what()} == std::string_view{"index 9"});
    REQUIRE_THROWS_AS(throw failure, std::logic_error);
}
TEST_CASE("error: fail_dimension_mismatch") {
    REQUIRE_THROWS_WITH(lazuli::detail::fail_dimension_mismatch("shapes disagree"),
                        Catch::Matchers::ContainsSubstring("shapes disagree"));
}
TEST_CASE("error: fail_out_of_bounds wording") {
    REQUIRE_THROWS_WITH(lazuli::detail::fail_out_of_bounds(7, 3),
                        Catch::Matchers::ContainsSubstring("7"));
    REQUIRE_THROWS_WITH(lazuli::detail::fail_out_of_bounds(7, 3),
                        Catch::Matchers::ContainsSubstring("3"));
}
TEST_CASE("error: fail_bad_alloc throws") {
    REQUIRE_THROWS_AS(lazuli::detail::fail_bad_alloc(), std::bad_alloc);
}

TEST_CASE("error: fail_out_of_bounds edges") {
    REQUIRE_THROWS_AS(lazuli::detail::fail_out_of_bounds(0, 0), lazuli::out_of_bounds);
    REQUIRE_THROWS_AS(lazuli::detail::fail_out_of_bounds(std::size_t{0} - 1, std::size_t{0} - 1),
                      lazuli::out_of_bounds);
}

}  // namespace
