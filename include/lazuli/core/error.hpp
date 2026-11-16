#pragma once

/// @file
/// @brief Error codes, the error value type and typed exceptions.

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>

namespace lazuli::detail {

/// Classification of every failure the library reports.
enum class error_code : std::int32_t {
    none = 0,
    dimension_mismatch,
    not_square,
    not_positive_definite,
    singular,
    not_converged,
    io_parse_error,
    out_of_bounds,
};

/// Stable, human readable spelling of a code.
[[nodiscard]] std::string_view to_string(error_code code) noexcept;

/// Value form of a failure, used by the `std::expected` facade.
struct error {
    error_code code = error_code::none;
    std::string message;

    [[nodiscard]] explicit operator bool() const noexcept { return code != error_code::none; }
};

/// Two operands, or a request, disagreed about a size.
class dimension_mismatch : public std::invalid_argument {
  public:
    using std::invalid_argument::invalid_argument;
};

/// A factorisation met a matrix it cannot decompose.
class singular_matrix : public std::runtime_error {
  public:
    using std::runtime_error::runtime_error;
};

/// An index fell outside its container.
class out_of_bounds : public std::out_of_range {
  public:
    using std::out_of_range::out_of_range;
};

/// Failure entry points. Defined in the library so that no header needs a throw
/// expression, which keeps every public header usable with exceptions disabled.
[[noreturn]] void fail_dimension_mismatch(std::string_view what);
[[noreturn]] void fail_out_of_bounds(std::size_t index, std::size_t bound);
[[noreturn]] void fail_bad_alloc();

}  // namespace lazuli::detail
