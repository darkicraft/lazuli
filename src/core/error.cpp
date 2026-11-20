#include <lazuli/core/error.hpp>

#include <cstdio>
#include <cstdlib>
#include <new>
#include <string>
#include <string_view>

namespace lazuli::detail {

std::string_view to_string(error_code code) noexcept {
    switch (code) {
    case error_code::none:
        return "none";
    case error_code::dimension_mismatch:
        return "dimension_mismatch";
    case error_code::not_square:
        return "not_square";
    case error_code::not_positive_definite:
        return "not_positive_definite";
    case error_code::not_symmetric:
        return "not_symmetric";
    case error_code::singular:
        return "singular";
    case error_code::not_converged:
        return "not_converged";
    case error_code::io_parse_error:
        return "io_parse_error";
    case error_code::out_of_bounds:
        return "out_of_bounds";
    }
    return "unknown";
}

#if !defined(__cpp_exceptions)
namespace {

/// Last resort when exceptions are unavailable: report and die, like a failed
/// assertion.
[[noreturn]] void abort_with(std::string_view what) noexcept {
    std::fputs("lazuli: ", stderr);
    std::fwrite(what.data(), 1, what.size(), stderr);
    std::fputc('\n', stderr);
    std::abort();
}

}  // namespace
#endif

[[noreturn]] void fail_dimension_mismatch(std::string_view what) {
#if defined(__cpp_exceptions)
    throw dimension_mismatch{std::string{what}};
#else
    abort_with(what);
#endif
}

[[noreturn]] void fail_out_of_bounds(std::size_t index, std::size_t bound) {
    const std::string message =
        "index " + std::to_string(index) + " is outside [0, " + std::to_string(bound) + ")";
#if defined(__cpp_exceptions)
    throw out_of_bounds{message};
#else
    abort_with(message);
#endif
}

[[noreturn]] void fail_bad_alloc() {
#if defined(__cpp_exceptions)
    throw std::bad_alloc{};
#else
    abort_with("allocation failed");
#endif
}

}  // namespace lazuli::detail
