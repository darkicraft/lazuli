#pragma once

/// @file
/// @brief Minimal test harness: no framework, no macros, one function per case.

#include <cstdio>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace lazuli_test {

inline void require(bool condition, std::string_view message) {
    if (!condition) {
        throw std::runtime_error{std::string{message}};
    }
}

template <typename A, typename B>
void require_equal(const A& actual, const B& expected, std::string_view what) {
    if (!(actual == expected)) {
        throw std::runtime_error{std::string{what} + ": values differ"};
    }
}

/// True when the callable throws exactly `E`.
template <typename E, typename F>
[[nodiscard]] bool throws(F&& callable) {
    try {
        std::forward<F>(callable)();
    } catch (const E&) {
        return true;
    } catch (...) {
        return false;
    }
    return false;
}

inline int run(std::string_view name, void (*body)()) {
    const int width = static_cast<int>(name.size());
    try {
        body();
        std::printf("  ok    %.*s\n", width, name.data());
        return 0;
    } catch (const std::exception& failure) {
        std::printf("  FAIL  %.*s\n        %s\n", width, name.data(), failure.what());
        return 1;
    } catch (...) {
        std::printf("  FAIL  %.*s\n        unknown exception\n", width, name.data());
        return 1;
    }
}

}  // namespace lazuli_test
