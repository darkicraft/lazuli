#pragma once

/// @file
/// @brief Deterministic pseudo-random sampling used by matrix construction.

#include <cstdint>
#include <type_traits>

namespace lazuli::detail {

/// Half-open sampling bounds: low inclusive, high exclusive.
template <typename T>
struct interval {
    T low{};
    T high{};
};

/// SplitMix64.
///
/// Deliberately not `std::uniform_real_distribution`: the standard leaves its output
/// implementation-defined, so a seed would produce different matrices under different
/// standard libraries and a benchmark would not be reproducible elsewhere.
class splitmix64 {
  public:
    explicit constexpr splitmix64(std::uint64_t seed) noexcept
        : m_state(seed) {}

    constexpr std::uint64_t next() noexcept {
        m_state += 0x9e3779b97f4a7c15ULL;
        std::uint64_t word = m_state;
        word = (word ^ (word >> 30)) * 0xbf58476d1ce4e5b9ULL;
        word = (word ^ (word >> 27)) * 0x94d049bb133111ebULL;
        return word ^ (word >> 31);
    }

  private:
    std::uint64_t m_state;
};

/// Uniform in [0, 1) from the top 53 bits.
[[nodiscard]] constexpr double unit_double(std::uint64_t bits) noexcept {
    return static_cast<double>(bits >> 11) * 0x1.0p-53;
}

/// Uniform in [0, 1) from the top 24 bits.
[[nodiscard]] constexpr float unit_float(std::uint64_t bits) noexcept {
    return static_cast<float>(bits >> 40) * 0x1.0p-24F;
}

template <typename T>
[[nodiscard]] T uniform_between(splitmix64& generator, interval<T> bounds) noexcept {
    if constexpr (std::is_same_v<T, double>) {
        return bounds.low + (bounds.high - bounds.low) * unit_double(generator.next());
    } else {
        static_assert(std::is_same_v<T, float>);
        return bounds.low + (bounds.high - bounds.low) * unit_float(generator.next());
    }
}

}  // namespace lazuli::detail
