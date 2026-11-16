#pragma once

/// @file
/// @brief Deterministic pseudo-random sampling used by matrix construction.

#include <complex>
#include <cstdint>
#include <type_traits>

namespace lazuli::detail {

/// Half-open sampling bounds: low inclusive, high exclusive.
///
/// For a complex `T`, the bounds' real parts are used for both components, so
/// `{-1.0, 1.0}` samples the whole unit square rather than a line.
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

template <typename T>
struct is_complex : std::false_type {};

template <typename T>
struct is_complex<std::complex<T>> : std::true_type {};

template <typename T>
inline constexpr bool is_complex_v = is_complex<T>::value;

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
    if constexpr (is_complex_v<T>) {
        using real = typename T::value_type;
        const interval<real> real_bounds{bounds.low.real(), bounds.high.real()};
        return T{uniform_between(generator, real_bounds), uniform_between(generator, real_bounds)};
    } else if constexpr (std::is_same_v<T, double>) {
        return bounds.low + (bounds.high - bounds.low) * unit_double(generator.next());
    } else {
        static_assert(std::is_same_v<T, float>);
        return bounds.low + (bounds.high - bounds.low) * unit_float(generator.next());
    }
}

}  // namespace lazuli::detail
