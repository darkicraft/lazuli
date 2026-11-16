#pragma once

/// @file
/// @brief Aligned allocation used by the owning containers.

#include <cstddef>
#include <cstdlib>
#include <utility>

#include "lazuli/core/error.hpp"

namespace lazuli::detail {

/// Alignment of every owning allocation, in bytes: one cache line.
inline constexpr std::size_t default_alignment = 64;

[[nodiscard]] inline void* aligned_allocate(std::size_t bytes, std::size_t alignment) {
    // std::aligned_alloc requires the size to be a multiple of the alignment.
    const std::size_t rounded = (bytes + alignment - 1) / alignment * alignment;
    void* pointer = std::aligned_alloc(alignment, rounded);
    if (pointer == nullptr) {
        fail_bad_alloc();
    }
    return pointer;
}

inline void aligned_release(void* pointer) noexcept {
    std::free(pointer);
}

/// Owning, cache-line aligned, uninitialised storage for `count` elements.
template <typename T>
class aligned_buffer {
  public:
    aligned_buffer() noexcept = default;

    explicit aligned_buffer(std::size_t count)
        : m_count(count) {
        if (m_count != 0) {
            m_data = static_cast<T*>(aligned_allocate(m_count * sizeof(T), default_alignment));
        }
    }

    aligned_buffer(const aligned_buffer&) = delete;
    aligned_buffer& operator=(const aligned_buffer&) = delete;

    aligned_buffer(aligned_buffer&& other) noexcept
        : m_data(other.m_data),
          m_count(other.m_count) {
        other.m_data = nullptr;
        other.m_count = 0;
    }

    aligned_buffer& operator=(aligned_buffer&& other) noexcept {
        aligned_buffer moved{std::move(other)};
        swap(moved);
        return *this;
    }

    ~aligned_buffer() { aligned_release(m_data); }

    void swap(aligned_buffer& other) noexcept {
        std::swap(m_data, other.m_data);
        std::swap(m_count, other.m_count);
    }

    [[nodiscard]] T* data() noexcept { return m_data; }
    [[nodiscard]] const T* data() const noexcept { return m_data; }
    [[nodiscard]] std::size_t size() const noexcept { return m_count; }
    [[nodiscard]] bool empty() const noexcept { return m_count == 0; }

  private:
    T* m_data = nullptr;
    std::size_t m_count = 0;
};

}  // namespace lazuli::detail
