#include <atomic>
#include <cstddef>

#include "lazuli/core/allocation.hpp"

namespace lazuli::detail {

namespace {

std::atomic<std::size_t> allocations{0};

}  // namespace

void note_allocation() noexcept {
    allocations.fetch_add(1, std::memory_order_relaxed);
}

std::size_t allocation_count() noexcept {
    return allocations.load(std::memory_order_relaxed);
}

void reset_allocation_count() noexcept {
    allocations.store(0, std::memory_order_relaxed);
}

}  // namespace lazuli::detail
