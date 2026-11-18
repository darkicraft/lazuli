#pragma once

/// @file
/// @brief Counting allocations, so tests can prove the expression layer adds no temporaries.

#include <cstddef>

namespace lazuli::detail {

/// Called by every owning allocation. Cheap enough to leave on: allocations are rare next to the
/// arithmetic they enable.
void note_allocation() noexcept;

/// Allocations made since the last reset. Tests only.
[[nodiscard]] std::size_t allocation_count() noexcept;

/// Tests only.
void reset_allocation_count() noexcept;

}  // namespace lazuli::detail
