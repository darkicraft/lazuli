#pragma once

/// @file
/// @brief Per-call settings. No global state: the library never changes the process-wide
/// OpenMP defaults, because a library that does that is a library that fights its host.

#include <cstddef>

namespace lazuli::detail {

struct options {
    /// Worker threads for one operation. 0 means "whatever the runtime considers the default",
    /// which respects `OMP_NUM_THREADS` if the program set it.
    std::size_t num_threads = 0;
};

}  // namespace lazuli::detail
