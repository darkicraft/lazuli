#pragma once

/// @file
/// @brief The scalar types a matrix may hold.

#include <type_traits>

namespace lazuli::detail {

/// Every element type the library supports.
///
/// Deliberately closed: `matrix<int>` is a modelling mistake, and `long double` has three
/// different ABIs across the targets we care about. Views carry the same type possibly with a
/// `const` qualifier, so constness is stripped before the check.
template <typename T>
concept scalar =
    std::is_same_v<std::remove_const_t<T>, float> || std::is_same_v<std::remove_const_t<T>, double>;

}  // namespace lazuli::detail
