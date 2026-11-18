#include <catch2/catch_test_macros.hpp>

#include <complex>

#include <lazuli/core/scalar.hpp>
#include <lazuli/lazuli.hpp>

// Supported element types exist.
static_assert(requires { typename lazuli::matrix<float>; });
static_assert(requires { typename lazuli::matrix<double>; });
static_assert(requires { typename lazuli::vector<float>; });
static_assert(requires { typename lazuli::vector<double>; });
static_assert(requires { typename lazuli::matrix_view<const double>; });

// The constraint itself. Asserting the negative through `requires { typename matrix<int>; }`
// is not portable — both GCC and Clang treat a constrained template's failure as a hard error
// there — so the concept is checked directly and the enforcement is verified by the compiler in
// every build that would otherwise mention an unsupported type.
static_assert(lazuli::detail::scalar<float>);
static_assert(lazuli::detail::scalar<double>);
static_assert(lazuli::detail::scalar<const double>);

static_assert(!lazuli::detail::scalar<int>);
static_assert(!lazuli::detail::scalar<unsigned>);
static_assert(!lazuli::detail::scalar<long double>);
static_assert(!lazuli::detail::scalar<std::complex<double>>);

TEST_CASE("scalars: only float and double are supported") {
    // The assertions above are the test; this case keeps them in the ctest listing.
    REQUIRE(true);
}
