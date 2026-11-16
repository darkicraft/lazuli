/// Public-surface probe: includes the umbrella header and nothing else.
///
/// Built with exceptions disabled, so it fails if the public headers ever
/// acquire a throw expression, an implementation-level include, or a dependency
/// on a transitive include.

#include <lazuli/lazuli.hpp>

namespace {

template <typename T>
void exercise() {
    lazuli::matrix<T> m(2, 2);
    lazuli::vector<T> v(2);
    lazuli::matrix_view<T> view = m;
    auto block = view.block(0, 0, 1, 1);
    block(0, 0) = m(0, 0);
    v[0] = m(0, 0);

    lazuli::error report{};
    report.code = lazuli::error_code::none;
    (void)v;
    (void)report;
}

}  // namespace

void lazuli_public_surface() {
    exercise<float>();
    exercise<double>();
}
