#include <lazuli/decomp/decomp.hpp>

#include "decomp_impl.hpp"

namespace lazuli::detail::decomp {

expected<lu_result<float>, error> lu(const matrix<float>& a, options settings) {
    return impl::lu_impl<float>(a, settings);
}

expected<lu_result<double>, error> lu(const matrix<double>& a, options settings) {
    return impl::lu_impl<double>(a, settings);
}

expected<vector<float>, error> solve(const matrix<float>& a,
                                     const vector<float>& b,
                                     options settings) {
    return impl::solve_impl<float>(a, b, settings);
}

expected<vector<double>, error> solve(const matrix<double>& a,
                                      const vector<double>& b,
                                      options settings) {
    return impl::solve_impl<double>(a, b, settings);
}

float det(const matrix<float>& a) {
    return impl::det_impl<float>(a);
}
double det(const matrix<double>& a) {
    return impl::det_impl<double>(a);
}

expected<matrix<float>, error> inverse(const matrix<float>& a, options settings) {
    return impl::inverse_impl<float>(a, settings);
}

expected<matrix<double>, error> inverse(const matrix<double>& a, options settings) {
    return impl::inverse_impl<double>(a, settings);
}

expected<matrix<float>, error> cholesky(const matrix<float>& a, options settings) {
    return impl::cholesky_impl<float>(a, settings);
}

expected<matrix<double>, error> cholesky(const matrix<double>& a, options settings) {
    return impl::cholesky_impl<double>(a, settings);
}

expected<std::vector<std::size_t>, error> lu_in_place(matrix_view<float> work, options settings) {
    if (work.rows() != work.cols()) {
        return unexpected{impl::make_error(error_code::not_square, "LU needs a square matrix")};
    }
    std::vector<std::size_t> pivots(work.rows());
    for (std::size_t index = 0; index < work.rows(); ++index) {
        pivots[index] = index;
    }
    const auto swaps = impl::lu_factor<float>(work, pivots, settings);
    if (!swaps) {
        return unexpected{swaps.error()};
    }
    return pivots;
}

expected<std::vector<std::size_t>, error> lu_in_place(matrix_view<double> work, options settings) {
    if (work.rows() != work.cols()) {
        return unexpected{impl::make_error(error_code::not_square, "LU needs a square matrix")};
    }
    std::vector<std::size_t> pivots(work.rows());
    for (std::size_t index = 0; index < work.rows(); ++index) {
        pivots[index] = index;
    }
    const auto swaps = impl::lu_factor<double>(work, pivots, settings);
    if (!swaps) {
        return unexpected{swaps.error()};
    }
    return pivots;
}

expected<qr_result<float>, error> qr(const matrix<float>& a, options settings) {
    return impl::qr_impl<float>(a, settings);
}

expected<qr_result<double>, error> qr(const matrix<double>& a, options settings) {
    return impl::qr_impl<double>(a, settings);
}

expected<vector<float>, error> eigenvalues(const matrix<float>& a) {
    return impl::eigenvalues_impl<float>(a);
}

expected<vector<double>, error> eigenvalues(const matrix<double>& a) {
    return impl::eigenvalues_impl<double>(a);
}

}  // namespace lazuli::detail::decomp
