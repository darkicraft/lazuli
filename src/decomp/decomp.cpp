#include <lazuli/decomp/decomp.hpp>

#include "decomp_impl.hpp"

namespace lazuli::detail::decomp {

std::expected<lu_result<float>, error> lu(const matrix<float>& a, options settings) {
    return impl::lu_impl<float>(a, settings);
}

std::expected<lu_result<double>, error> lu(const matrix<double>& a, options settings) {
    return impl::lu_impl<double>(a, settings);
}

std::expected<vector<float>, error> solve(const matrix<float>& a,
                                          const vector<float>& b,
                                          options settings) {
    return impl::solve_impl<float>(a, b, settings);
}

std::expected<vector<double>, error> solve(const matrix<double>& a,
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

std::expected<matrix<float>, error> inverse(const matrix<float>& a, options settings) {
    return impl::inverse_impl<float>(a, settings);
}

std::expected<matrix<double>, error> inverse(const matrix<double>& a, options settings) {
    return impl::inverse_impl<double>(a, settings);
}

std::expected<matrix<float>, error> cholesky(const matrix<float>& a, options settings) {
    return impl::cholesky_impl<float>(a, settings);
}

std::expected<matrix<double>, error> cholesky(const matrix<double>& a, options settings) {
    return impl::cholesky_impl<double>(a, settings);
}

std::expected<qr_result<float>, error> qr(const matrix<float>& a, options settings) {
    return impl::qr_impl<float>(a, settings);
}

std::expected<qr_result<double>, error> qr(const matrix<double>& a, options settings) {
    return impl::qr_impl<double>(a, settings);
}

std::expected<vector<float>, error> eigenvalues(const matrix<float>& a) {
    return impl::eigenvalues_impl<float>(a);
}

std::expected<vector<double>, error> eigenvalues(const matrix<double>& a) {
    return impl::eigenvalues_impl<double>(a);
}

}  // namespace lazuli::detail::decomp
