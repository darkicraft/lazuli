#include <lazuli/decomp/decomp.hpp>
#include <lazuli/lazuli.hpp>

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>

namespace {

[[nodiscard]] double max_abs_difference(const lazuli::matrix<double>& left,
                                        const lazuli::matrix<double>& right) {
    REQUIRE(left.rows() == right.rows());
    REQUIRE(left.cols() == right.cols());
    double worst = 0.0;
    for (std::size_t row = 0; row < left.rows(); ++row) {
        for (std::size_t column = 0; column < left.cols(); ++column) {
            worst = std::max(worst, std::abs(left(row, column) - right(row, column)));
        }
    }
    return worst;
}

[[nodiscard]] double solve_residual(const lazuli::matrix<double>& a,
                                    const lazuli::vector<double>& x,
                                    const lazuli::vector<double>& b) {
    double worst = 0.0;
    for (std::size_t row = 0; row < a.rows(); ++row) {
        double sum = 0.0;
        for (std::size_t column = 0; column < a.cols(); ++column) {
            sum += a(row, column) * x[column];
        }
        worst = std::max(worst, std::abs(sum - b[row]));
    }
    return worst;
}

/// The permutation matrix implied by the factorisation's pivot list.
[[nodiscard]] lazuli::matrix<double> permutation_of(const lazuli::lu_result<double>& factors) {
    const std::size_t n = factors.pivots.size();
    lazuli::matrix<double> p(n, n);
    for (std::size_t row = 0; row < n; ++row) {
        p(row, factors.pivots[row]) = 1.0;
    }
    return p;
}

}  // namespace

TEST_CASE("decomp: LU reproduces PA on a matrix that needs pivoting") {
    const lazuli::matrix<double> a{{0.0, 2.0}, {3.0, 4.0}};
    const auto factors = lazuli::lu(a);
    REQUIRE(factors.has_value());

    const lazuli::matrix<double> product = factors->l * factors->u;
    const lazuli::matrix<double> expected = permutation_of(*factors) * a;
    REQUIRE(max_abs_difference(product, expected) <= 1e-12);
    REQUIRE(factors->l(0, 0) == 1.0);
    REQUIRE(factors->l(1, 1) == 1.0);
    REQUIRE(factors->l(0, 1) == 0.0);
    REQUIRE(factors->u(1, 0) == 0.0);
}

TEST_CASE("decomp: LU on a random matrix reproduces PA") {
    const auto a = lazuli::matrix<double>::random(40, 40, 7, {-1.0, 1.0});
    const auto factors = lazuli::lu(a);
    REQUIRE(factors.has_value());
    const lazuli::matrix<double> product = factors->l * factors->u;
    const lazuli::matrix<double> expected = permutation_of(*factors) * a;
    REQUIRE(max_abs_difference(product, expected) <= 1e-10);
}

TEST_CASE("decomp: LU rejects a non-square matrix") {
    const lazuli::matrix<double> wide(2, 3);
    const auto factors = lazuli::lu(wide);
    REQUIRE(!factors.has_value());
    REQUIRE(factors.error().code == lazuli::error_code::not_square);
}

TEST_CASE("decomp: LU reports a singular matrix") {
    const lazuli::matrix<double> singular{{1.0, 2.0}, {2.0, 4.0}};
    const auto factors = lazuli::lu(singular);
    REQUIRE(!factors.has_value());
    REQUIRE(factors.error().code == lazuli::error_code::singular);
}

TEST_CASE("decomp: solve") {
    const auto a = lazuli::matrix<double>::random(50, 50, 11, {-1.0, 1.0});
    const auto b = lazuli::vector<double>::random(50, 13, {-1.0, 1.0});
    const auto x = lazuli::solve(a, b);
    REQUIRE(x.has_value());
    REQUIRE(solve_residual(a, *x, b) <= 1e-10);
}

TEST_CASE("decomp: solve honours the thread setting") {
    const auto a = lazuli::matrix<double>::random(120, 120, 17, {-1.0, 1.0});
    const auto b = lazuli::vector<double>::random(120, 19, {-1.0, 1.0});
    const auto one = lazuli::solve(a, b, lazuli::options{1});
    const auto four = lazuli::solve(a, b, lazuli::options{4});
    REQUIRE(one.has_value());
    REQUIRE(four.has_value());
    for (std::size_t index = 0; index < b.size(); ++index) {
        REQUIRE(std::abs((*one)[index] - (*four)[index]) <= 1e-10);
    }
}

TEST_CASE("decomp: solve rejects a mismatched right-hand side") {
    const lazuli::matrix<double> a(3, 3);
    const lazuli::vector<double> b(2);
    const auto x = lazuli::solve(a, b);
    REQUIRE(!x.has_value());
    REQUIRE(x.error().code == lazuli::error_code::dimension_mismatch);
}

TEST_CASE("decomp: solve reports a singular matrix") {
    const lazuli::matrix<double> singular{{0.0, 1.0}, {0.0, 2.0}};
    const lazuli::vector<double> b{1.0, 1.0};
    const auto x = lazuli::solve(singular, b);
    REQUIRE(!x.has_value());
    REQUIRE(x.error().code == lazuli::error_code::singular);
}

TEST_CASE("decomp: determinant of known matrices") {
    REQUIRE(std::abs(lazuli::det(lazuli::matrix<double>{{4.0, 3.0}, {6.0, 3.0}}) + 6.0) <= 1e-12);
    REQUIRE(std::abs(lazuli::det(lazuli::matrix<double>{{1.0, 0.0}, {0.0, 1.0}}) - 1.0) <= 1e-12);
    REQUIRE(std::abs(lazuli::det(lazuli::matrix<double>{{0.0, 1.0}, {1.0, 0.0}}) + 1.0) <= 1e-12);
    REQUIRE(lazuli::det(lazuli::matrix<double>{{1.0, 2.0}, {2.0, 4.0}}) == 0.0);
}

TEST_CASE("decomp: determinant matches the product of the LU diagonal") {
    const auto a = lazuli::matrix<double>::random(30, 30, 23, {-1.0, 1.0});
    const auto factors = lazuli::lu(a);
    REQUIRE(factors.has_value());
    double product = factors->swaps % 2 == 0 ? 1.0 : -1.0;
    for (std::size_t index = 0; index < 30; ++index) {
        product *= factors->u(index, index);
    }
    REQUIRE(std::abs(lazuli::det(a) - product) <= 1e-9);
}

TEST_CASE("decomp: inverse multiplies back to the identity") {
    const auto a = lazuli::matrix<double>::random(40, 40, 29, {-1.0, 1.0});
    const auto inverse = lazuli::inverse(a);
    REQUIRE(inverse.has_value());

    lazuli::matrix<double> identity(40, 40);
    for (std::size_t index = 0; index < 40; ++index) {
        identity(index, index) = 1.0;
    }
    REQUIRE(max_abs_difference(a * (*inverse), identity) <= 1e-9);
}

TEST_CASE("decomp: inverse rejects a singular matrix") {
    const lazuli::matrix<double> singular{{1.0, 1.0}, {1.0, 1.0}};
    const auto inverse = lazuli::inverse(singular);
    REQUIRE(!inverse.has_value());
    REQUIRE(inverse.error().code == lazuli::error_code::singular);
}

TEST_CASE("decomp: Cholesky rebuilds a positive-definite matrix") {
    const auto b = lazuli::matrix<double>::random(30, 30, 31, {-1.0, 1.0});
    lazuli::matrix<double> spd = b.transpose() * b;
    for (std::size_t index = 0; index < 30; ++index) {
        spd(index, index) += 30.0;
    }

    const auto lower = lazuli::cholesky(spd);
    REQUIRE(lower.has_value());
    REQUIRE(max_abs_difference((*lower) * lower->transpose(), spd) <= 1e-10);
}

TEST_CASE("decomp: Cholesky rejects a matrix that is not positive definite") {
    const lazuli::matrix<double> indefinite{{1.0, 2.0}, {2.0, 1.0}};
    const auto lower = lazuli::cholesky(indefinite);
    REQUIRE(!lower.has_value());
    REQUIRE(lower.error().code == lazuli::error_code::not_positive_definite);
}

TEST_CASE("decomp: Cholesky rejects an asymmetric matrix") {
    const lazuli::matrix<double> asymmetric{{1.0, 2.0}, {0.0, 1.0}};
    const auto lower = lazuli::cholesky(asymmetric);
    REQUIRE(!lower.has_value());
    REQUIRE(lower.error().code == lazuli::error_code::not_positive_definite);
}

TEST_CASE("decomp: float works as well as double") {
    const auto a = lazuli::matrix<float>::random(30, 30, 37, {-1.0F, 1.0F});
    const auto b = lazuli::vector<float>::random(30, 41, {-1.0F, 1.0F});

    const auto x = lazuli::solve(a, b);
    REQUIRE(x.has_value());
    float worst = 0.0F;
    for (std::size_t row = 0; row < a.rows(); ++row) {
        float sum = 0.0F;
        for (std::size_t column = 0; column < a.cols(); ++column) {
            sum += a(row, column) * (*x)[column];
        }
        worst = std::max(worst, std::abs(sum - b[row]));
    }
    REQUIRE(worst <= 1e-3F);

    const auto factors = lazuli::lu(a);
    REQUIRE(factors.has_value());
    REQUIRE(std::abs(lazuli::det(a)) > 0.0F);
}
