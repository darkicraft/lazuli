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

[[nodiscard]] lazuli::matrix<double> identity(std::size_t n) {
    lazuli::matrix<double> result(n, n);
    for (std::size_t index = 0; index < n; ++index) {
        result(index, index) = 1.0;
    }
    return result;
}

[[nodiscard]] bool is_upper_triangular(const lazuli::matrix<double>& matrix, double tolerance) {
    for (std::size_t row = 0; row < matrix.rows(); ++row) {
        for (std::size_t column = 0; column < row && column < matrix.cols(); ++column) {
            if (std::abs(matrix(row, column)) > tolerance) {
                return false;
            }
        }
    }
    return true;
}

}  // namespace

TEST_CASE("decomp: QR of a square matrix") {
    const auto a = lazuli::matrix<double>::random(40, 40, 43, {-1.0, 1.0});
    const auto factors = lazuli::nt::qr(a);
    REQUIRE(factors.has_value());
    REQUIRE(factors->q.rows() == 40);
    REQUIRE(factors->q.cols() == 40);
    REQUIRE(factors->r.rows() == 40);
    REQUIRE(factors->r.cols() == 40);
    REQUIRE(is_upper_triangular(factors->r, 1e-12));
    REQUIRE(max_abs_difference(factors->q * factors->r, a) <= 1e-10);
}

TEST_CASE("decomp: Q is orthogonal") {
    const auto a = lazuli::matrix<double>::random(30, 30, 47, {-1.0, 1.0});
    const auto factors = lazuli::nt::qr(a);
    REQUIRE(factors.has_value());
    const lazuli::matrix<double> product = factors->q.transpose() * factors->q;
    REQUIRE(max_abs_difference(product, identity(30)) <= 1e-12);
}

TEST_CASE("decomp: QR of a tall matrix") {
    const auto a = lazuli::matrix<double>::random(50, 30, 53, {-1.0, 1.0});
    const auto factors = lazuli::nt::qr(a);
    REQUIRE(factors.has_value());
    REQUIRE(factors->q.rows() == 50);
    REQUIRE(factors->q.cols() == 50);
    REQUIRE(factors->r.rows() == 50);
    REQUIRE(factors->r.cols() == 30);
    REQUIRE(is_upper_triangular(factors->r, 1e-12));
    REQUIRE(max_abs_difference(factors->q * factors->r, a) <= 1e-10);
}

TEST_CASE("decomp: QR of the identity") {
    const auto factors = lazuli::nt::qr(identity(4));
    REQUIRE(factors.has_value());
    REQUIRE(max_abs_difference(factors->q * factors->r, identity(4)) <= 1e-12);

    // The signs of the factors are free: (Q, R) and (-Q, -R) are both valid, so only the
    // magnitudes are asserted.
    REQUIRE(is_upper_triangular(factors->r, 1e-12));
    for (std::size_t index = 0; index < 4; ++index) {
        REQUIRE(std::abs(std::abs(factors->r(index, index)) - 1.0) <= 1e-12);
    }
}

TEST_CASE("decomp: QR of a rank-deficient matrix still reconstructs") {
    lazuli::matrix<double> a{{1.0, 2.0}, {2.0, 4.0}};
    const auto factors = lazuli::nt::qr(a);
    REQUIRE(factors.has_value());
    REQUIRE(max_abs_difference(factors->q * factors->r, a) <= 1e-12);
}

TEST_CASE("decomp: QR rejects a matrix with more columns than rows") {
    const lazuli::matrix<double> wide(3, 5);
    const auto factors = lazuli::nt::qr(wide);
    REQUIRE(!factors.has_value());
    REQUIRE(factors.error().code == lazuli::error_code::dimension_mismatch);
}

TEST_CASE("decomp: eigenvalues of a diagonal matrix") {
    lazuli::matrix<double> diagonal(3, 3);
    diagonal(0, 0) = 5.0;
    diagonal(1, 1) = -2.0;
    diagonal(2, 2) = 1.0;

    const auto values = lazuli::nt::eigenvalues(diagonal);
    REQUIRE(values.has_value());
    REQUIRE(values->size() == 3);
    REQUIRE(std::abs((*values)[0] + 2.0) <= 1e-12);
    REQUIRE(std::abs((*values)[1] - 1.0) <= 1e-12);
    REQUIRE(std::abs((*values)[2] - 5.0) <= 1e-12);
}

TEST_CASE("decomp: eigenvalues of a known symmetric matrix") {
    const lazuli::matrix<double> a{{2.0, 1.0, 0.0}, {1.0, 2.0, 0.0}, {0.0, 0.0, 3.0}};
    const auto values = lazuli::nt::eigenvalues(a);
    REQUIRE(values.has_value());
    REQUIRE(std::abs((*values)[0] - 1.0) <= 1e-12);
    REQUIRE(std::abs((*values)[1] - 3.0) <= 1e-12);
    REQUIRE(std::abs((*values)[2] - 3.0) <= 1e-12);
}

TEST_CASE("decomp: eigenvalues agree with the trace and the Frobenius norm") {
    const auto b = lazuli::matrix<double>::random(60, 60, 59, {-1.0, 1.0});
    lazuli::matrix<double> a = b + b.transpose();
    for (std::size_t index = 0; index < 60; ++index) {
        a(index, index) += 3.0;
    }

    const auto values = lazuli::nt::eigenvalues(a);
    REQUIRE(values.has_value());

    double trace = 0.0;
    double squares = 0.0;
    for (std::size_t row = 0; row < 60; ++row) {
        trace += a(row, row);
        for (std::size_t column = 0; column < 60; ++column) {
            squares += a(row, column) * a(row, column);
        }
    }
    double value_sum = 0.0;
    double value_squares = 0.0;
    for (std::size_t index = 0; index < values->size(); ++index) {
        value_sum += (*values)[index];
        value_squares += (*values)[index] * (*values)[index];
    }
    REQUIRE(std::abs(value_sum - trace) <= 1e-9);
    REQUIRE(std::abs(value_squares - squares) <= 1e-8);
}

TEST_CASE("decomp: eigenvalues reject an asymmetric matrix") {
    const lazuli::matrix<double> asymmetric{{1.0, 2.0}, {0.0, 1.0}};
    const auto values = lazuli::nt::eigenvalues(asymmetric);
    REQUIRE(!values.has_value());
    REQUIRE(values.error().code == lazuli::error_code::not_symmetric);
}

TEST_CASE("decomp: eigenvalues reject a non-square matrix") {
    const lazuli::matrix<double> wide(2, 3);
    const auto values = lazuli::nt::eigenvalues(wide);
    REQUIRE(!values.has_value());
    REQUIRE(values.error().code == lazuli::error_code::not_square);
}

TEST_CASE("decomp: QR and eigenvalues work in float") {
    const auto a = lazuli::matrix<float>::random(20, 20, 61, {-1.0F, 1.0F});
    const auto factors = lazuli::nt::qr(a);
    REQUIRE(factors.has_value());
    float worst = 0.0F;
    const lazuli::matrix<float> product = factors->q * factors->r;
    for (std::size_t row = 0; row < 20; ++row) {
        for (std::size_t column = 0; column < 20; ++column) {
            worst = std::max(worst, std::abs(product(row, column) - a(row, column)));
        }
    }
    REQUIRE(worst <= 1e-4F);

    lazuli::matrix<float> symmetric = a + a.transpose();
    const auto values = lazuli::nt::eigenvalues(symmetric);
    REQUIRE(values.has_value());
    REQUIRE(values->size() == 20);
}
