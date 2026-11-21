#include <lazuli/lazuli.hpp>

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstddef>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <system_error>

namespace {

namespace fs = std::filesystem;

/// A path in the temporary directory that is removed when the test ends, pass or fail.
class scratch_file {
  public:
    explicit scratch_file(const char* name)
        : m_path{fs::temp_directory_path() / (std::string{"lazuli_facade_"} + name + ".lzm")} {}
    ~scratch_file() {
        std::error_code ignored;
        fs::remove(m_path, ignored);
    }
    scratch_file(const scratch_file&) = delete;
    scratch_file& operator=(const scratch_file&) = delete;

    [[nodiscard]] std::string path() const { return m_path.string(); }

  private:
    fs::path m_path;
};

}  // namespace

TEST_CASE("facade: a successful call returns the value directly") {
    const auto a = lazuli::matrix<double>::random(20, 20, 3, {-1.0, 1.0});
    const auto b = lazuli::vector<double>::random(20, 5, {-1.0, 1.0});

    const lazuli::vector<double> direct = lazuli::solve(a, b);
    const auto reported = lazuli::nt::solve(a, b);
    REQUIRE(reported.has_value());
    REQUIRE(direct.size() == reported->size());
    for (std::size_t index = 0; index < direct.size(); ++index) {
        REQUIRE(std::abs(direct[index] - (*reported)[index]) <= 1e-12);
    }

    const lazuli::matrix<double> inverse = lazuli::inverse(a);
    REQUIRE(inverse.rows() == a.rows());
    REQUIRE(inverse.cols() == a.cols());

    const auto factors = lazuli::lu(a);
    REQUIRE(factors.l.rows() == a.rows());
    REQUIRE(factors.u.cols() == a.cols());
}

TEST_CASE("facade: every failure code throws its own type") {
    lazuli::matrix<double> wide(2, 3);
    REQUIRE_THROWS_AS(lazuli::lu(wide), lazuli::not_square);
    REQUIRE_THROWS_AS(lazuli::inverse(wide), lazuli::not_square);
    REQUIRE_THROWS_AS(lazuli::lu_in_place(wide.view()), lazuli::not_square);
    REQUIRE_THROWS_AS(lazuli::eigenvalues(wide), lazuli::not_square);
    REQUIRE_THROWS_AS(lazuli::qr(wide), lazuli::dimension_mismatch);

    const lazuli::matrix<double> singular{{1.0, 2.0}, {2.0, 4.0}};
    REQUIRE_THROWS_AS(lazuli::lu(singular), lazuli::singular_matrix);
    REQUIRE_THROWS_AS(lazuli::inverse(singular), lazuli::singular_matrix);

    const lazuli::matrix<double> square(2, 2);
    const lazuli::vector<double> short_rhs(1);
    REQUIRE_THROWS_AS(lazuli::solve(square, short_rhs), lazuli::dimension_mismatch);

    const lazuli::matrix<double> indefinite{{1.0, 2.0}, {2.0, 1.0}};
    REQUIRE_THROWS_AS(lazuli::cholesky(indefinite), lazuli::not_positive_definite);

    const lazuli::matrix<double> asymmetric{{1.0, 2.0}, {0.0, 1.0}};
    REQUIRE_THROWS_AS(lazuli::eigenvalues(asymmetric), lazuli::not_symmetric);
}

TEST_CASE("facade: the typed exceptions keep the standard bases") {
    const lazuli::matrix<double> wide(2, 3);
    const lazuli::matrix<double> singular{{1.0, 2.0}, {2.0, 4.0}};

    REQUIRE_THROWS_AS(lazuli::lu(wide), std::invalid_argument);
    REQUIRE_THROWS_AS(lazuli::lu(wide), std::exception);
    REQUIRE_THROWS_AS(lazuli::lu(singular), std::runtime_error);
    REQUIRE_THROWS_AS(lazuli::lu(singular), std::exception);

    try {
        (void)lazuli::lu(wide);
        FAIL("a non-square matrix must throw");
    } catch (const lazuli::not_square& failure) {
        REQUIRE(std::string{failure.what()}.find("square") != std::string::npos);
    }
}

TEST_CASE("facade: the no-throw form reports the same failure as a value") {
    const lazuli::matrix<double> wide(2, 3);
    const auto result = lazuli::nt::lu(wide);
    REQUIRE(!result.has_value());
    REQUIRE(result.error().code == lazuli::error_code::not_square);
    REQUIRE(!result.error().message.empty());
}

TEST_CASE("facade: io throws io_error") {
    scratch_file file{"round_trip"};
    const lazuli::matrix<double> original{{1.0, 2.0}, {3.0, 4.0}};

    lazuli::io::save(file.path(), original);
    const lazuli::matrix<double> loaded = lazuli::io::load<double>(file.path());
    REQUIRE(loaded.approx_equal(original, 0.0));

    REQUIRE_THROWS_AS(lazuli::io::load<float>(file.path()), lazuli::io_error);
    REQUIRE_THROWS_AS(lazuli::io::load<double>(file.path() + ".missing"), lazuli::io_error);

    const fs::path missing_directory = fs::temp_directory_path() / "lazuli_no_such_directory";
    REQUIRE_THROWS_AS(lazuli::io::save(missing_directory / "x.lzm", original), lazuli::io_error);
}
