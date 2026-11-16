#include <catch2/catch_test_macros.hpp>

#include <complex>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <string_view>

#include <lazuli/lazuli.hpp>

namespace {

namespace fs = std::filesystem;

/// A path in the temporary directory that is removed when the test ends, pass or fail.
class scratch_file {
  public:
    explicit scratch_file(const char* name)
        : m_path{fs::temp_directory_path() / (std::string{"lazuli_io_"} + name + ".lzm")} {}
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

std::string header_bytes(std::string_view magic,
                         std::uint16_t version,
                         std::uint8_t tag,
                         std::uint64_t rows,
                         std::uint64_t cols) {
    std::string bytes{magic};
    const auto append = [&bytes](const auto& value) {
        bytes.append(reinterpret_cast<const char*>(&value), sizeof value);
    };
    append(version);
    append(tag);
    append(std::uint8_t{0});
    append(rows);
    append(cols);
    return bytes;
}

void write_bytes(const std::string& path, std::string_view bytes) {
    std::ofstream stream{path, std::ios::binary | std::ios::trunc};
    stream.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
}

template <typename T>
void round_trip(const char* name) {
    scratch_file file{name};
    const lazuli::matrix<T> original{{T{1}, T{2}, T{3}}, {T{4}, T{5}, T{6}}};
    const auto written = lazuli::io::save(file.path(), original);
    REQUIRE(written.has_value());

    const auto loaded = lazuli::io::load<T>(file.path());
    REQUIRE(loaded.has_value());
    REQUIRE(loaded->rows() == original.rows());
    REQUIRE(loaded->cols() == original.cols());
    for (std::size_t row = 0; row < original.rows(); ++row) {
        for (std::size_t column = 0; column < original.cols(); ++column) {
            REQUIRE((*loaded)(row, column) == original(row, column));
        }
    }
}

}  // namespace

TEST_CASE("io: double round trip") {
    round_trip<double>("double");
}

TEST_CASE("io: float round trip") {
    round_trip<float>("float");
}

TEST_CASE("io: complex float round trip") {
    round_trip<std::complex<float>>("complex_float");
}

TEST_CASE("io: complex double round trip") {
    round_trip<std::complex<double>>("complex_double");
}

TEST_CASE("io: degenerate shapes round trip") {
    scratch_file file{"degenerate"};
    for (const auto& shape : {std::pair<std::size_t, std::size_t>{0, 0},
                              std::pair<std::size_t, std::size_t>{0, 4},
                              std::pair<std::size_t, std::size_t>{4, 0}}) {
        const lazuli::matrix<double> original(shape.first, shape.second);
        REQUIRE(lazuli::io::save(file.path(), original).has_value());
        const auto loaded = lazuli::io::load<double>(file.path());
        REQUIRE(loaded.has_value());
        REQUIRE(loaded->rows() == shape.first);
        REQUIRE(loaded->cols() == shape.second);
        REQUIRE(loaded->empty());
    }
}

TEST_CASE("io: strided views are written row by row") {
    scratch_file file{"strided"};
    const lazuli::matrix<double> original{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};
    REQUIRE(lazuli::io::save(file.path(), original.transpose()).has_value());

    const auto loaded = lazuli::io::load<double>(file.path());
    REQUIRE(loaded.has_value());
    REQUIRE(loaded->rows() == 3);
    REQUIRE(loaded->cols() == 2);
    for (std::size_t row = 0; row < 3; ++row) {
        for (std::size_t column = 0; column < 2; ++column) {
            REQUIRE((*loaded)(row, column) == original(column, row));
        }
    }
}

TEST_CASE("io: a block view writes only its window") {
    scratch_file file{"block"};
    const lazuli::matrix<double> original{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {7.0, 8.0, 9.0}};
    REQUIRE(lazuli::io::save(file.path(), original.block(1, 1, 2, 2)).has_value());

    const auto loaded = lazuli::io::load<double>(file.path());
    REQUIRE(loaded.has_value());
    REQUIRE(loaded->rows() == 2);
    REQUIRE((*loaded)(0, 0) == 5.0);
    REQUIRE((*loaded)(1, 1) == 9.0);
}

TEST_CASE("io: a missing file is an io_parse_error") {
    const std::string path = (fs::temp_directory_path() / "lazuli_io_absent.lzm").string();
    fs::remove(path);
    const auto loaded = lazuli::io::load<double>(path);
    REQUIRE(!loaded.has_value());
    REQUIRE(loaded.error().code == lazuli::error_code::io_parse_error);
    REQUIRE(loaded.error().message.find("cannot open") != std::string::npos);
}

TEST_CASE("io: a bad magic is rejected") {
    scratch_file file{"magic"};
    write_bytes(file.path(), header_bytes("NOPE", 1, 2, 1, 1));
    const auto loaded = lazuli::io::load<double>(file.path());
    REQUIRE(!loaded.has_value());
    REQUIRE(loaded.error().code == lazuli::error_code::io_parse_error);
}

TEST_CASE("io: a truncated header is rejected") {
    scratch_file file{"short_header"};
    write_bytes(file.path(), "LZLI\x01");
    const auto loaded = lazuli::io::load<double>(file.path());
    REQUIRE(!loaded.has_value());
    REQUIRE(loaded.error().message.find("truncated header") != std::string::npos);
}

TEST_CASE("io: a truncated payload is rejected") {
    scratch_file file{"short_payload"};
    std::string bytes = header_bytes("LZLI", 1, 2, 2, 3);
    const double one_value = 1.0;
    bytes.append(reinterpret_cast<const char*>(&one_value), sizeof one_value);
    write_bytes(file.path(), bytes);

    const auto loaded = lazuli::io::load<double>(file.path());
    REQUIRE(!loaded.has_value());
    REQUIRE(loaded.error().message.find("truncated payload") != std::string::npos);
}

TEST_CASE("io: the scalar type must match the request") {
    scratch_file file{"tag"};
    const lazuli::matrix<float> values{{1.0F, 2.0F}};
    REQUIRE(lazuli::io::save(file.path(), values).has_value());
    const auto loaded = lazuli::io::load<double>(file.path());
    REQUIRE(!loaded.has_value());
    REQUIRE(loaded.error().message.find("scalar type") != std::string::npos);
}

TEST_CASE("io: impossible dimensions are rejected") {
    scratch_file file{"huge"};
    write_bytes(file.path(),
                header_bytes("LZLI", 1, 2, std::uint64_t{1} << 40, std::uint64_t{1} << 40));
    const auto loaded = lazuli::io::load<double>(file.path());
    REQUIRE(!loaded.has_value());
    REQUIRE(loaded.error().code == lazuli::error_code::io_parse_error);
}

TEST_CASE("io: an unwritable path is an error") {
    const std::string path =
        (fs::temp_directory_path() / "lazuli_no_such_directory" / "out.lzm").string();
    const lazuli::matrix<double> values(1, 1);
    const auto written = lazuli::io::save(path, values);
    REQUIRE(!written.has_value());
    REQUIRE(written.error().code == lazuli::error_code::io_parse_error);
}
