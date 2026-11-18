#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <expected>
#include <filesystem>
#include <fstream>
#include <ios>
#include <limits>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

#include "lazuli/core/error.hpp"
#include "lazuli/io/io.hpp"
#include "lazuli/matrix/matrix.hpp"
#include "lazuli/matrix/view.hpp"

namespace lazuli::detail::io {

namespace {

constexpr std::string_view magic = "LZLI";
constexpr std::uint16_t format_version = 1;
constexpr std::size_t header_size = 24;

// The format stores IEEE-754 payloads in host byte order and the header in little endian.
// Every supported platform (x86-64, aarch64, wasm) is little endian, so the payload needs no
// swapping; a big-endian port would have to add it here rather than silently writing wrong data.
static_assert(std::endian::native == std::endian::little,
              "the binary format assumes a little-endian host");

template <typename T>
struct tag_of;

template <>
struct tag_of<float> {
    static constexpr scalar_tag value = scalar_tag::float32;
};

template <>
struct tag_of<double> {
    static constexpr scalar_tag value = scalar_tag::float64;
};

[[nodiscard]] error make_error(std::string message) {
    return error{error_code::io_parse_error, std::move(message)};
}

[[nodiscard]] std::unexpected<error> fail(std::string message) {
    return std::unexpected{make_error(std::move(message))};
}

template <typename U>
[[nodiscard]] U read_little_endian(const std::byte* bytes) noexcept {
    U value{};
    std::memcpy(&value, bytes, sizeof value);
    return value;
}

template <typename U>
void write_little_endian(std::array<std::byte, header_size>& header,
                         std::size_t offset,
                         U value) noexcept {
    std::memcpy(header.data() + offset, &value, sizeof value);
}

[[nodiscard]] std::string describe(const std::filesystem::path& path) {
    return "path \"" + path.string() + "\"";
}

template <typename T>
[[nodiscard]] std::expected<matrix<T>, error> load_file(const std::filesystem::path& path) {
    std::ifstream stream{path, std::ios::binary};
    if (!stream) {
        return fail("cannot open " + describe(path));
    }

    std::array<std::byte, header_size> header{};
    stream.read(reinterpret_cast<char*>(header.data()), static_cast<std::streamsize>(header_size));
    if (stream.gcount() != static_cast<std::streamsize>(header_size)) {
        return fail(describe(path) + ": truncated header");
    }
    if (std::memcmp(header.data(), magic.data(), magic.size()) != 0) {
        return fail(describe(path) + ": not a lazuli matrix file");
    }
    if (read_little_endian<std::uint16_t>(header.data() + 4) != format_version) {
        return fail(describe(path) + ": unsupported format version");
    }
    const auto tag = static_cast<scalar_tag>(std::to_integer<std::uint8_t>(header[6]));
    if (tag != tag_of<T>::value) {
        return fail(describe(path) + ": scalar type does not match the requested matrix");
    }

    const std::uint64_t rows = read_little_endian<std::uint64_t>(header.data() + 8);
    const std::uint64_t cols = read_little_endian<std::uint64_t>(header.data() + 16);
    if (cols != 0 && rows > std::numeric_limits<std::size_t>::max() / cols) {
        return fail(describe(path) + ": dimensions are too large");
    }
    const auto elements = static_cast<std::size_t>(rows * cols);
    if (rows > std::numeric_limits<std::size_t>::max() ||
        cols > std::numeric_limits<std::size_t>::max()) {
        return fail(describe(path) + ": dimensions are too large");
    }

    matrix<T> result{static_cast<std::size_t>(rows), static_cast<std::size_t>(cols)};
    if (elements != 0) {
        const auto bytes = static_cast<std::streamsize>(elements * sizeof(T));
        stream.read(reinterpret_cast<char*>(result.data()), bytes);
        if (stream.gcount() != bytes) {
            return fail(describe(path) + ": truncated payload");
        }
    }
    if (!stream) {
        return fail(describe(path) + ": read failed");
    }
    return result;
}

template <typename T>
[[nodiscard]] std::expected<void, error> save_file(const std::filesystem::path& path,
                                                   matrix_view<const T> data) {
    using element = std::remove_const_t<T>;
    std::ofstream stream{path, std::ios::binary | std::ios::trunc};
    if (!stream) {
        return fail("cannot create " + describe(path));
    }

    std::array<std::byte, header_size> header{};
    std::memcpy(header.data(), magic.data(), magic.size());
    write_little_endian(header, 4, format_version);
    header[6] = static_cast<std::byte>(tag_of<element>::value);
    header[7] = std::byte{0};
    write_little_endian(header, 8, static_cast<std::uint64_t>(data.rows()));
    write_little_endian(header, 16, static_cast<std::uint64_t>(data.cols()));
    stream.write(reinterpret_cast<const char*>(header.data()),
                 static_cast<std::streamsize>(header.size()));

    // Row by row, because a view may be strided or transposed.
    for (std::size_t row = 0; row < data.rows(); ++row) {
        if (data.col_stride() == 1) {
            const auto bytes = static_cast<std::streamsize>(data.cols() * sizeof(T));
            stream.write(reinterpret_cast<const char*>(data.data() + row * data.row_stride()),
                         bytes);
        } else {
            for (std::size_t column = 0; column < data.cols(); ++column) {
                stream.write(reinterpret_cast<const char*>(&data(row, column)),
                             static_cast<std::streamsize>(sizeof(T)));
            }
        }
    }

    if (!stream) {
        return fail(describe(path) + ": write failed");
    }
    return {};
}

}  // namespace

template <scalar T>
std::expected<matrix<T>, error> load(const std::filesystem::path& path) {
    return load_file<T>(path);
}

template <scalar T>
std::expected<void, error> save(const std::filesystem::path& path, matrix_view<T> data) {
    return save_file<std::remove_const_t<T>>(path, data);
}

template std::expected<matrix<float>, error> load<float>(const std::filesystem::path&);
template std::expected<matrix<double>, error> load<double>(const std::filesystem::path&);

template std::expected<void, error> save<float>(const std::filesystem::path&, matrix_view<float>);
template std::expected<void, error> save<const float>(const std::filesystem::path&,
                                                      matrix_view<const float>);
template std::expected<void, error> save<double>(const std::filesystem::path&, matrix_view<double>);
template std::expected<void, error> save<const double>(const std::filesystem::path&,
                                                       matrix_view<const double>);

}  // namespace lazuli::detail::io
