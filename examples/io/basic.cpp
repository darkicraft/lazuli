// Writing a matrix to disk and reading it back.
//
//   cmake --build build --target io_basic
//   ./build/examples/io/basic

#include <lazuli/lazuli.hpp>

#include <cstdio>
#include <filesystem>
#include <system_error>

int main() {
    const std::filesystem::path path =
        std::filesystem::temp_directory_path() / "lazuli_example_io.lzm";

    const lazuli::matrix<double> original{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};

    // save() takes an owning matrix or any view. A failure throws io_error.
    lazuli::io::save(path, original);
    std::printf("wrote %zu x %zu to %s\n", original.rows(), original.cols(), path.c_str());

    // load<T>() names the scalar type, because the file does not carry a C++ type.
    const lazuli::matrix<double> loaded = lazuli::io::load<double>(path);

    std::printf("read back %zu x %zu\n", loaded.rows(), loaded.cols());
    for (std::size_t row = 0; row < loaded.rows(); ++row) {
        for (std::size_t column = 0; column < loaded.cols(); ++column) {
            std::printf(" %5.1f", loaded(row, column));
        }
        std::printf("\n");
    }

    // Asking for the wrong scalar type is an exception by default, and a value
    // through the no-throw facade.
    try {
        (void)lazuli::io::load<float>(path);
    } catch (const lazuli::io_error& failure) {
        std::printf("throws: %s\n", failure.what());
    }
    const auto mismatched = lazuli::nt::io::load<float>(path);
    if (!mismatched) {
        std::printf("nt::io::load<float>: %s\n", mismatched.error().message.c_str());
    }

    std::error_code ignored;
    std::filesystem::remove(path, ignored);
    return 0;
}
