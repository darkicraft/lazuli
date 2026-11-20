#include <lazuli/core/aligned.hpp>

#include <catch2/catch_message.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cstddef>
#include <cstdint>
#include <utility>

namespace {

using lazuli::detail::aligned_allocate;
using lazuli::detail::aligned_buffer;
using lazuli::detail::aligned_release;
using lazuli::detail::default_alignment;

constexpr std::uintptr_t cache_line = lazuli::detail::default_alignment;

TEST_CASE("aligned: default is empty") {
    const aligned_buffer<double> buffer;
    REQUIRE(buffer.size() == std::size_t{0});
    INFO("must be empty");
    REQUIRE(buffer.empty());
    INFO("must not own storage");
    REQUIRE(buffer.data() == nullptr);
}
TEST_CASE("aligned: sized buffer reports size") {
    const aligned_buffer<double> buffer{5};
    REQUIRE(buffer.size() == std::size_t{5});
    INFO("must not be empty");
    REQUIRE(!buffer.empty());
    INFO("must own storage");
    REQUIRE(buffer.data() != nullptr);
}
TEST_CASE("aligned: every size is aligned") {
    for (std::size_t count = 1; count <= 17; ++count) {
        const aligned_buffer<double> buffer{count};
        INFO("buffer must start on a cache line");
        REQUIRE(reinterpret_cast<std::uintptr_t>(buffer.data()) % cache_line == 0);
    }
    const aligned_buffer<std::byte> odd{1000};
    INFO("odd size alignment");
    REQUIRE(reinterpret_cast<std::uintptr_t>(odd.data()) % cache_line == 0);
}
TEST_CASE("aligned: odd byte counts are aligned") {
    for (std::size_t bytes : {std::size_t{1},
                              std::size_t{3},
                              std::size_t{7},
                              std::size_t{63},
                              std::size_t{65},
                              std::size_t{1000}}) {
        void* pointer = aligned_allocate(bytes, default_alignment);
        INFO("allocation must succeed");
        REQUIRE(pointer != nullptr);
        INFO("alignment");
        REQUIRE(reinterpret_cast<std::uintptr_t>(pointer) % cache_line == 0);
        aligned_release(pointer);
    }
}
TEST_CASE("aligned: release accepts null") {
    aligned_release(nullptr);
}
TEST_CASE("aligned: move construction") {
    aligned_buffer<double> source{4};
    double* const address = source.data();
    aligned_buffer<double> moved{std::move(source)};
    INFO("storage must move");
    REQUIRE(moved.data() == address);
    REQUIRE(moved.size() == std::size_t{4});
    INFO("source must be emptied");
    REQUIRE(source.data() == nullptr);
    REQUIRE(source.size() == std::size_t{0});
}
TEST_CASE("aligned: move assignment") {
    aligned_buffer<double> target{2};
    aligned_buffer<double> source{7};
    double* const address = source.data();
    target = std::move(source);
    INFO("storage must move");
    REQUIRE(target.data() == address);
    REQUIRE(target.size() == std::size_t{7});
    INFO("source must be emptied");
    REQUIRE(source.data() == nullptr);
}
TEST_CASE("aligned: self move keeps the buffer") {
    aligned_buffer<double> buffer{3};
    double* const address = buffer.data();
    aligned_buffer<double>& alias = buffer;
    buffer = std::move(alias);
    INFO("self move must not free the storage");
    REQUIRE(buffer.data() == address);
    REQUIRE(buffer.size() == std::size_t{3});
}
TEST_CASE("aligned: swap exchanges buffers") {
    aligned_buffer<double> left{2};
    aligned_buffer<double> right{9};
    double* const left_address = left.data();
    double* const right_address = right.data();
    left.swap(right);
    INFO("left takes right storage");
    REQUIRE(left.data() == right_address);
    INFO("right takes left storage");
    REQUIRE(right.data() == left_address);
    REQUIRE(left.size() == std::size_t{9});
    REQUIRE(right.size() == std::size_t{2});
}

}  // namespace
