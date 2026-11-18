include(FetchContent)

# One checkout of the test framework shared by every build directory.
set(FETCHCONTENT_BASE_DIR
    "${PROJECT_SOURCE_DIR}/.deps"
    CACHE PATH "Shared download cache for FetchContent")

set(CATCH_BUILD_TESTING
    OFF
    CACHE BOOL "" FORCE)

FetchContent_Declare(
    catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG v3.16.1
    GIT_SHALLOW TRUE)
FetchContent_MakeAvailable(catch2)

list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)

add_executable(
    lazuli_tests
    bench/baseline.cpp
    core/aligned.cpp
    core/error.cpp
    expr/alias.cpp
    expr/elementwise.cpp
    expr/matmul.cpp
    expr/vector.cpp
    io/io.cpp
    matrix/fill.cpp
    matrix/matrix.cpp
    matrix/random.cpp
    matrix/scalar.cpp
    matrix/vector.cpp
    matrix/view.cpp)

target_link_libraries(
    lazuli_tests
    PRIVATE lazuli::lazuli
            lazuli_baseline
            Catch2::Catch2WithMain)
target_include_directories(lazuli_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
lazuli_set_warnings(lazuli_tests)

# One ctest entry per TEST_CASE, discovered at build time.
include(Catch)
catch_discover_tests(lazuli_tests)

# The public headers must remain usable without exceptions, and the
# probe doubles as a check that the umbrella is self-sufficient.
add_library(lazuli_public_surface STATIC public_surface.cpp)
target_link_libraries(lazuli_public_surface PRIVATE lazuli::lazuli)
target_include_directories(
    lazuli_public_surface
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
lazuli_set_warnings(lazuli_public_surface)
target_compile_options(lazuli_public_surface PRIVATE -fno-exceptions)
