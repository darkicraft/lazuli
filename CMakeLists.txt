cmake_minimum_required(VERSION 3.28)

project(
    lazuli
    VERSION 0.1.0
    LANGUAGES CXX)

# Keep the generated database where the clangd/Zed symlink expects it.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Modules are deliberately parked; keep the build free of the module scanner.
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)
set(CMAKE_CXX_EXTENSIONS OFF)

option(LAZULI_BUILD_TESTS "Build the test suite" ON)
option(LAZULI_BUILD_EXAMPLES "Build the examples" ON)
option(LAZULI_BUILD_BENCH "Build the benchmark executables" ON)
option(LAZULI_WARNINGS_AS_ERRORS "Treat compiler warnings as errors" ON)

# Apply the project's warning set to a target.
function(lazuli_set_warnings target)
    target_compile_options(
        ${target}
        PRIVATE -Wall
                -Wextra
                -Wpedantic
                -Wconversion
                -Wsign-conversion
                -Wshadow)
    if(LAZULI_WARNINGS_AS_ERRORS)
        target_compile_options(${target} PRIVATE -Werror)
    endif()
endfunction()

add_library(
    lazuli STATIC
    src/core/allocation.cpp
    src/core/error.cpp
    src/decomp/decomp.cpp
    src/io/io.cpp
    src/kernel/matmul.cpp)
add_library(lazuli::lazuli ALIAS lazuli)

target_compile_features(lazuli PUBLIC cxx_std_20)
target_include_directories(
    lazuli
    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
           $<INSTALL_INTERFACE:include>)

lazuli_set_warnings(lazuli)

# OpenMP is a hard dependency: the kernel always uses it for parallelism.
find_package(OpenMP REQUIRED)
target_link_libraries(lazuli PRIVATE OpenMP::OpenMP_CXX)

# The frozen baseline is the tests' independent reference and the benchmark
# denominator.
if(LAZULI_BUILD_TESTS OR LAZULI_BUILD_BENCH)
    add_subdirectory(bench)
endif()

if(LAZULI_BUILD_TESTS)
    enable_testing()

    set(lazuli_guards ${CMAKE_CURRENT_SOURCE_DIR}/cmake/guards.cmake)
    add_test(NAME guards COMMAND ${CMAKE_COMMAND} -P ${lazuli_guards})

    # One tiny object target per public header, so the build system
    # compiles them in parallel instead of one compiler launch after
    # another. `ctest` then just builds the aggregate.
    file(
        GLOB_RECURSE lazuli_headers
        RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}/include"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/*.hpp")
    set(lazuli_header_targets "")
    foreach(header IN LISTS lazuli_headers)
        string(MAKE_C_IDENTIFIER "${header}" slug)
        set(source "${CMAKE_CURRENT_BINARY_DIR}/header_checks/${slug}.cpp")
        file(WRITE "${source}" "#include <${header}>\n")
        add_library(header_check_${slug} OBJECT EXCLUDE_FROM_ALL "${source}")
        target_include_directories(
            header_check_${slug}
            PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include")
        target_compile_features(header_check_${slug} PRIVATE cxx_std_20)
        lazuli_set_warnings(header_check_${slug})
        list(APPEND lazuli_header_targets header_check_${slug})
    endforeach()
    add_custom_target(header_checks
                      COMMENT "Compile every public header on its own"
                      DEPENDS ${lazuli_header_targets})
    add_test(NAME headers
             COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR}
                     --target header_checks)

    add_subdirectory(tests)
endif()

if(LAZULI_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()
