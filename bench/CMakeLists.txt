# The naive references are the tests' independent oracle and the benchmark's
# naive stage.
add_library(lazuli_naive STATIC
    references/naive/matmul.cpp
    references/naive/lu_cholesky.cpp)
target_compile_options(lazuli_naive PRIVATE -O0)
target_link_libraries(lazuli_naive PUBLIC lazuli::lazuli)
target_include_directories(lazuli_naive PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
lazuli_set_warnings(lazuli_naive)

if(LAZULI_BUILD_BENCH)
    find_package(PkgConfig QUIET)
    if(PkgConfig_FOUND)
        pkg_check_modules(LAZULI_OPENBLAS QUIET IMPORTED_TARGET openblas)
    endif()

    add_executable(
        bench
        main.cpp
        common/bench.cpp
        operations/cholesky.cpp
        operations/lu.cpp
        operations/matmul.cpp
        operations/openblas.cpp
        operations/threads.cpp
        references/simd/matmul.cpp
        references/simd/lu_cholesky.cpp)
    target_link_libraries(bench PRIVATE lazuli::lazuli lazuli_naive)
    target_include_directories(bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    if(LAZULI_OPENBLAS_FOUND)
        target_sources(bench PRIVATE references/openblas.cpp)
        target_link_libraries(bench PRIVATE PkgConfig::LAZULI_OPENBLAS)
        target_compile_definitions(bench PRIVATE LAZULI_HAVE_OPENBLAS=1)
        message(STATUS "bench: OpenBLAS reference enabled")
    else()
        message(STATUS "bench: OpenBLAS not found; the menu reports that")
    endif()
    lazuli_set_warnings(bench)
    set_target_properties(
        bench
        PROPERTIES OUTPUT_NAME bench
                   RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bench"
                   FOLDER "bench")

    # File-backed matmul validator used by tools/validate_matmul.py. Not part of
    # the menu.
    add_executable(bench_validate_matmul tools/validate_matmul.cpp)
    target_link_libraries(bench_validate_matmul PRIVATE lazuli_naive)
    target_include_directories(bench_validate_matmul
                               PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    lazuli_set_warnings(bench_validate_matmul)
    set_target_properties(
        bench_validate_matmul
        PROPERTIES OUTPUT_NAME validate_matmul
                   RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bench/tools"
                   FOLDER "bench")
endif()
