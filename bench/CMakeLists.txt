# The frozen baseline is the tests' independent reference and the benchmark denominator. This
# subdirectory is only configured when tests or benchmarks are enabled. It is compiled with -O0.
add_library(lazuli_baseline STATIC baseline/matmul.cpp)
target_compile_options(lazuli_baseline PRIVATE -O0)
target_link_libraries(lazuli_baseline PUBLIC lazuli::lazuli)
target_include_directories(lazuli_baseline PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
lazuli_set_warnings(lazuli_baseline)

if(LAZULI_BUILD_BENCH)
    find_package(PkgConfig QUIET)
    if(PkgConfig_FOUND)
        pkg_check_modules(LAZULI_OPENBLAS QUIET IMPORTED_TARGET openblas)
    endif()

    # Naive LU and Cholesky, also compiled with -O0.
    add_library(lazuli_naive STATIC naive/linalg.cpp)
    target_compile_options(lazuli_naive PRIVATE -O0)
    target_link_libraries(lazuli_naive PUBLIC lazuli::lazuli)
    target_include_directories(lazuli_naive PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
    lazuli_set_warnings(lazuli_naive)

    add_executable(bench main.cpp stages/linalg.cpp)
    target_link_libraries(bench PRIVATE lazuli::lazuli lazuli_baseline lazuli_naive)
    target_include_directories(bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    if(LAZULI_OPENBLAS_FOUND)
        target_sources(bench PRIVATE reference/openblas.cpp)
        target_link_libraries(bench PRIVATE PkgConfig::LAZULI_OPENBLAS)
        target_compile_definitions(bench PRIVATE LAZULI_HAVE_OPENBLAS=1)
        message(STATUS "bench: OpenBLAS reference enabled")
    else()
        message(STATUS "bench: OpenBLAS not found; the menu option reports that")
    endif()
    lazuli_set_warnings(bench)
    set_target_properties(
        bench
        PROPERTIES OUTPUT_NAME bench
                   RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bench"
                   FOLDER "bench")

    # File-based validator utility, kept for tools/validate_matmul.py. Not part of the menu.
    add_executable(bench_time_matmul tools/time_matmul.cpp)
    target_link_libraries(bench_time_matmul PRIVATE lazuli_baseline)
    target_include_directories(bench_time_matmul PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    lazuli_set_warnings(bench_time_matmul)
    set_target_properties(
        bench_time_matmul
        PROPERTIES OUTPUT_NAME time_matmul
                   RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bench"
                   FOLDER "bench")
endif()
