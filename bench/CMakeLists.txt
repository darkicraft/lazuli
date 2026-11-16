# The frozen baseline is built unconditionally: the tests use it as the
# yardstick for every optimised kernel that follows.
add_library(lazuli_baseline STATIC baseline/matmul.cpp)
target_link_libraries(lazuli_baseline PUBLIC lazuli::lazuli)
target_include_directories(lazuli_baseline PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
lazuli_set_warnings(lazuli_baseline)

if(LAZULI_BUILD_BENCH)
    add_executable(bench_time_matmul time_matmul.cpp)
    target_link_libraries(bench_time_matmul PRIVATE lazuli_baseline)
    lazuli_set_warnings(bench_time_matmul)
    set_target_properties(
        bench_time_matmul
        PROPERTIES OUTPUT_NAME time_matmul
                   RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bench"
                   FOLDER "bench")
endif()
