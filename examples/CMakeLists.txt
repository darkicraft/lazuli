# Examples are arranged by feature, mirroring `include/`. Each target is a small
# standalone program demonstrating one part of the API. The build mirrors the
# source tree and the binary is named after the source file:
#
#   examples/matrix/basic.cpp  ->  build/examples/matrix/basic
function(lazuli_add_example name source)
    add_executable(${name} ${source})
    target_link_libraries(${name} PRIVATE lazuli::lazuli)
    lazuli_set_warnings(${name})

    get_filename_component(source_dir "${source}" DIRECTORY)
    get_filename_component(source_name "${source}" NAME_WE)
    set_target_properties(
        ${name}
        PROPERTIES OUTPUT_NAME "${source_name}"
                   RUNTIME_OUTPUT_DIRECTORY
                   "${CMAKE_BINARY_DIR}/examples/${source_dir}"
                   FOLDER "examples")

    # Examples are also smoke tests: they must build and run cleanly.
    if(LAZULI_BUILD_TESTS)
        add_test(NAME "example: ${name}" COMMAND ${name})
    endif()
endfunction()

lazuli_add_example(matrix_basic matrix/basic.cpp)
lazuli_add_example(matrix_random matrix/random.cpp)
lazuli_add_example(matrix_views matrix/views.cpp)
lazuli_add_example(vector_basic vector/basic.cpp)
lazuli_add_example(linalg_basic linalg/basic.cpp)
lazuli_add_example(io_basic io/basic.cpp)
lazuli_add_example(expr_basic expr/basic.cpp)
lazuli_add_example(expr_threads expr/threads.cpp)
lazuli_add_example(errors_handling errors/handling.cpp)
