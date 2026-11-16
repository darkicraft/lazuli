add_executable(lazuli_tests test_matrix.cpp)
target_link_libraries(lazuli_tests PRIVATE lazuli::lazuli)
lazuli_set_warnings(lazuli_tests)
add_test(NAME unit COMMAND lazuli_tests)

# The public headers must remain usable without exceptions.
add_library(lazuli_public_surface STATIC public_surface.cpp)
target_link_libraries(lazuli_public_surface PRIVATE lazuli::lazuli)
lazuli_set_warnings(lazuli_public_surface)
target_compile_options(lazuli_public_surface PRIVATE -fno-exceptions)
