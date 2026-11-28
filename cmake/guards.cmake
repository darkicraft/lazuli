cmake_minimum_required(VERSION 3.28)

# Mechanical enforcement of the rules that decay silently:
#   1. no macros under include/
#   2. no assert() under include/
#   3. no implementation-level headers (omp, intrinsics, windows) under include/
#   4. the umbrella header is nothing but includes and the alias block
#
# Written in CMake rather than shell so it runs unchanged on Linux, macOS and
# Windows, and so that CI needs no extra tooling.

get_filename_component(
    LAZULI_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)

set(problems "")
set(umbrella_name "include/lazuli/lazuli.hpp")
set(umbrella "${LAZULI_SOURCE_DIR}/${umbrella_name}")

file(GLOB_RECURSE headers RELATIVE "${LAZULI_SOURCE_DIR}"
     "${LAZULI_SOURCE_DIR}/include/*.hpp")
list(LENGTH headers header_count)

set(banned_headers "omp|immintrin|intrin|xmmintrin|emmintrin|windows")
set(banned_include "#[ \t]*include[ \t]*<(${banned_headers})\\.h>")

foreach(header IN LISTS headers)
    file(READ "${LAZULI_SOURCE_DIR}/${header}" content)

    if(content MATCHES "(^|\n)[ \t]*#[ \t]*define")
        list(APPEND problems "${header}: defines a macro")
    endif()

    if(content MATCHES "(^|[^A-Za-z0-9_])assert[ \t]*\\(")
        list(APPEND problems "${header}: uses assert()")
    endif()

    if(content MATCHES "${banned_include}")
        list(APPEND problems "${header}: includes an implementation header")
    endif()
endforeach()

if(NOT EXISTS "${umbrella}")
    list(APPEND problems "${umbrella_name}: missing")
else()
    file(STRINGS "${umbrella}" lines)
    foreach(line IN LISTS lines)
        string(STRIP "${line}" text)
        if(text STREQUAL "")
            # blank line
        elseif(text MATCHES "^[{}]")
            # namespace braces
        elseif(text MATCHES "^#(include |pragma once)")
            # includes and include guard
        elseif(text MATCHES "^namespace lazuli")
        elseif(text MATCHES "^using ")
        elseif(text MATCHES "^//")
        else()
            list(APPEND problems "${umbrella_name}: unexpected '${text}'")
        endif()
    endforeach()
endif()

if(problems)
    foreach(problem IN LISTS problems)
        message(STATUS "guard: ${problem}")
    endforeach()
    message(FATAL_ERROR "guard check failed")
endif()

message(STATUS "guards: 4 checks passed on ${header_count} headers")
