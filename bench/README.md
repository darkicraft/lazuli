# Benchmarks

One executable, one menu, one file per operation. It times each operation at 256, 512 and 1024,
prints GFLOPS and the speedup against the naive stage, and returns to the menu.

```sh
cmake --build build/release --target bench
./build/release/bench/bench
```

```
1) matmul       naive -> SIMD -> blocked -> full
2) LU           naive -> SIMD -> blocked -> full
3) Cholesky     naive -> SIMD -> blocked -> full
4) thread sweep matmul, 1, 2, 3, 4, 6, 8 and 12 threads
5) OpenBLAS reference
6) iterations   current: 3
7) exit
```

`iterations` controls how many timed repetitions each stage gets; the first run is always
discarded. The `-O0` naive stage is slow at 1024, so use `1` for a quick look.

## Layout

```text
bench/
  main.cpp                  menu and dispatch only
  operations/               one file per menu entry
    matmul.cpp              run_matmul
    lu.cpp                  run_lu
    cholesky.cpp            run_cholesky
    threads.cpp             run_threads
    openblas.cpp            run_openblas
  references/
    naive/                  matmul + LU/Cholesky, compiled with -O0
    simd/                   explicit AVX2/FMA matmul + LU/Cholesky
    openblas.*              optional OpenBLAS wrapper
  common/bench.hpp          best_of, gflops, stage-row formatting
  tools/validate_matmul.cpp file-backed matmul validator
```

The menu is the benchmark. `tools/validate_matmul.cpp` is not a menu entry; it is the
file-backed matmul validator used by `tools/validate_matmul.py`.

## The stages

- **naive** — the frozen matmul baseline and hand-written LU/Cholesky, compiled with `-O0`.
- **SIMD** — explicit AVX2/FMA with no packing, blocking or threads.
- **blocked** — the library kernel/factorisation at one thread.
- **full** — the library at the default thread count.

## Measured, one quick run, iterations = 1

| Operation | n | naive | SIMD | blocked/1t | full/all | SIMD vs naive | full vs naive |
|---|---|---|---|---|---|---|---|
| matmul | 256 | 0.14 | 8.34 | 14.42 | 43.59 | 57.6x | 301x |
| matmul | 512 | 0.13 | 7.36 | 14.16 | 41.65 | 57.5x | 325x |
| matmul | 1024 | 0.14 | 4.71 | 17.80 | 48.64 | 34.7x | 359x |
| lu | 256 | 0.15 | 3.61 | 5.06 | 3.09 | 24.2x | 20.7x |
| lu | 512 | 0.15 | 3.66 | 8.26 | 9.93 | 23.8x | 64.6x |
| lu | 1024 | 0.15 | 2.95 | 9.15 | 9.98 | 19.7x | 66.4x |
| cholesky | 256 | 0.13 | 4.89 | 2.95 | 2.70 | 37.7x | 20.8x |
| cholesky | 512 | 0.14 | 4.21 | 3.74 | 4.23 | 31.0x | 31.2x |
| cholesky | 1024 | 0.15 | 4.06 | 4.05 | 7.78 | 26.7x | 51.2x |

Matmul gains most from blocking and threads. LU at 256 loses to parallel overhead. The simple
vectorised Cholesky beats the library’s blocked path at small sizes because the library pays for
its panels and for allocating its result; the library’s full version wins at 1024.

## Other menu options

- **thread sweep** — the kernel at 1, 2, 3, 4, 6, 8 and 12 threads.
- **OpenBLAS reference** — if `pkg-config` found OpenBLAS at configure time. One thread and the
  hardware thread count; the numbers are indicative only.

## Naive reference and validator

`lazuli_naive` is built whenever tests or benchmarks are enabled, because the tests use its
matmul as their independent correctness reference. It is compiled with `-O0`.

`bench/tools/validate_matmul.cpp` builds `build/bench/tools/validate_matmul`. It reads two `.lzm`
files, runs matmul, and writes the result for the NumPy check in
`tools/validate_matmul.py`.
