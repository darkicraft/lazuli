# Benchmarks

One executable, one menu, one file per operation. It times each operation at 256, 512 and 1024,
prints GFLOPS and the speedup against the naive stage, and returns to the menu.

```sh
cmake --build build/release --target bench
./build/release/bench/bench
```

```
lazuli benchmark | AVX2 | 12 hardware threads | 3 iterations

  m  matmul       naive / SIMD / blocked 1t / full all
  l  LU           naive / SIMD / blocked 1t / full all
  c  Cholesky     naive / SIMD / blocked 1t / full all
  t  thread sweep matmul at 1, 2, 4, 6, 8, 12 threads
  o  OpenBLAS     reference, if built
  v  verify       QR residuals and Jacobi eigenvalue checks
  a  all staged   matmul + LU + Cholesky + thread sweep
  i  iterations   set repetitions (current: 3)
  ?  help
  q  quit

bench [3]>
```

The full menu is printed at startup and by `?`; after an operation only the two-line key
footer and the prompt are printed. Mnemonic keys are the primary interface; `1`–`8` and the full
names (`matmul`, `lu`, `cholesky`, `threads`, `openblas`, `verify`, `all`, `iterations`, `help`,
`quit`) are aliases. Input is trimmed and case-insensitive.

`iterations` controls how many timed repetitions each stage gets; the first run is always
discarded. The `-O0` naive stage is slow at 1024, so use `1` for a quick look. `a` runs matmul,
LU, Cholesky and the thread sweep in sequence for a report snapshot. `v` is separate because it
measures correctness and time for algorithms with no staged performance path.

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
    verify.cpp              run_verify
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

- **thread sweep** — the kernel at powers of two up to the hardware count, half the
  hardware count when it is at least four, and the full hardware count. On 12 threads that is
  1, 2, 4, 6, 8, 12; on 64 it is 1, 2, 4, 8, 16, 32, 64.
- **OpenBLAS reference** — if `pkg-config` found OpenBLAS at configure time. One thread and the
  hardware thread count; the numbers are indicative only.
- **verify** — QR orthogonality `|Q^T Q - I|` and reconstruction `|Q R - A|`, plus the Jacobi
  eigenvalue invariants `|sum - tr A|` and `|sum^2 - ||A||_F^2|`, at 64, 128 and 256. It is not
  staged: QR and Jacobi have a single implementation, and Jacobi is O(n^3) per sweep (about
  0.7 s at 256 in a release build, 16 s at 512), so verification stops at modest sizes.

## Naive reference and validator

`lazuli_naive` is built whenever tests or benchmarks are enabled, because the tests use its
matmul as their independent correctness reference. It is compiled with `-O0`.

`bench/tools/validate_matmul.cpp` builds `build/bench/tools/validate_matmul`. It reads two `.lzm`
files, runs matmul, and writes the result for the NumPy check in
`tools/validate_matmul.py`.
