# Benchmarks

The pieces here:

- `baseline/` — the frozen naive matrix multiplication. It has no blocking, no threads and no
  explicit vectorisation, and it is the denominator for every speedup the project reports.
  **Do not optimise it.** If it changes, every number measured before the change is void.
- `time_matmul.cpp` — a CLI that times the baseline or the kernel on two matrices read from files.
- `kernel_timing.cpp`, `thread_sweep.cpp`, `decomp.cpp` — the in-process timing tools whose output
  is quoted below.
- `openblas_timing.cpp` — the optional OpenBLAS reference column. It is built only when
  `pkg-config` finds `openblas`, and it is report context, never a library dependency.

The tests link `baseline/` as their yardstick, so it is built even with
`-DLAZULI_BUILD_BENCH=OFF`; only the CLIs are gated by that option.

## Build

```sh
cmake -S . -B build/release -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build/release --target bench_time_matmul
cmake --build build/release --target bench_kernel_timing
cmake --build build/release --target bench_openblas_timing   # only if OpenBLAS was found
```

## Run

Operands come from files, not from constants, so nothing on the timed path can be folded away.
`tools/validate_matmul.py` generates them and also checks the result against NumPy:

```sh
.venv/bin/python tools/validate_matmul.py --size 1024 --reps 5 \
    --bench build/release/bench/time_matmul
```

To time an existing pair directly:

```sh
./build/release/bench/time_matmul a.lzm b.lzm --reps 5
# baseline path
./build/release/bench/time_matmul a.lzm b.lzm --reps 5 --kernel
# library kernel
```

`--kernel` swaps the frozen baseline for `lazuli::detail::kernel::matmul`, which is what the
library itself calls. Both print `impl=baseline` or `impl=kernel` so a number is never
unattributed.

The first run is a discarded warm-up. The reported figure is the median of the rest, and the
checksum proves the result was consumed.

## Recording a number

Record a measurement with the machine, the build type, the size and the command. A timing without
that context is a rumour. Run-to-run spread on a 15 W laptop is several percent, so treat anything
under ~5% as noise until it repeats.

| Date | Impl | Size | Median | GFLOPS | Where |
|---|---|---|---|---|---|
| 2026-11-16 | baseline | 1024³ | 580 ms | 3.70 | WSL2, GCC 16.2, `-O3 -DNDEBUG` |
| 2026-11-18 | kernel | 1024³ | 121 ms | 17.39 | same, single threaded |

The baseline figure moved from 1.58 to 3.70 GFLOPS between runs of the *same code*. That is the
spread to expect on this laptop, and the reason a single run never goes in a report.

## M3 — the blocked AVX2 kernel

`detail::kernel::matmul` packs both operands into micro-panels, blocks for L2/L1, and runs an
AVX2 register tile: **12x4 for double** (12 accumulators of 4) and **12x8 for float** (12
accumulators of 8). Non-x86 targets, pre-AVX2 CPUs and destinations whose columns are not
contiguous take a scalar fallback. Still single threaded.

The numbers come from a committed tool, not from a scratch file:

```sh
cmake --build build/release --target bench_kernel_timing
./build/release/bench/kernel_timing
```

It times the frozen baseline and the kernel in the same process with no file I/O in the loop, and
reports the best of five runs. Latest output on this machine:

```
double
  n=128   baseline=  5.58 gflops  kernel= 13.37 gflops  speedup=2.4x
  n=256   baseline=  3.82 gflops  kernel= 18.48 gflops  speedup=4.8x
  n=512   baseline=  3.71 gflops  kernel= 16.70 gflops  speedup=4.5x
  n=1024  baseline=  2.80 gflops  kernel= 16.70 gflops  speedup=6.0x
float
  n=128   baseline= 13.52 gflops  kernel= 36.25 gflops  speedup=2.7x
  n=256   baseline= 10.66 gflops  kernel= 43.42 gflops  speedup=4.1x
  n=512   baseline=  8.62 gflops  kernel= 38.46 gflops  speedup=4.5x
  n=1024  baseline=  9.14 gflops  kernel= 32.92 gflops  speedup=3.6x
```

The float baseline is the scalar i-k-j loop, because the frozen baseline is double only.

### Why these block sizes

`MC=96, NC=256, KC=128` follows from the cache sizes of the target class of machine, read here
with `lscpu`:

- L1d: 192 KiB across 6 cores, so **32 KiB per core**
- L2: 3 MiB across 6 cores, so **512 KiB per core**
- L3: 16 MiB shared

Two constraints follow:

- the packed B panel (`KC x NC` doubles = 128 x 256 x 8 = 256 KiB) plus the packed A panel
  (`MC x KC` = 96 x 128 x 8 = 96 KiB) must fit the 512 KiB L2;
- one A micro-panel (`KC x 12` = 12 KiB) plus one B micro-panel (`KC x 4` = 4 KiB) must fit the
  32 KiB L1.

`MC` is a multiple of the 12-row tile (96 = 8 x 12) so the panel loop stays even. The sweep then
confirmed the arithmetic:

| MC | NC | KC | GFLOPS |
|---|---|---|---|
| 64 | 256 | 256 | 12.0 |
| 72 | 128 | 128 | 13.8 |
| 72 | 128 | 256 | 12.6 |
| 96 | 256 | 128 | 17.4 |

These are compile-time constants. Changing them changes speed, never correctness, and the binary
still runs on any x86-64 CPU because the AVX2 path is chosen at run time — unlike
`-march=native`, which would bake the instruction set into the binary.

With no hardware counters available on this machine, the evidence for the blocking is these
timings rather than cache-miss data.

## M4 — OpenMP over the output tiles

The kernel parallelises over row blocks of C: each worker packs its own A panel and writes its
own rows, so there is no shared write and no lock. The B panel is packed once and read by every
worker. The thread count is a per-call setting, never a change to the process-wide OpenMP state:

```cpp
lazuli::matrix<double> c = (a * b).eval(lazuli::options{4});   // 4 workers
lazuli::matrix<double> d = a * b;                              // runtime default
```

Outputs below `64^3` multiplications run serially, because fork/join costs more than it saves
there.

```sh
cmake --build build/release --target bench_thread_sweep
./build/release/bench/thread_sweep
```

| n | 1 | 2 | 3 | 4 | 6 | 8 | 12 |
|---|---|---|---|---|---|---|---|
| 256 GFLOPS | 14.70 | 20.87 | 30.70 | 37.16 | 30.63 | 31.61 | 22.43 |
| 512 GFLOPS | 16.85 | 25.53 | 40.44 | 41.71 | 61.07 | 52.24 | 47.80 |
| 1024 GFLOPS | 30.68 | 50.49 | 68.92 | 72.22 | **80.64** | 79.93 | 65.42 |

Speedup at 1024 against one thread: 2.63x at six threads, then it *falls* — 12 hardware threads
on a 6-core part oversubscribes, and the all-core clock is well below the single-core boost. The
same shape, earlier, at 512: 3.62x at six threads.

Against the frozen baseline (3.9 GFLOPS at 1024, single threaded) the threaded kernel is about
20x. That comparison is the one the project reports; the single-thread figures above are there to
show where the threading gains stop.

## M5 — LU and Cholesky

Both are blocked on `panel_width = 64`, and both use the matmul kernel for their trailing updates,
so the thread setting reaches them; the panel factorisation itself is serial, because the
dependencies run down it.

```sh
cmake --build build/release --target bench_decomp
./build/release/bench/decomp
```

| n | lu/1t | lu/all | chol/1t | chol/all |
|---|---|---|---|---|
| 512 | 4.07 | 4.27 (1.05x) | 2.50 | 2.77 (1.1x) |
| 1024 | 6.19 | 7.64 (1.2x) | 3.52 | 4.82 (1.4x) |

(128 and 256 are too small to report: panel and allocation overhead dominate and the numbers do
not repeat.)

These are 5-8x below the matmul kernel, and the profile says why. At 1024 the Cholesky time splits
as block factorisation 1 ms, panel solve 22 ms, kernel update 18 ms, and the update's subtraction
6 ms. The panel solve is level-2 and memory-bound, so it is the floor no matter how fast the
kernel is.

Three attempts were made, measured over five runs each, and the honest result is that **none of
them moved the numbers beyond this machine's noise (±15%)**:

- **Fusing the subtraction into the kernel** (`matmul_subtract`, `fnmadd`) removes a temporary,
  its zeroing and a separate pass. Kept: it is simpler and cheaper in allocations, even though the
  end-to-end timings do not separate it from the old version.
- **Parallelising the panel solve** over rows made LU **4x slower** at 12 threads (244 ms against
  62 ms at 4 threads): it is bandwidth-bound and oversubscription turns it into contention.
  Reverted.
- **Blocking the Cholesky panel solve** into 16-wide sub-blocks with kernel updates measured
  slightly *worse* (median 4.11 against ~4.4 GFLOPS) because the extra small kernel calls cost
  more than the level-2 work they replace. Reverted.

What would actually be needed is a properly tuned level-3 TRSM (and recursive blocking) written
against a quiet measurement setup; on a 15 W laptop with ±15% run-to-run spread that is not a
change this project can validate, so the factorisations stay correct-but-unoptimised.

Medians of five runs (GFLOPS):

| n | lu/1t | lu/all | chol/1t | chol/all |
|---|---|---|---|---|
| 512 | 4.14 | 4.44 | 2.48 | 2.60 |
| 1024 | 6.19 | 6.42 | 3.51 | 4.32 |

## Staged speedup: naive -> SIMD -> blocked -> full

`bench_stages` separates the three mechanisms instead of collapsing them into one number:

- **naive** — the frozen baseline. The compiler is free to auto-vectorise it, so this is not a
  hand-written scalar loop.
- **simd** — explicit AVX2/FMA, but no packing, no cache blocking and no threads. For matmul this
  is a direct i-k-j vector loop; for LU it vectorises the trailing rank-1 update; for Cholesky it
  vectorises the row dot products.
- **blocked/1t** — the library kernel/factorisation with one thread.
- **full/all** — the library with the default thread count.

```sh
cmake --build build/release --target bench_stages
./build/release/bench/stages
./build/release/bench/stages --csv
```

One run on the 15 W Ryzen 5 PRO 5650U, GCC 16.2, `-O3 -DNDEBUG`, best of three (GFLOPS):

| Operation | n | naive | SIMD | blocked/1t | full/all | SIMD vs naive | full vs naive |
|---|---|---|---|---|---|---|---|
| matmul | 256 | 4.05 | 7.94 | 12.38 | 38.61 | 2.0x | 9.5x |
| matmul | 512 | 4.28 | 6.52 | 14.76 | 46.62 | 1.5x | 10.9x |
| matmul | 1024 | 2.92 | 4.90 | 17.06 | 47.60 | 1.7x | 16.3x |
| lu | 256 | 2.05 | 3.53 | 6.85 | 5.24 | 1.7x | 2.6x |
| lu | 512 | 2.17 | 4.08 | 9.35 | 12.54 | 1.9x | 5.8x |
| lu | 1024 | 1.85 | 3.70 | 11.72 | 16.38 | 2.0x | 8.8x |
| cholesky | 256 | 1.62 | 3.98 | 3.29 | 2.85 | 2.5x | 1.8x |
| cholesky | 512 | 1.30 | 4.65 | 4.28 | 5.07 | 3.6x | 3.9x |
| cholesky | 1024 | 1.25 | 4.04 | 5.12 | 6.59 | 3.2x | 5.3x |

The explicit SIMD step is deliberately small for matmul: the frozen baseline is already compiled
with the optimiser's auto-vectoriser, so the measured gap is explicit SIMD versus whatever the
compiler chose. The large matmul step is blocking plus OpenMP, not the vector width alone.

Two honest conclusions from the table:

- **LU** gains from the vectorised trailing update and then from blocking; at 256 the all-thread
  stage is slower than one thread, because the matrix is too small to pay for fork/join.
- **Cholesky** is the exception at small sizes: the simple vectorised dot-product version beats
  the library's blocked/1-thread path at 256 and 512. The library version pays for its panel
  structure and for allocating its result, while the bench-only version factors a preallocated
  copy. At 1024 the library's all-thread stage catches up. This is the same level-2/level-3
  trade-off described in M5, not a claim that the blocked version is universally faster.

Run-to-run spread on this laptop is large: an earlier run of the same binary gave 9.4x instead of
16.3x for 1024 matmul, and the M4 best is 80.6 GFLOPS on a quiet machine. Treat the ratios as the
shape of the gain, not a precise number. `--csv` emits the same rows for plotting.

## Reference: OpenBLAS

This column answers "how far is this kernel from a mature BLAS?", not "did we beat OpenBLAS?".
It is report context: OpenBLAS is ten years of hand-tuned kernels, and it is not a dependency of
the library.

`bench_openblas_timing` calls `cblas_dgemm` / `cblas_sgemm` on the same row-major buffers, with
the same sizes, warm-up and best-of-nine policy. It is deliberately a **separate process** from
`kernel_timing`: OpenMP and OpenBLAS both keep worker pools warm, and running them back to back
measures the two runtimes contending for the same cores. Build it with `pkg-config` present:

```sh
cmake --build build/release --target bench_openblas_timing
./build/release/bench/openblas_timing 6
```

The argument is the OpenBLAS worker count. Six means one thread per physical core on this 6C/12T
part; twelve threads was repeatedly slower and far less stable, because OpenBLAS's pthread pool
spins while the other SMT siblings want the same execution units.

One paired session (2026-11-21, GCC 16.2, OpenBLAS 0.3.32 pthread, 1024³):

| 1024³ | kernel 1t | OpenBLAS 1t | kernel 12t | OpenBLAS 6t |
|---|---|---|---|---|
| double GFLOPS | 15.7 | 20.8 | 56.7 | 84.6 |
| float GFLOPS | 29.2 | 47.4 | 89.0 | 176.5 |

The single-thread columns are the stable comparison: OpenBLAS is about 1.3x faster in double and
1.6x in float per core. The all-thread columns move by tens of percent between runs on this
15 W laptop (the M4 table above has a better kernel day at 80.6 GFLOPS double), so treat them as
an indication of the ceiling rather than a precise ratio.

OpenBLAS is discovered with `pkg-config --libs openblas`. Without it the target is skipped and
the rest of the build is unchanged; no library or test depends on it.
