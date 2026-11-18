# Benchmarks

Two things live here:

- `baseline/` — the frozen naive matrix multiplication. It has no blocking, no threads and no
  explicit vectorisation, and it is the denominator for every speedup the project reports.
  **Do not optimise it.** If it changes, every number measured before the change is void.
- `time_matmul.cpp` — a CLI that times the baseline on two matrices read from files.

The tests link `baseline/` as their yardstick, so it is built even with
`-DLAZULI_BUILD_BENCH=OFF`; only the CLI is gated by that option.

## Build

```sh
cmake -S . -B build/release -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build/release --target bench_time_matmul
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

There is no OpenMP yet: threading lands in M4, on top of this kernel.
