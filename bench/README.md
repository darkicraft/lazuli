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
./build/release/bench/time_matmul a.lzm b.lzm --reps 5 --output c.lzm
# size=1024x1024x1024 reps=5 median_ms=1355.354 gflops=1.584 checksum=5.108456711e+03
```

The first run is a discarded warm-up. The reported figure is the median of the rest, and the
checksum proves the result was consumed.

## Recording a number

Record a measurement with the machine, the build type, the size and the command. A timing without
that context is a rumour. Run-to-run spread on a 15 W laptop is several percent, so treat anything
under ~5% as noise until it repeats.

| Date | Size | Reps | Median | GFLOPS | Where |
|---|---|---|---|---|---|
| 2026-11-16 | 1024³ | 5 | 1355 ms | 1.584 | WSL2, GCC 16.2, `-O3 -DNDEBUG` |

The row above was taken in WSL2 on the development machine; it should be re-measured natively
before it is quoted anywhere, since virtualisation and the host scheduler both skew the clock.
