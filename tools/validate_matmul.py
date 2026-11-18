#!/usr/bin/env python3
"""Validate the baseline matmul against NumPy, and report the baseline.

Generates two random matrices, writes them in lazuli's binary format, runs the benchmark
binary on those files, and compares the result with numpy.matmul. The format is implemented
here independently of the C++ reader, so a round-trip through both sides is what gets tested.

    .venv/bin/python tools/validate_matmul.py --size 1024 --seed 0
"""

from __future__ import annotations

import argparse
import pathlib
import struct
import subprocess
import sys
import tempfile

import numpy as np

HEADER = struct.Struct("<4sHBBQQ")
MAGIC = b"LZLI"
VERSION = 1
TAG_FLOAT64 = 2
HEADER_SIZE = HEADER.size


def write_matrix(path: pathlib.Path, matrix: np.ndarray) -> None:
    data = np.ascontiguousarray(matrix, dtype="<f8")
    rows, cols = data.shape
    with open(path, "wb") as stream:
        stream.write(HEADER.pack(MAGIC, VERSION, TAG_FLOAT64, 0, rows, cols))
        stream.write(data.tobytes(order="C"))


def read_matrix(path: pathlib.Path) -> np.ndarray:
    with open(path, "rb") as stream:
        header = HEADER.unpack(stream.read(HEADER_SIZE))
        magic, version, tag, _reserved, rows, cols = header
        if magic != MAGIC:
            raise ValueError(f"{path}: not a lazuli file")
        if version != VERSION:
            raise ValueError(f"{path}: unsupported version {version}")
        if tag != TAG_FLOAT64:
            raise ValueError(f"{path}: expected float64, got tag {tag}")
        payload = stream.read(rows * cols * 8)
        if len(payload) != rows * cols * 8:
            raise ValueError(f"{path}: truncated payload")
    return np.frombuffer(payload, dtype="<f8").reshape(rows, cols).copy()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--bench", type=pathlib.Path, default=pathlib.Path("build/bench/time_matmul"))
    parser.add_argument(
        "--kernel",
        action="store_true",
        help="time the library kernel instead of the frozen baseline",
    )
    parser.add_argument("--keep", action="store_true", help="keep the generated files")
    options = parser.parse_args()

    if not options.bench.is_file():
        print(f"benchmark binary not found: {options.bench}", file=sys.stderr)
        return 2

    rng = np.random.default_rng(options.seed)
    a = rng.uniform(-1.0, 1.0, (options.size, options.size))
    b = rng.uniform(-1.0, 1.0, (options.size, options.size))
    expected = a @ b

    workdir = pathlib.Path(tempfile.mkdtemp(prefix="lazuli-validate-"))
    a_path, b_path, c_path = workdir / "a.lzm", workdir / "b.lzm", workdir / "c.lzm"
    write_matrix(a_path, a)
    write_matrix(b_path, b)

    command = [str(options.bench), str(a_path), str(b_path), "--reps", str(options.reps),
               "--output", str(c_path)]
    if options.kernel:
        command.append("--kernel")
    result = subprocess.run(
        command,
        check=False, capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(result.stderr or result.stdout, file=sys.stderr)
        return 1
    print(result.stdout.strip())

    actual = read_matrix(c_path)
    if actual.shape != expected.shape:
        print(f"shape mismatch: {actual.shape} vs {expected.shape}", file=sys.stderr)
        return 1

    difference = np.abs(actual - expected)
    scale = np.linalg.norm(expected)
    relative = float(np.linalg.norm(difference) / scale) if scale else 0.0
    print(f"max_abs_error={difference.max():.3e} relative_error={relative:.3e}")

    if not options.keep:
        for path in (a_path, b_path, c_path):
            path.unlink(missing_ok=True)
        workdir.rmdir()
    else:
        print(f"kept: {workdir}")

    if relative > 1e-12:
        print("FAIL: baseline disagrees with NumPy", file=sys.stderr)
        return 1
    print("PASS: baseline agrees with NumPy")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
