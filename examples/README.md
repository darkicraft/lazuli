# Lazuli examples

Every user-facing feature gets a small standalone program here, grouped by the
part of the library it demonstrates. They are built as part of the normal build
and also run as smoke tests, so an example breaks the moment the API it shows
stops working.

## Running

The binary is named after the source file, and the build mirrors this tree:

| Source | Binary |
|---|---|
| `examples/matrix/basic.cpp` | `build/examples/matrix/basic` |
| `examples/matrix/views.cpp` | `build/examples/matrix/views` |
| `examples/matrix/random.cpp` | `build/examples/matrix/random` |
| `examples/vector/basic.cpp` | `build/examples/vector/basic` |
| `examples/io/basic.cpp` | `build/examples/io/basic` |
| `examples/expr/basic.cpp` | `build/examples/expr/basic` |
| `examples/errors/handling.cpp` | `build/examples/errors/handling` |

```sh
cmake --build build --target matrix_basic
./build/examples/matrix/basic
```

## What each one shows

- **`matrix/basic`** — the three constructors (zero-filled, initialiser list,
  `uninitialized`), unchecked `operator()` against checked `at()`, deep copies,
  and the row-major strides.
- **`matrix/views`** — `block`, `row`, `col`, `transpose` and the implicit
  conversion to a `matrix_view`, all as windows that write back to the original.
- **`matrix/random`** — reproducible `random(rows, cols, seed, bounds)`, `fill` with a value
  and with a generator, and the same on a block.
- **`io/basic`** — writing a matrix to disk, reading it back with an explicit scalar type, and
  the error returned when the type does not match.
- **`expr/basic`** — lazy expressions: a fused elementwise chain, a product that writes straight
  into the destination, a product inside a larger expression, `noalias()`, `eval()`, a transposed
  operand, and a matrix-vector product.
- **`vector/basic`** — construction, indexing, iteration with plain pointers, and
  `as_column()` handing a vector to matrix-shaped code without copying.
- **`errors/handling`** — what is thrown where, catching through the standard
  exception bases, the stable `error_code` values, and the `error` value type for
  code that must not throw.

## Adding an example

Add `examples/<feature>/<name>.cpp`, register it with one line in
`CMakeLists.txt`, and list it in the table above. When a feature grows, extend
the existing example rather than starting a second one — these are meant to stay
current, not accumulate.
