# mojo-parasail

`mojo-parasail` is a focused Mojo port of the compute-bound core of
[parasail](https://github.com/jeffdaily/parasail): local Smith--Waterman
alignment with an affine gap penalty. It exposes a familiar Python API while
putting the score-only dynamic-programming kernel in a compiled Mojo shared
library.

```python
import mojo_parasail as parasail

matrix = parasail.matrix_create("ACGT", 2, -1)
result = parasail.sw("ACCGT", "ACG", 5, 1, matrix)
assert (result.score, result.end_query, result.end_ref) == (4, 1, 1)

trace = parasail.sw_trace("ACCGT", "ACG", 5, 1, matrix)
assert trace.cigar.decode == b"2="
```

## Coverage

Implemented API surface:

| area | API |
| --- | --- |
| substitution matrices | `Matrix`, `matrix_create`, `set_value`, `copy` |
| score-only local alignment | `sw`, `sw_diag`, `sw_scan`, `sw_striped`, and width/saturation suffix aliases |
| trace and derived results | `sw_trace`, `sw_stats`, `sw_table`, `sw_rowcol`, including backend/width aliases |
| profiles | `Profile`, `profile_create`, and width suffix aliases |

`sw` is the compiled linear-space affine-gap implementation. It uses a scalar
kernel for small alignments and a native-width striped SIMD kernel once the
matrix reaches 65,536 cells. The trace,
table, row/column, and statistic APIs use a NumPy full matrix because their
requested result or traceback requires full-matrix state. Width and backend
names are compatibility aliases; they do not select different integer widths
or saturating-overflow behaviour.

Not yet covered: global/semi-global (`nw`, `sg`) alignment, built-in biological
matrices such as `blosum62`, PSSMs, scan/striped profile-only calling forms,
and the upstream traceback rendering helpers. Use `matrix_create` for custom
DNA/protein alphabets in the covered subset.

## Install and verify

```bash
pixi install
pixi run build
pixi run test
pixi run bench
```

The Pixi environment installs the published `parasail` wheel solely for
parity testing and benchmarking. The `mojo_parasail` runtime does not import
or call it.

## Parity

The tests compare matrix mapping, local-alignment scores and endpoints, trace
and statistics, score tables, row/column outputs, compatibility aliases,
profiles, and boundary validation against the installed upstream parasail
wheel. The comparisons assert returned values, not merely successful calls.

## Benchmark

Measured locally with `pixi run bench`. Times are the best of five single-call
measurements, including the Python/ctypes boundary. The initial Mojo call
allocates its reusable scratch columns before timing.

| case | mojo-parasail | parasail | ratio | result |
| --- | ---: | ---: | ---: | --- |
| SW score (500 x 500) | 0.84 ms | 0.87 ms | 1.04x | faster |
| SW score (2000 x 2000) | 12.24 ms | 14.42 ms | 1.18x | faster |

The large score path builds a compact striped query profile in reusable
thread-local scratch, evaluates native-width SIMD vectors, and applies a lazy
horizontal-gap correction. Padded SIMD lanes are excluded from endpoint
selection. Smaller matrices stay on the scalar kernel to avoid profile setup
overhead. Both paths keep sequence and matrix NumPy buffers zero-copy across
the FFI boundary, and all scratch allocations are reused.

CPU threading is intentionally not used. A single alignment's row, column, and
lazy-gap updates are dependency-linked; synchronizing wavefronts would add
overhead to the input sizes served by this API. There is no independent batch
dimension to parallelize.

GPU execution is intentionally not implemented. Each striped cell performs
only a handful of integer operations while loading and storing several score
vectors, comfortably below one operation per byte and therefore well below the
roughly two-operations-per-byte threshold at which transfer and launch overhead
could pay off.

## How it works

```
Python result objects
       |
ctypes: byte and signed-integer NumPy buffer addresses
       |
src/capi.mojo: exported C ABI wrapper
       |
Mojo affine-gap recurrence: scalar or striped SIMD with lazy F correction
```

The ctypes boundary passes contiguous NumPy buffers as integer addresses, as
required by Mojo's non-parametric C export rules. Mojo reconstructs mutable
pointers and performs no allocation. The scalar kernel uses two O(n) columns
and a direct byte-pair lookup. The striped kernel uses three O(m) score vectors
plus an O(alphabet × m) query profile. All of that storage is Python-owned,
thread-local, and retained for later calls.

## License

MIT
