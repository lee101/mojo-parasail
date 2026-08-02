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

`sw` is the compiled linear-space affine-gap implementation. The trace,
table, row/column, and statistic APIs use a NumPy full matrix because their
requested result or traceback requires full-matrix state. Width and backend
names are compatibility aliases; they do not select upstream's distinct SIMD
algorithms or saturating-overflow behaviour.

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
| SW score (500 x 500) | 1.29 ms | 0.87 ms | 0.68x | slower |
| SW score (2000 x 2000) | 20.59 ms | 13.96 ms | 0.68x | slower |

The score path reuses its thread-local, zero-copy NumPy work columns and uses
native-width SIMD with a scalar remainder for their independent initialization.
The affine recurrence itself has a loop-carried horizontal gap state, so this
linear-space layout cannot safely vectorize or parallelize the cell loop.
Parasail's mature C striped/scan kernels exploit a different SIMD formulation,
and remains faster.

GPU execution is intentionally not implemented. Each cell performs only a few
integer operations per several working-column loads and stores, well below the
arithmetic intensity at which transfer and launch overhead can pay off.

## How it works

```
Python result objects
       |
ctypes: byte and signed-integer NumPy buffer addresses
       |
src/capi.mojo: exported C ABI wrapper
       |
Mojo affine-gap recurrence: H and E columns, scalar F state
```

The ctypes boundary passes contiguous NumPy buffers as integer addresses, as
required by Mojo's non-parametric C export rules. Mojo reconstructs mutable
pointers, performs no allocation, and writes its two O(n) work columns into
Python-owned memory. The substitution matrix is expanded once to a contiguous
signed-integer byte lookup table, making each recurrence cell a pair
of sequential working-column accesses and one direct substitution lookup.

## License

MIT
