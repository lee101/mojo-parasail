"""Benchmark score-only local alignment against the upstream parasail wheel."""

from __future__ import annotations

import math
import os
import sys
import time

import numpy as np
import parasail

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"))
import mojo_parasail as mojo  # noqa: E402


def best_time(fn, repeats=5):
    best = math.inf
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - start)
    return best


def main():
    rng = np.random.default_rng(0)
    print("| case | mojo-parasail | parasail | ratio | result |")
    print("| --- | ---: | ---: | ---: | --- |")
    for length in (500, 2000):
        query = "".join(rng.choice(np.array(list("ACGT")), size=length))
        target = "".join(rng.choice(np.array(list("ACGT")), size=length))
        ours_matrix, upstream_matrix = mojo.matrix_create("ACGT", 2, -1), parasail.matrix_create("ACGT", 2, -1)
        ours = lambda: mojo.sw(query, target, 5, 1, ours_matrix)
        upstream = lambda: parasail.sw(query, target, 5, 1, upstream_matrix)
        ours()
        upstream()
        a, b = best_time(ours), best_time(upstream)
        verdict = "faster" if a < b else "slower"
        print(f"| SW score ({length} x {length}) | {a * 1e3:.2f} ms | {b * 1e3:.2f} ms | {b / a:.2f}x | {verdict} |")


if __name__ == "__main__":
    main()
