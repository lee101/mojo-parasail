"""Loading and narrow ctypes bindings for the Mojo score kernel."""

from __future__ import annotations

import ctypes
import os
import shutil
import subprocess
import sys
import threading

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LIB = os.path.join(ROOT, "dist", "libmojo-parasail.so")
SOURCE = os.path.join(ROOT, "src", "capi.mojo")
I = ctypes.c_int64
_INT32 = np.iinfo(np.int32)
_STRIPED_MIN_CELLS = 65_536


def build(force: bool = False) -> str:
    """Build the shared library when it is absent or older than its source."""
    if not force and os.path.exists(LIB) and os.path.getmtime(LIB) >= os.path.getmtime(SOURCE):
        return LIB
    script = os.path.join(ROOT, "build", "build.sh")
    proc = subprocess.run(["bash", script], capture_output=True, text=True, timeout=600)
    if proc.returncode or not os.path.exists(LIB):
        raise RuntimeError((proc.stderr or proc.stdout).strip())
    return LIB


_library: ctypes.CDLL | None = None
_scratch = threading.local()


def lib() -> ctypes.CDLL:
    global _library
    if _library is None:
        _library = ctypes.CDLL(build())
        _library.mps_sw_score.argtypes = [I] * 10
        _library.mps_sw_score.restype = I
        _library.mps_sw_striped.argtypes = [I] * 14
        _library.mps_sw_striped.restype = I
    return _library


def score(
    query: np.ndarray,
    target: np.ndarray,
    lookup: np.ndarray,
    open_: int,
    extend: int,
    mapper: np.ndarray | None = None,
    matrix: np.ndarray | None = None,
):
    """Run the O(|target|) affine-gap score kernel and return score/endpoints."""
    # This function owns the only raw-pointer crossing in the package.  Do not
    # rely on callers using the same arrays as the public Python API: ctypes
    # will happily hand Mojo an address for a strided or differently typed view.
    for name, array, dtype, size in (
        ("query", query, np.dtype(np.uint8), None),
        ("target", target, np.dtype(np.uint8), None),
        ("lookup", lookup, np.dtype(np.int64), 256 * 256),
    ):
        if not isinstance(array, np.ndarray) or array.dtype != dtype:
            raise TypeError(f"{name} must be a NumPy array with dtype {dtype}")
        if array.ndim != 1 or not array.flags.c_contiguous:
            raise ValueError(f"{name} must be one-dimensional and C-contiguous")
        if size is not None and array.size != size:
            raise ValueError(f"{name} must contain exactly {size} entries")
        if array.size and array.ctypes.data == 0:
            raise ValueError(f"{name} must have a non-null data pointer")
    if not isinstance(open_, int) or isinstance(open_, bool) or not isinstance(extend, int) or isinstance(extend, bool):
        raise TypeError("open and extend penalties must be integers")
    if not 0 <= open_ <= _INT32.max or not 0 <= extend <= _INT32.max:
        raise OverflowError("penalties must fit in a signed 32-bit integer")
    if not query.size or not target.size:
        raise ValueError("query and target must be non-empty")
    if query.size * target.size >= _STRIPED_MIN_CELLS and mapper is not None and matrix is not None:
        if mapper.dtype != np.int32 or mapper.shape != (256,) or not mapper.flags.c_contiguous:
            raise ValueError("mapper must be a contiguous 256-entry int32 array")
        if matrix.dtype != np.int32 or matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or not matrix.flags.c_contiguous:
            raise ValueError("matrix must be a contiguous square int32 array")
        padded_capacity = query.size + 64
        matrix_size = matrix.shape[0]
        for name, size in (
            ("striped_store", padded_capacity),
            ("striped_load", padded_capacity),
            ("striped_gaps", padded_capacity),
            ("striped_profile", matrix_size * padded_capacity),
            ("striped_result", 2),
        ):
            array = getattr(_scratch, name, None)
            if array is None or array.size < size:
                setattr(_scratch, name, np.empty(size, dtype=np.int64))
        store = _scratch.striped_store
        load = _scratch.striped_load
        gaps = _scratch.striped_gaps
        profile = _scratch.striped_profile
        result = _scratch.striped_result
        value = lib().mps_sw_striped(
            int(query.ctypes.data), int(target.ctypes.data),
            int(matrix.ctypes.data), int(mapper.ctypes.data),
            int(store.ctypes.data), int(load.ctypes.data), int(gaps.ctypes.data),
            int(profile.ctypes.data), int(result.ctypes.data), query.size, target.size,
            open_, extend, matrix_size,
        )
        return int(value), int(result[0]), int(result[1])
    required = target.size + 3
    work = getattr(_scratch, "work", None)
    gaps = getattr(_scratch, "gaps", None)
    if work is None or work.size < required:
        work = np.empty(required, dtype=np.int64)
        _scratch.work = work
    if gaps is None or gaps.size < target.size + 1:
        gaps = np.empty(target.size + 1, dtype=np.int64)
        _scratch.gaps = gaps
    value = lib().mps_sw_score(
        int(query.ctypes.data), int(target.ctypes.data), int(lookup.ctypes.data),
        int(work.ctypes.data), int(gaps.ctypes.data), query.size, target.size,
        open_, extend, 256,
    )
    return int(value), int(work[target.size + 1]), int(work[target.size + 2])


if __name__ == "__main__":
    print(build(force="--force" in sys.argv))
