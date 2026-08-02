"""A focused, compatible Python API for SIMD-style Smith--Waterman alignment."""

from .core import (
    Cigar,
    Matrix,
    Profile,
    Result,
    matrix_create,
    profile_create,
    profile_create_8,
    profile_create_16,
    profile_create_32,
    profile_create_64,
    profile_create_sat,
    sw,
    sw_diag,
    sw_diag_8,
    sw_diag_16,
    sw_diag_32,
    sw_diag_64,
    sw_diag_sat,
    sw_rowcol,
    sw_scan,
    sw_striped,
    sw_stats,
    sw_table,
    sw_trace,
)
from ._lib import build

__version__ = "0.1.0"

# The width-specific parasail entry points retain the same mathematical result.
for _family in ("sw", "sw_diag", "sw_scan", "sw_striped"):
    for _width in ("8", "16", "32", "64", "sat"):
        globals().setdefault(f"{_family}_{_width}", globals()[_family])
for _family in ("sw_trace", "sw_stats", "sw_table", "sw_rowcol"):
    for _backend in ("diag", "scan", "striped"):
        globals()[f"{_family}_{_backend}"] = globals()[_family]
        for _width in ("8", "16", "32", "64", "sat"):
            globals()[f"{_family}_{_backend}_{_width}"] = globals()[_family]
    for _width in ("8", "16", "32", "64", "sat"):
        globals()[f"{_family}_{_width}"] = globals()[_family]

__all__ = [name for name in globals() if not name.startswith("_")]
