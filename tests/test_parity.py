import numpy as np
import parasail
import pytest

import mojo_parasail as mojo
from mojo_parasail import _lib


@pytest.fixture(scope="module")
def matrices():
    return parasail.matrix_create("ACGT", 2, -1), mojo.matrix_create("ACGT", 2, -1)


def triple(result):
    return result.score, result.end_query, result.end_ref


def test_matrix_create_matches_upstream(matrices):
    upstream, ours = matrices
    assert ours.size == upstream.size
    assert np.array_equal(ours.matrix, upstream.matrix)
    assert np.array_equal(ours.mapper, upstream.mapper)
    assert ours.min == upstream.min and ours.max == upstream.max


@pytest.mark.parametrize("query,target,open_,extend", [
    ("ACCGT", "ACG", 5, 1),
    ("GGCGTCCTTTCGTGTGGCTA", "GTGCCCCGTATGCGGCC", 6, 2),
    ("AAAA", "AA", 1, 1),
    ("GATTACA", "GCATGCU", 5, 1),
])
def test_score_and_endpoints_match_upstream(matrices, query, target, open_, extend):
    upstream, ours = matrices
    assert triple(mojo.sw(query, target, open_, extend, ours)) == triple(parasail.sw(query, target, open_, extend, upstream))


def test_randomized_score_and_endpoints_match_upstream(matrices):
    upstream, ours = matrices
    rng = np.random.default_rng(42)
    alphabet = np.array(list("ACGT"))
    for _ in range(200):
        query = "".join(rng.choice(alphabet, size=rng.integers(1, 45)))
        target = "".join(rng.choice(alphabet, size=rng.integers(1, 45)))
        open_, extend = int(rng.integers(1, 10)), int(rng.integers(1, 4))
        assert triple(mojo.sw(query, target, open_, extend, ours)) == triple(parasail.sw(query, target, open_, extend, upstream))


@pytest.mark.parametrize("target", ["ACG", "ACGT", "ACGTA", "ACGTACGTACGTACG", "ACGTACGTACGTACGT", "ACGTACGTACGTACGTA"])
def test_score_matches_upstream_across_simd_initialization_tail(matrices, target):
    upstream, ours = matrices
    assert triple(mojo.sw("GATTACA", target, 5, 1, ours)) == triple(parasail.sw("GATTACA", target, 5, 1, upstream))


def test_score_reuses_thread_local_scratch(matrices):
    _, matrix = matrices
    query = np.frombuffer(b"ACGT", dtype=np.uint8)
    target = np.frombuffer(b"ACGTACGT", dtype=np.uint8)
    _lib.score(query, target, matrix._lookup.ravel(), 5, 1)
    work, gaps = _lib._scratch.work, _lib._scratch.gaps
    _lib.score(query, target[:3], matrix._lookup.ravel(), 5, 1)
    assert _lib._scratch.work is work
    assert _lib._scratch.gaps is gaps


def test_ffi_rejects_strided_or_wrongly_typed_buffers(matrices):
    _, matrix = matrices
    query = np.frombuffer(b"ACGT", dtype=np.uint8)
    target = np.frombuffer(b"ACGTACGT", dtype=np.uint8)
    with pytest.raises(ValueError, match="C-contiguous"):
        _lib.score(query[::2], target, matrix._lookup.ravel(), 5, 1)
    with pytest.raises(TypeError, match="uint8"):
        _lib.score(query.astype(np.int8), target, matrix._lookup.ravel(), 5, 1)
    with pytest.raises(ValueError, match="exactly"):
        _lib.score(query, target, matrix._lookup.ravel()[:-1], 5, 1)


def test_trace_cigar_and_stats_match_published_example(matrices):
    upstream, ours = matrices
    query, target = "ACCGT", "ACG"
    reference, candidate = parasail.sw_trace(query, target, 5, 1, upstream), mojo.sw_trace(query, target, 5, 1, ours)
    assert triple(candidate) == triple(reference)
    assert candidate.cigar.decode == reference.cigar.decode == b"2="
    assert (candidate.cigar.beg_query, candidate.cigar.beg_ref) == (reference.cigar.beg_query, reference.cigar.beg_ref) == (0, 0)
    reference_stats, candidate_stats = parasail.sw_stats(query, target, 5, 1, upstream), mojo.sw_stats(query, target, 5, 1, ours)
    assert (candidate_stats.score, candidate_stats.matches, candidate_stats.similar, candidate_stats.length) == (
        reference_stats.score, reference_stats.matches, reference_stats.similar, reference_stats.length
    )


def test_table_and_rowcol_match_upstream(matrices):
    upstream, ours = matrices
    query, target = "ACGTAC", "ATAC"
    reference_table, candidate_table = parasail.sw_table(query, target, 4, 1, upstream), mojo.sw_table(query, target, 4, 1, ours)
    assert triple(candidate_table) == triple(reference_table)
    assert np.array_equal(candidate_table.score_table, reference_table.score_table)
    reference_rowcol, candidate_rowcol = parasail.sw_rowcol(query, target, 4, 1, upstream), mojo.sw_rowcol(query, target, 4, 1, ours)
    assert np.array_equal(candidate_rowcol.score_row, reference_rowcol.score_row)
    assert np.array_equal(candidate_rowcol.score_col, reference_rowcol.score_col)


def test_backend_and_width_aliases_are_equivalent(matrices):
    _, matrix = matrices
    expected = triple(mojo.sw("GATTACA", "GCATGCU", 5, 1, matrix))
    for name in ("sw_diag_8", "sw_scan_16", "sw_striped_32", "sw_64", "sw_sat"):
        assert triple(getattr(mojo, name)("GATTACA", "GCATGCU", 5, 1, matrix)) == expected


def test_profile_keeps_query_and_matrix(matrices):
    _, matrix = matrices
    profile = mojo.profile_create_16("ACCGT", matrix)
    assert triple(mojo.sw_striped_16(profile, "ACG", 5, 1, matrix)) == (4, 1, 1)


def test_input_validation(matrices):
    _, matrix = matrices
    with pytest.raises(ValueError):
        mojo.sw("A", "A", -1, 1, matrix)
    with pytest.raises(TypeError):
        mojo.sw("A", "A", 1, 1, object())
    with pytest.raises(OverflowError):
        mojo.matrix_create("AC", 1 << 40, -1)
    with pytest.raises(ValueError):
        mojo.matrix_create("AÅ", 2, -1)
    with pytest.raises(IndexError):
        matrix.set_value(matrix.size, 0, 1)
