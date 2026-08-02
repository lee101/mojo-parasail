"""Python-facing matrices, results, and trace reconstruction.

The hot score-only path is Mojo. Trace/table/stat variants intentionally keep
their O(mn) state in NumPy, because their output itself is O(mn).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import _lib

_NEG = -(1 << 60)
_INT32 = np.iinfo(np.int32)


def _int32(value, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    if not _INT32.min <= value <= _INT32.max:
        raise OverflowError(f"{name} must fit in a signed 32-bit integer")
    return value


def _int64_nonnegative(value, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    if value > _INT32.max:
        raise OverflowError(f"{name} must fit in a signed 32-bit integer")
    return value


def _bytes(value) -> np.ndarray:
    if isinstance(value, str):
        value = value.encode()
    elif isinstance(value, bytearray):
        value = bytes(value)
    if not isinstance(value, bytes):
        raise TypeError("sequence must be str or bytes")
    return np.frombuffer(value, dtype=np.uint8)


class Matrix:
    """Substitution matrix compatible with ``parasail.matrix_create``."""

    def __init__(self, alphabet: str, match: int, mismatch: int, name: str = ""):
        if not isinstance(alphabet, str) or not alphabet:
            raise TypeError("alphabet must be a non-empty string")
        try:
            alphabet.encode("ascii")
        except UnicodeEncodeError as error:
            raise ValueError("alphabet must contain only ASCII characters") from error
        if len(set(alphabet.upper())) != len(alphabet):
            raise ValueError("alphabet must contain unique characters ignoring case")
        match, mismatch = _int32(match, "match"), _int32(mismatch, "mismatch")
        self.alphabet = alphabet
        self.name = name.encode()
        self.size = len(alphabet) + 1
        self.mapper = np.full(256, len(alphabet), dtype=np.int32)
        self.matrix = np.zeros((self.size, self.size), dtype=np.int32)
        self._lookup = np.zeros((256, 256), dtype=np.int64)
        for i, char_i in enumerate(alphabet):
            for code in (ord(char_i), ord(char_i.lower())):
                self.mapper[code] = i
            for j, char_j in enumerate(alphabet):
                value = match if i == j else mismatch
                self.matrix[i, j] = value
                for left in (ord(char_i), ord(char_i.lower())):
                    for right in (ord(char_j), ord(char_j.lower())):
                        self._lookup[left, right] = value
        self.min = int(self.matrix.min())
        self.max = int(self.matrix.max())

    def copy(self):
        copied = Matrix.__new__(Matrix)
        copied.alphabet = self.alphabet
        copied.name = self.name
        copied.size = self.size
        copied.mapper = self.mapper.copy()
        copied.matrix = self.matrix.copy()
        copied._lookup = self._lookup.copy()
        copied.min, copied.max = self.min, self.max
        return copied

    def set_value(self, row: int, col: int, value: int):
        if not isinstance(row, int) or isinstance(row, bool) or not isinstance(col, int) or isinstance(col, bool):
            raise TypeError("matrix indices must be integers")
        if not 0 <= row < self.size or not 0 <= col < self.size:
            raise IndexError("matrix indices are out of range")
        value = _int32(value, "matrix value")
        self.matrix[row, col] = value
        left = np.flatnonzero(self.mapper == row)
        right = np.flatnonzero(self.mapper == col)
        self._lookup[np.ix_(left, right)] = value
        self.min, self.max = int(self.matrix.min()), int(self.matrix.max())


def matrix_create(alphabet: str, match: int, mismatch: int) -> Matrix:
    return Matrix(alphabet, match, mismatch)


@dataclass
class Profile:
    query: object
    matrix: Matrix


def profile_create(query, matrix: Matrix) -> Profile:
    return Profile(query, matrix)


profile_create_8 = profile_create_16 = profile_create_32 = profile_create_64 = profile_create_sat = profile_create


class Cigar:
    _ops = {"M": 0, "I": 1, "D": 2, "=": 7, "X": 8}

    def __init__(self, operations=(), beg_query: int = 0, beg_ref: int = 0):
        self.beg_query = beg_query
        self.beg_ref = beg_ref
        runs = []
        for op in operations:
            if runs and runs[-1][0] == op:
                runs[-1] = (op, runs[-1][1] + 1)
            else:
                runs.append((op, 1))
        self.decode = "".join(f"{count}{op}" for op, count in runs).encode()
        self.seq = np.array([(count << 4) | self._ops[op] for op, count in runs], dtype=np.uint32)
        self.len = len(runs)
        self.decode_op = None
        self.decode_len = None


class Result:
    def __init__(self, score: int, end_query: int, end_ref: int, *, query=None, ref=None, matrix=None):
        self._score = int(score)
        self._end_query = int(end_query)
        self._end_ref = int(end_ref)
        self.len_query = len(query) if query is not None else 0
        self.len_ref = len(ref) if ref is not None else 0
        self.query, self.ref, self.matrix = query, ref, matrix
        self._cigar = None

    @property
    def score(self): return self._score
    @property
    def end_query(self): return self._end_query
    @property
    def end_ref(self): return self._end_ref
    @property
    def saturated(self): return False
    @property
    def cigar(self):
        if self._cigar is None:
            raise AttributeError("Result has no traceback")
        return self._cigar

    def get_cigar(self, *args, **kwargs): return self.cigar


def _validate(open_: int, extend: int, matrix: Matrix):
    if not isinstance(matrix, Matrix):
        raise TypeError("matrix must be a mojo_parasail.Matrix")
    _int64_nonnegative(open_, "open penalty")
    _int64_nonnegative(extend, "extend penalty")


def _profile_query(query, matrix):
    if isinstance(query, Profile):
        if matrix is not query.matrix:
            matrix = query.matrix
        query = query.query
    return query, matrix


def sw(s1, s2, open: int, extend: int, matrix: Matrix) -> Result:
    s1, matrix = _profile_query(s1, matrix)
    _validate(open, extend, matrix)
    query, target = _bytes(s1), _bytes(s2)
    if not query.size or not target.size:
        raise ValueError("sequences must be non-empty")
    value, end_query, end_ref = _lib.score(query, target, matrix._lookup.ravel(), open, extend)
    return Result(value, end_query, end_ref, query=s1, ref=s2, matrix=matrix)


def _full(query: np.ndarray, target: np.ndarray, open_: int, extend: int, matrix: Matrix):
    m, n = len(query), len(target)
    h = np.zeros((m + 1, n + 1), dtype=np.int64)
    e = np.full((m + 1, n + 1), _NEG, dtype=np.int64)
    f = np.full((m + 1, n + 1), _NEG, dtype=np.int64)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            e[i, j] = max(h[i - 1, j] - open_, e[i - 1, j] - extend)
            f[i, j] = max(h[i, j - 1] - open_, f[i, j - 1] - extend)
            h[i, j] = max(0, h[i - 1, j - 1] + matrix._lookup[query[i - 1], target[j - 1]], e[i, j], f[i, j])
    maximum = h[1:, 1:].max()
    candidates = np.argwhere(h[1:, 1:] == maximum) + 1
    leftmost_column = candidates[:, 1].min()
    where = candidates[candidates[:, 1] == leftmost_column][0]
    return h, e, f, int(where[0] - 1), int(where[1] - 1)


def _trace(query, target, h, e, f, end_i, end_j, open_, extend, matrix):
    i, j, operations = end_i + 1, end_j + 1, []
    state = "H"
    while i > 0 and j > 0:
        if state == "H":
            if h[i, j] == 0:
                break
            if h[i, j] == f[i, j]:
                state = "F"
            elif h[i, j] == e[i, j]:
                state = "E"
            else:
                operations.append("=" if query[i - 1] == target[j - 1] else "X")
                i, j = i - 1, j - 1
        elif state == "E":
            operations.append("I")
            state = "H" if e[i, j] == h[i - 1, j] - open_ else "E"
            i -= 1
        else:
            operations.append("D")
            state = "H" if f[i, j] == h[i, j - 1] - open_ else "F"
            j -= 1
    return Cigar(reversed(operations), i, j)


def sw_trace(s1, s2, open: int, extend: int, matrix: Matrix) -> Result:
    s1, matrix = _profile_query(s1, matrix)
    _validate(open, extend, matrix)
    query, target = _bytes(s1), _bytes(s2)
    if not query.size or not target.size:
        raise ValueError("sequences must be non-empty")
    h, e, f, end_i, end_j = _full(query, target, int(open), int(extend), matrix)
    result = Result(int(h.max()), end_i, end_j, query=s1, ref=s2, matrix=matrix)
    result._cigar = _trace(query, target, h, e, f, end_i, end_j, int(open), int(extend), matrix)
    return result


def sw_stats(s1, s2, open: int, extend: int, matrix: Matrix) -> Result:
    result = sw_trace(s1, s2, open, extend, matrix)
    ops = result.cigar.decode.decode()
    query, target = _bytes(result.query), _bytes(result.ref)
    qi, ri, matches, similar, length = result.cigar.beg_query, result.cigar.beg_ref, 0, 0, 0
    import re
    for count, op in re.findall(r"(\d+)([=XID])", ops):
        count = int(count)
        for _ in range(count):
            if op in "=X":
                matches += int(query[qi] == target[ri])
                similar += int(matrix._lookup[query[qi], target[ri]] > 0)
                qi += 1; ri += 1
            elif op == "I": qi += 1
            else: ri += 1
            length += 1
    result.matches, result.similar, result.length = matches, similar, length
    return result


def sw_table(s1, s2, open: int, extend: int, matrix: Matrix) -> Result:
    s1, matrix = _profile_query(s1, matrix)
    _validate(open, extend, matrix)
    query, target = _bytes(s1), _bytes(s2)
    if not query.size or not target.size:
        raise ValueError("sequences must be non-empty")
    h, _, _, end_i, end_j = _full(query, target, int(open), int(extend), matrix)
    result = Result(int(h.max()), end_i, end_j, query=s1, ref=s2, matrix=matrix)
    result.score_table = h[1:, 1:].astype(np.int32)
    return result


def sw_rowcol(s1, s2, open: int, extend: int, matrix: Matrix) -> Result:
    s1, matrix = _profile_query(s1, matrix)
    _validate(open, extend, matrix)
    query, target = _bytes(s1), _bytes(s2)
    if not query.size or not target.size:
        raise ValueError("sequences must be non-empty")
    h, _, _, end_i, end_j = _full(query, target, int(open), int(extend), matrix)
    result = Result(int(h.max()), end_i, end_j, query=s1, ref=s2, matrix=matrix)
    result.score_row, result.score_col = h[-1, 1:].astype(np.int32), h[1:, -1].astype(np.int32)
    return result


sw_diag = sw_scan = sw_striped = sw
sw_diag_8 = sw_diag_16 = sw_diag_32 = sw_diag_64 = sw_diag_sat = sw
