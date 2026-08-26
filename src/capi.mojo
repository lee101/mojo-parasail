"""C ABI for the local affine-gap Smith--Waterman score kernel."""

from std.sys.info import simd_width_of as simdwidthof

comptime I64Ptr = UnsafePointer[Int, AnyOrigin[mut=True]]
comptime I32Ptr = UnsafePointer[Int32, AnyOrigin[mut=True]]
comptime U8Ptr = UnsafePointer[UInt8, AnyOrigin[mut=True]]
comptime W = simdwidthof[DType.float64]()


def ip(addr: Int) -> I64Ptr:
    return I64Ptr(unsafe_from_address=addr)


def bp(addr: Int) -> U8Ptr:
    return U8Ptr(unsafe_from_address=addr)


def i32p(addr: Int) -> I32Ptr:
    return I32Ptr(unsafe_from_address=addr)


def shift_lanes_left(value: SIMD[DType.int, W]) -> SIMD[DType.int, W]:
    var shifted = SIMD[DType.int, W](0)
    comptime for lane in range(1, W):
        shifted[lane] = value[lane - 1]
    return shifted


@export("mps_sw_score")
def mps_sw_score(
    query_addr: Int,
    ref_addr: Int,
    table_addr: Int,
    h_addr: Int,
    e_addr: Int,
    query_len: Int,
    ref_len: Int,
    open_penalty: Int,
    extend_penalty: Int,
    alphabet_size: Int,
) abi("C") -> Int:
    """Return score; write zero-based end positions to h[ref_len+1:ref_len+3]."""
    var h = ip(h_addr)
    var e = ip(e_addr)
    var query = bp(query_addr)
    var target = bp(ref_addr)
    var table = ip(table_addr)
    var j = 0
    var zero = SIMD[DType.int, W](0)
    var neg = SIMD[DType.int, W](-2147483647)
    while j + W <= ref_len + 1:
        h.store(j, zero)
        e.store(j, neg)
        j += W
    while j < ref_len + 1:
        h[j] = 0
        e[j] = -2147483647
        j += 1
    var best = 0
    var best_query = -1
    var best_ref = -1
    for i in range(query_len):
        var diagonal = h[0]
        h[0] = 0
        var f = -2147483647
        var table_row = Int(query[i]) * alphabet_size
        for j in range(ref_len):
            var above = h[j + 1]
            var open_e = above - open_penalty
            var extend_e = e[j + 1] - extend_penalty
            e[j + 1] = max(open_e, extend_e)
            var open_f = h[j] - open_penalty
            var extend_f = f - extend_penalty
            f = max(open_f, extend_f)
            var r = Int(target[j])
            var score = diagonal + table[table_row + r]
            var cell = max(0, max(score, max(e[j + 1], f)))
            h[j + 1] = cell
            diagonal = above
            # parasail resolves score ties at the leftmost reference endpoint.
            if cell > best or (cell == best and (best_ref < 0 or j < best_ref)):
                best = cell
                best_query = i
                best_ref = j
    h[ref_len + 1] = best_query
    h[ref_len + 2] = best_ref
    return best


@export("mps_sw_striped")
def mps_sw_striped(
    query_addr: Int,
    ref_addr: Int,
    matrix_addr: Int,
    mapper_addr: Int,
    h_store_addr: Int,
    h_load_addr: Int,
    e_addr: Int,
    profile_addr: Int,
    result_addr: Int,
    query_len: Int,
    ref_len: Int,
    open_penalty: Int,
    extend_penalty: Int,
    matrix_size: Int,
) abi("C") -> Int:
    var query = bp(query_addr)
    var target = bp(ref_addr)
    var matrix = i32p(matrix_addr)
    var mapper = i32p(mapper_addr)
    var initial_store = ip(h_store_addr)
    var h_store = initial_store
    var h_load = ip(h_load_addr)
    var e = ip(e_addr)
    var profile = ip(profile_addr)
    var result = ip(result_addr)
    var seg_len = (query_len + W - 1) // W
    var padded = seg_len * W
    var zero = SIMD[DType.int, W](0)
    var negative = SIMD[DType.int, W](-2147483647)
    var gap_initial = SIMD[DType.int, W](-open_penalty)
    for segment in range(seg_len):
        h_store.store(segment * W, zero)
        h_load.store(segment * W, zero)
        e.store(segment * W, gap_initial)
    for alphabet_index in range(matrix_size):
        for segment in range(seg_len):
            var values = SIMD[DType.int, W](0)
            comptime for lane in range(W):
                var query_index = segment + lane * seg_len
                if query_index < query_len:
                    var mapped = Int(mapper[Int(query[query_index])])
                    values[lane] = Int(
                        matrix[mapped * matrix_size + alphabet_index]
                    )
            profile.store(
                alphabet_index * padded + segment * W, values
            )
    var best = 0
    var best_query = 0
    var best_ref = 0
    for j in range(ref_len):
        var swap = h_load
        h_load = h_store
        h_store = swap
        var f = zero
        var h = shift_lanes_left(
            h_load.load[width=W]((seg_len - 1) * W)
        )
        var profile_base = Int(mapper[Int(target[j])]) * padded
        var column_max = negative
        for segment in range(seg_len):
            var offset = segment * W
            h += profile.load[width=W](profile_base + offset)
            var vertical = e.load[width=W](offset)
            h = max(zero, max(h, max(vertical, f)))
            h_store.store(offset, h)
            column_max = max(column_max, h)
            var h_open = h - open_penalty
            vertical = max(vertical - extend_penalty, h_open)
            e.store(offset, vertical)
            f = max(f - extend_penalty, h_open)
            h = h_load.load[width=W](offset)
        var lazy_done = False
        for _ in range(W):
            f = shift_lanes_left(f)
            for segment in range(seg_len):
                var offset = segment * W
                h = max(h_store.load[width=W](offset), f)
                h_store.store(offset, h)
                column_max = max(column_max, h)
                var h_open = h - open_penalty
                f -= extend_penalty
                if not any(f.gt(h_open)):
                    lazy_done = True
                    break
            if lazy_done:
                break
        var candidate = column_max.reduce_max()
        if candidate > best:
            best = candidate
            best_ref = j
            best_query = query_len - 1
            for segment in range(seg_len):
                h = h_store.load[width=W](segment * W)
                comptime for lane in range(W):
                    var query_index = segment + lane * seg_len
                    if (
                        query_index < query_len
                        and h[lane] == best
                        and query_index < best_query
                    ):
                        best_query = query_index
    result[0] = best_query
    result[1] = best_ref
    return best
