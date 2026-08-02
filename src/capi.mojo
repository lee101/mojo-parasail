"""C ABI for the local affine-gap Smith--Waterman score kernel."""

from std.sys.info import simd_width_of

comptime I64Ptr = UnsafePointer[Int, AnyOrigin[mut=True]]
comptime U8Ptr = UnsafePointer[UInt8, AnyOrigin[mut=True]]
comptime W = simd_width_of[DType.int]()


def ip(addr: Int) -> I64Ptr:
    return I64Ptr(unsafe_from_address=addr)


def bp(addr: Int) -> U8Ptr:
    return U8Ptr(unsafe_from_address=addr)


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

