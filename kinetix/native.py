"""ctypes bindings to the C++ BitNet kernels (build with `make lib`)."""
import ctypes
import os

_LIB = os.path.join(os.path.dirname(__file__), "..", "csrc", "libbitnet.so")


def available() -> bool:
    return os.path.exists(_LIB)


def _lib():
    lib = ctypes.CDLL(_LIB)
    p = ctypes.c_void_p
    sz = ctypes.c_size_t
    lib.bn_pack.argtypes = [p, sz, p]
    lib.bn_gemv_ternary.argtypes = [p, sz, sz, p, p]
    lib.bn_sparse24_compress.argtypes = [p, sz, p]
    lib.bn_sparse24_compress.restype = ctypes.c_int
    lib.bn_gemv_sparse24.argtypes = [p, sz, sz, p, p]
    lib.bn_xnor_dot.argtypes = [p, p, sz]
    lib.bn_xnor_dot.restype = ctypes.c_int32
    return lib


def _i8(v):
    return (ctypes.c_int8 * len(v))(*v)


def pack(w):
    out = (ctypes.c_uint8 * ((len(w) + 3) // 4))()
    _lib().bn_pack(_i8(w), len(w), out)
    return bytes(out)


def gemv_ternary(packed_rows: list[bytes], cols: int, x: list[int]) -> list[int]:
    buf = b"".join(packed_rows)
    y = (ctypes.c_int32 * len(packed_rows))()
    _lib().bn_gemv_ternary(buf, len(packed_rows), cols, _i8(x), y)
    return list(y)


def sparse24_compress(rows: list[list[int]]) -> bytes:
    groups = (len(rows[0]) + 3) // 4
    out = bytearray()
    for r in rows:
        buf = (ctypes.c_uint8 * groups)()
        if _lib().bn_sparse24_compress(_i8(r), len(r), buf) != 0:
            raise ValueError("row violates 2:4 sparsity")
        out += bytes(buf)
    return bytes(out)


def gemv_sparse24(comp: bytes, rows: int, cols: int, x: list[int]) -> list[int]:
    y = (ctypes.c_int32 * rows)()
    _lib().bn_gemv_sparse24(comp, rows, cols, _i8(x), y)
    return list(y)


def xnor_dot(a: list[int], w: list[int]) -> int:
    """Dot of two +/-1 vectors."""
    def bits(v):
        words = (ctypes.c_uint64 * ((len(v) + 63) // 64))()
        for i, s in enumerate(v):
            if s > 0:
                words[i // 64] |= 1 << (i % 64)
        return words
    return _lib().bn_xnor_dot(bits(a), bits(w), len(a))
