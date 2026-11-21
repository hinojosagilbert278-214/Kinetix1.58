"""Micro-benchmarks: router latency, BitLinear (Python) vs native GEMV. Run: python bench/bench.py"""
import random
import sys
import time

sys.path.insert(0, ".")
from kinetix import native
from kinetix.bitnet import BitLinear, quantize_ternary, pack_ternary
from kinetix.router import classify


def timeit(fn, n):
    t = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t) / n * 1e3


print(f"router classify: {timeit(lambda: classify('compare our internal spec to the latest rfc'), 2000):.4f} ms")
rnd = random.Random(0)
rows, cols = 256, 512
w = [[rnd.gauss(0, 1) for _ in range(cols)] for _ in range(rows)]
x = [rnd.gauss(0, 1) for _ in range(cols)]
layer = BitLinear(w, 2, 4)
print(f"BitLinear python {rows}x{cols}: {timeit(lambda: layer(x), 3):.2f} ms")
if native.available():
    q, _ = quantize_ternary(w)
    packed = [pack_ternary(r) for r in q]
    xi = [rnd.randint(-127, 127) for _ in range(cols)]
    print(f"native ternary gemv {rows}x{cols}: {timeit(lambda: native.gemv_ternary(packed, cols, xi), 50):.3f} ms (includes ctypes marshalling)")
else:
    print("native lib not built (make lib)")
