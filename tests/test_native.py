import random
import subprocess

import pytest

from kinetix import native
from kinetix.bitnet import apply_nm_sparsity, pack_ternary, quantize_ternary

subprocess.run(["make", "lib"], check=False, capture_output=True)
pytestmark = pytest.mark.skipif(not native.available(), reason="libbitnet.so not built")


def _rows(rnd, r, c):
    w = [[rnd.gauss(0, 1) for _ in range(c)] for _ in range(r)]
    q, _ = quantize_ternary(w)
    return w, q


def test_pack_matches_python():
    rnd = random.Random(1)
    row = [rnd.choice([-1, 0, 1]) for _ in range(37)]
    assert native.pack(row) == pack_ternary(row)


def test_gemv_ternary_matches_reference():
    rnd = random.Random(2)
    _, q = _rows(rnd, 6, 50)
    x = [rnd.randint(-127, 127) for _ in range(50)]
    ref = [sum(w * a for w, a in zip(r, x)) for r in q]
    assert native.gemv_ternary([pack_ternary(r) for r in q], 50, x) == ref


def test_sparse24_matches_dense():
    rnd = random.Random(3)
    w, q = _rows(rnd, 5, 32)
    q = [apply_nm_sparsity(wr, qr, 2, 4) for wr, qr in zip(w, q)]
    x = [rnd.randint(-127, 127) for _ in range(32)]
    ref = [sum(a * b for a, b in zip(r, x)) for r in q]
    assert native.gemv_sparse24(native.sparse24_compress(q), 5, 32, x) == ref


def test_sparse24_rejects_dense_group():
    with pytest.raises(ValueError):
        native.sparse24_compress([[1, 1, 1, 0]])


def test_xnor_dot():
    rnd = random.Random(4)
    a = [rnd.choice([-1, 1]) for _ in range(131)]
    w = [rnd.choice([-1, 1]) for _ in range(131)]
    assert native.xnor_dot(a, w) == sum(x * y for x, y in zip(a, w))
