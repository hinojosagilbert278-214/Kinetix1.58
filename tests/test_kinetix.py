import random

from kinetix.acespec import AceSpec
from kinetix.bitnet import BitLinear, pack_ternary, quantize_ternary, unpack_ternary
from kinetix.pipeline import HybridEngine
from kinetix.retrieval import Doc, DualStore, HybridIndex
from kinetix.router import HYBRID, PRIVATE, PUBLIC, classify
from kinetix.stepgate import CLOUD, LOCAL, Gate

DOC = """# Leave Policy
Employees accrue 20 days of leave.
Exceptions require manager approval.
# Security Policy
Passwords rotate every 90 days.
"""


def test_router():
    assert classify("latest news on rust").name == PUBLIC
    assert classify("summarize my uploaded policy").name == PRIVATE
    assert classify("compare our internal spec to the latest RFC").name == HYBRID


def test_parent_child_returns_full_section():
    s = DualStore()
    s.ingest(DOC)
    top = s.search("manager approval exceptions", 1)[0]
    assert "20 days" in top.text and "Exceptions" in top.text
    assert "Passwords" not in top.text


def test_hybrid_index():
    i = HybridIndex()
    i.add(Doc("a", "bitnet ternary weights"))
    i.add(Doc("b", "cooking pasta recipes"))
    assert i.search("ternary bitnet")[0][0].id == "a"


def test_pack_roundtrip_and_ternary():
    row = [random.choice([-1, 0, 1]) for _ in range(23)]
    assert unpack_ternary(pack_ternary(row), 23) == row
    q, _ = quantize_ternary([[0.9, -1.2, 0.01, 0.4]])
    assert set(q[0]) <= {-1, 0, 1}


def test_bitlinear_sparse():
    rnd = random.Random(0)
    w = [[rnd.gauss(0, 1) for _ in range(64)] for _ in range(8)]
    x = [rnd.gauss(0, 1) for _ in range(64)]
    layer = BitLinear(w, n=2, m=4)
    assert layer.sparsity() >= 0.5
    assert len(layer(x)) == 8


def test_gate():
    g = Gate()
    assert g.decide(0.1, 100) == LOCAL
    assert g.decide(0.9, 100) == CLOUD
    assert g.decide(0.1, 10_000) == CLOUD
    assert g.decide(0.1, 100, multi_doc_graph=True) == CLOUD
    tl, tc = g.thresholds(1.0, 0.0)
    assert tl < tc


def test_acespec_matches_target_and_uses_cache():
    vocab = list(range(10))

    def target(p):
        t = (sum(p) * 7 + len(p)) % 10
        return [t] + [v for v in vocab if v != t]

    def draft(p):
        t, alt = target(p)[0], target(p)[1]
        return [alt, t] if len(p) % 5 == 0 else [t, alt]

    spec = AceSpec(draft, target)
    out = spec.generate([1, 2], 20)
    ref, cur = [], (1, 2)
    for _ in range(20):
        t = target(cur)[0]
        ref.append(t)
        cur += (t,)
    assert out == ref
    assert spec.stats.cache_hits > 0 or spec.stats.cache_misses > 0


def test_pipeline_routing():
    web = HybridIndex()
    web.add(Doc("w", "latest rfc standard for http"))
    priv = DualStore()
    priv.ingest(DOC)
    eng = HybridEngine(web, priv, lambda q, c: "local", lambda q, c: "cloud")
    assert eng.ask("summarize my leave policy").engine == "local"
    assert eng.ask("compare our internal spec to the latest rfc").engine == "cloud"
