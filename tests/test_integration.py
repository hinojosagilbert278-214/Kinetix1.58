"""End-to-end: route -> retrieve -> gate -> local draft with AceSpec verification -> cloud escalation."""
from kinetix.acespec import AceSpec
from kinetix.pipeline import HybridEngine
from kinetix.retrieval import Doc, DualStore, HybridIndex
from kinetix.stepgate import CLOUD, LOCAL, Gate

DOC = "# Leave Policy\nEmployees accrue 20 days of leave.\n# Security\nPasswords rotate every 90 days.\n"


def _target(p):
    t = (sum(p) * 7 + len(p)) % 10
    return [t] + [v for v in range(10) if v != t]


def _engine():
    web = HybridIndex()
    web.add(Doc("w", "latest rfc standard for http"))
    priv = DualStore()
    priv.ingest(DOC)
    return HybridEngine(web, priv, lambda q, c: "local", lambda q, c: "cloud")


def test_private_query_stays_local_and_uses_parent_context():
    a = _engine().ask("summarize my leave policy")
    assert (a.route, a.engine) == ("private", LOCAL)
    assert any("20 days" in c for c in a.contexts)


def test_hybrid_and_uncertain_queries_escalate():
    e = _engine()
    assert e.ask("compare our internal spec to the latest rfc").engine == CLOUD
    assert e.ask("summarize my leave policy", uncertainty=0.9).engine == CLOUD


def test_poor_draft_triggers_fail_up_after_speculation():
    bad_draft = lambda p: [(_target(p)[0] + 1) % 10, _target(p)[0]]  # never agrees on top-1
    spec = AceSpec(bad_draft, _target)
    out = spec.generate([1], 8)
    assert out == [t for t in _greedy(8)]
    assert spec.stats.acceptance < Gate().alpha_min
    assert Gate().decide(0.1, 100, acceptance=spec.stats.acceptance) == CLOUD


def test_good_draft_stays_local_and_cheap_uplink():
    spec = AceSpec(_target, _target)
    spec.generate([1], 20)
    assert spec.stats.acceptance == 1.0
    assert Gate().decide(0.1, 100, acceptance=spec.stats.acceptance) == LOCAL
    assert spec.stats.uplink_tokens <= 20 + spec.gamma


def _greedy(n):
    cur, out = (1,), []
    for _ in range(n):
        t = _target(cur)[0]
        out.append(t)
        cur += (t,)
    return out
