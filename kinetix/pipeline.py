"""End-to-end orchestration: route -> retrieve -> gate -> local/cloud."""
from dataclasses import dataclass
from typing import Callable

from .retrieval import DualStore, HybridIndex, tokenize
from .router import HYBRID, PRIVATE, PUBLIC, classify
from .stepgate import CLOUD, Gate


@dataclass
class Answer:
    route: str
    engine: str
    contexts: list[str]
    text: str


class HybridEngine:
    def __init__(self, web: HybridIndex, private: DualStore,
                 local_llm: Callable[[str, list[str]], str],
                 cloud_llm: Callable[[str, list[str]], str],
                 gate: Gate | None = None):
        self.web, self.private = web, private
        self.local_llm, self.cloud_llm = local_llm, cloud_llm
        self.gate = gate or Gate()

    def ask(self, query: str, uncertainty: float = 0.1, risk: float = 0.0, budget: float = 1.0) -> Answer:
        route = classify(query).name
        ctx: list[str] = []
        if route in (PUBLIC, HYBRID):
            ctx += [d.text for d, _ in self.web.search(query, 2)]
        if route in (PRIVATE, HYBRID):
            ctx += [p.text for p in self.private.search(query, 2)]
        ctx_tokens = sum(len(tokenize(c)) for c in ctx) + len(tokenize(query))
        engine = self.gate.decide(uncertainty, ctx_tokens, risk, budget, multi_doc_graph=route == HYBRID)
        llm = self.cloud_llm if engine == CLOUD else self.local_llm
        return Answer(route, engine, ctx, llm(query, ctx))
