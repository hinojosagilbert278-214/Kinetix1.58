"""STEPGATE risk/uncertainty gate (section 2.3, 3.1, 4.2)."""
from dataclasses import dataclass

LOCAL, CLOUD = "local", "cloud"


@dataclass
class Gate:
    base_local: float = 0.35  # tau_L
    base_cloud: float = 0.65  # tau_C
    window: int = 2048        # W_SLM, in tokens
    alpha_min: float = 0.3    # minimum draft acceptance rate

    def thresholds(self, risk: float, budget: float) -> tuple[float, float]:
        """Higher risk tightens cutoffs; larger budget (0..1) loosens the cloud one. tau_L < tau_C always."""
        tl = self.base_local * (1 - 0.5 * risk)
        tc = self.base_cloud * (1 - 0.3 * risk) * (0.7 + 0.3 * budget)
        return tl, max(tc, tl + 1e-6)

    def decide(self, q_t: float, context_tokens: int, risk=0.0, budget=1.0,
               multi_doc_graph=False, acceptance: float | None = None) -> str:
        tl, tc = self.thresholds(risk, budget)
        if context_tokens > self.window or multi_doc_graph:
            return CLOUD
        if acceptance is not None and acceptance < self.alpha_min:
            return CLOUD
        if q_t > tc:
            return CLOUD
        return LOCAL  # q_t <= tau_L runs local; (tau_L, tau_C] runs local with verification
