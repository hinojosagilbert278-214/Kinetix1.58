"""AceSpec asymmetric edge-cloud speculative decoding (section 4.1), simulated on token lists.

The edge drafts a main chain and, while waiting for the cloud, caches alternative
branches keyed by (prefix, token). Only the main chain is uplinked. On rejection the
cloud returns the corrected token; if the edge pre-computed that branch it resumes via
an O(1) dict lookup instead of re-drafting.
"""
from dataclasses import dataclass, field
from typing import Callable

# model(prefix) -> ranked list of next-token candidates (best first)
Model = Callable[[tuple], list]


@dataclass
class Stats:
    cloud_rounds: int = 0
    uplink_tokens: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    accepted: int = 0
    drafted: int = 0

    @property
    def acceptance(self) -> float:
        return self.accepted / self.drafted if self.drafted else 1.0


@dataclass
class AceSpec:
    draft: Model
    target: Model
    gamma: int = 4         # main-chain length per round
    branch_width: int = 2  # candidates considered per position
    stats: Stats = field(default_factory=Stats)

    def _greedy_continue(self, prefix: tuple) -> list:
        out, cur = [], prefix
        for _ in range(self.gamma):
            t = self.draft(cur)[0]
            out.append(t)
            cur += (t,)
        return out

    def _draft_tree(self, prefix: tuple):
        chain, cache, cur = [], {}, prefix
        for _ in range(self.gamma):
            cands = self.draft(cur)
            chain.append(cands[0])
            for alt in cands[1:self.branch_width]:
                cache[(cur, alt)] = self._greedy_continue(cur + (alt,))
            cur += (cands[0],)
        return chain, cache

    def generate(self, prompt: list, max_new: int) -> list:
        seq = tuple(prompt)
        n0 = len(seq)
        while len(seq) - n0 < max_new:
            chain, cache = self._draft_tree(seq)
            while chain and len(seq) - n0 < max_new:
                self.stats.cloud_rounds += 1
                self.stats.uplink_tokens += len(chain)
                self.stats.drafted += len(chain)
                cur, resumed = seq, None
                for tok in chain:
                    want = self.target(cur)[0]
                    if want == tok:
                        cur += (tok,)
                        self.stats.accepted += 1
                        continue
                    resumed = cache.get((cur, want))
                    self.stats.cache_hits += resumed is not None
                    self.stats.cache_misses += resumed is None
                    cur += (want,)
                    break
                else:
                    cur += (self.target(cur)[0],)  # bonus token from verifier
                seq = cur
                if resumed is None:
                    break
                chain = resumed  # O(1) recovery: continue from the cached branch
                _, cache = self._draft_tree(seq)
        return list(seq[n0:n0 + max_new])
