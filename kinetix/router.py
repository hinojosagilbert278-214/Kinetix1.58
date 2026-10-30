"""Logical Intent Gatekeeper (section 2): keyword-weighted linear classifier."""
from dataclasses import dataclass
import re

PUBLIC, PRIVATE, HYBRID = "public", "private", "hybrid"

_PUBLIC = {"latest", "news", "today", "current", "recent", "price", "web", "standard", "rfc", "2025", "2026"}
_PRIVATE = {"uploaded", "document", "my", "our", "policy", "manual", "notes", "attached", "internal", "file", "spec"}


@dataclass
class Route:
    name: str
    public_score: float
    private_score: float


def classify(query: str) -> Route:
    toks = set(re.findall(r"[a-z0-9]+", query.lower()))
    pub = float(len(toks & _PUBLIC))
    priv = float(len(toks & _PRIVATE))
    if pub and priv:
        name = HYBRID
    elif priv:
        name = PRIVATE
    else:
        name = PUBLIC
    return Route(name, pub, priv)
