"""Hybrid BM25 + dense retrieval and the parent-child dual store (section 1)."""
import hashlib
import math
import re
from collections import Counter
from dataclasses import dataclass, field

DIM = 256


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def embed(text: str) -> list[float]:
    """Deterministic hashed bag-of-words embedding (stand-in for a real encoder)."""
    v = [0.0] * DIM
    for t in tokenize(text):
        h = int(hashlib.md5(t.encode()).hexdigest(), 16)
        v[h % DIM] += 1.0 if (h >> 64) & 1 else -1.0
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


class HashEmbedder:
    """Zero-dependency default embedder."""

    def encode(self, text: str) -> list[float]:
        return embed(text)


class OnnxEmbedder:
    """Bi-encoder (e.g. quantized BGE / MiniLM exported to ONNX). Needs onnxruntime, tokenizers, numpy.

    `model_path` is the .onnx file and `tokenizer_path` its tokenizer.json.
    """

    def __init__(self, model_path: str, tokenizer_path: str, max_len: int = 256):
        import numpy as np
        import onnxruntime as ort
        from tokenizers import Tokenizer

        self._np = np
        self.sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.tok = Tokenizer.from_file(tokenizer_path)
        self.tok.enable_truncation(max_len)
        self.tok.enable_padding()
        self.inputs = {i.name for i in self.sess.get_inputs()}

    def encode(self, text: str) -> list[float]:
        np = self._np
        enc = self.tok.encode(text)
        feed = {
            "input_ids": np.array([enc.ids], dtype=np.int64),
            "attention_mask": np.array([enc.attention_mask], dtype=np.int64),
            "token_type_ids": np.zeros((1, len(enc.ids)), dtype=np.int64),
        }
        hidden = self.sess.run(None, {k: v for k, v in feed.items() if k in self.inputs})[0][0]
        mask = np.array(enc.attention_mask, dtype=np.float32)[:, None]
        v = (hidden * mask).sum(0) / max(mask.sum(), 1.0)  # mean pooling
        v = v / (np.linalg.norm(v) or 1.0)
        return v.tolist()


def _cos(a, b):
    return sum(x * y for x, y in zip(a, b))


@dataclass
class Doc:
    id: str
    text: str
    meta: dict = field(default_factory=dict)


class HybridIndex:
    def __init__(self, k1=1.5, b=0.75, alpha=0.5, embedder=None):
        self.embedder = embedder or HashEmbedder()
        self.docs: list[Doc] = []
        self.tf: list[Counter] = []
        self.vecs: list[list[float]] = []
        self.df: Counter = Counter()
        self.k1, self.b, self.alpha = k1, b, alpha

    def add(self, doc: Doc):
        c = Counter(tokenize(doc.text))
        self.docs.append(doc)
        self.tf.append(c)
        self.vecs.append(self.embedder.encode(doc.text))
        self.df.update(c.keys())

    def _bm25(self, q: list[str]) -> list[float]:
        n = len(self.docs)
        avg = sum(sum(c.values()) for c in self.tf) / n if n else 1
        out = []
        for c in self.tf:
            dl = sum(c.values())
            s = 0.0
            for t in q:
                if t in c:
                    idf = math.log(1 + (n - self.df[t] + 0.5) / (self.df[t] + 0.5))
                    s += idf * c[t] * (self.k1 + 1) / (c[t] + self.k1 * (1 - self.b + self.b * dl / avg))
            out.append(s)
        return out

    def search(self, query: str, k=3) -> list[tuple[Doc, float]]:
        if not self.docs:
            return []
        bm = self._bm25(tokenize(query))
        mx = max(bm) or 1.0
        qv = self.embedder.encode(query)
        scored = [
            (d, self.alpha * (b / mx) + (1 - self.alpha) * max(_cos(qv, v), 0.0))
            for d, b, v in zip(self.docs, bm, self.vecs)
        ]
        scored.sort(key=lambda x: -x[1])
        return [s for s in scored[:k] if s[1] > 0]


def parent_child_chunks(markdown: str, source="doc"):
    """Split by heading into parents; lines inside each section become children."""
    parents, children = [], []
    sections = re.split(r"(?m)^(?=#{1,6} )", markdown)
    for i, sec in enumerate(s for s in sections if s.strip()):
        pid = f"{source}-p{i}"
        parents.append(Doc(pid, sec.strip(), {"source": source}))
        body = [ln.strip() for ln in sec.splitlines() if ln.strip()]
        heading = body[0] if body[0].startswith("#") else ""
        for j, ln in enumerate(body[1:] if heading else body):
            children.append(Doc(f"{pid}-c{j}", f"{heading} {ln}".strip(), {"parent_id": pid}))
    return parents, children


class DualStore:
    """Search small (child index), read big (parent KV lookup)."""

    def __init__(self, embedder=None):
        self.children = HybridIndex(embedder=embedder)
        self.parents: dict[str, Doc] = {}

    def ingest(self, markdown: str, source="doc"):
        parents, children = parent_child_chunks(markdown, source)
        for p in parents:
            self.parents[p.id] = p
        for c in children:
            self.children.add(c)

    def search(self, query: str, k=2) -> list[Doc]:
        seen, out = set(), []
        for child, _ in self.children.search(query, k * 3):
            pid = child.meta["parent_id"]
            if pid not in seen:
                seen.add(pid)
                out.append(self.parents[pid])
            if len(out) == k:
                break
        return out
