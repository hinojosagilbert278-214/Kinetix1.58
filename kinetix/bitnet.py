"""BitNet b1.58 ternary quantization, 4-per-byte packing, N:M sparsity, BitLinear (section 3)."""


def quantize_ternary(w: list[list[float]]):
    """Absmean quantization: W -> round(clip(W/gamma, -1, 1)) in {-1,0,+1}."""
    flat = [abs(x) for row in w for x in row]
    gamma = (sum(flat) / len(flat)) or 1e-8
    q = [[max(-1, min(1, round(x / gamma))) for x in row] for row in w]
    return q, gamma


def pack_ternary(row: list[int]) -> bytes:
    """Pack four ternary weights per byte (2 bits each, value+1)."""
    out = bytearray()
    for i in range(0, len(row), 4):
        b = 0
        for j, v in enumerate(row[i:i + 4]):
            b |= (v + 1) << (2 * j)
        out.append(b)
    return bytes(out)


def unpack_ternary(data: bytes, n: int) -> list[int]:
    vals = [((b >> (2 * j)) & 3) - 1 for b in data for j in range(4)]
    return vals[:n]


def apply_nm_sparsity(row: list[float], q: list[int], n=2, m=4) -> list[int]:
    """Keep at most n non-zeros per group of m, preferring the largest |w|."""
    out = list(q)
    for i in range(0, len(q), m):
        idx = [k for k in range(i, min(i + m, len(q))) if out[k] != 0]
        idx.sort(key=lambda k: -abs(row[k]))
        for k in idx[n:]:
            out[k] = 0
    return out


def quantize_activations_int8(x: list[float]):
    s = (max(abs(v) for v in x) or 1e-8) / 127
    return [round(v / s) for v in x], s


class BitLinear:
    """y = gamma * x_scale * (W_ternary @ x_int8): only integer add/sub in the inner loop."""

    def __init__(self, weights: list[list[float]], n=None, m=None):
        q, self.gamma = quantize_ternary(weights)
        if n and m:
            q = [apply_nm_sparsity(r, qr, n, m) for r, qr in zip(weights, q)]
        self.cols = len(weights[0])
        self.packed = [pack_ternary(r) for r in q]

    def __call__(self, x: list[float]) -> list[float]:
        xi, s = quantize_activations_int8(x)
        out = []
        for p in self.packed:
            acc = 0
            for w, a in zip(unpack_ternary(p, self.cols), xi):
                if w == 1:
                    acc += a
                elif w == -1:
                    acc -= a
            out.append(acc * s * self.gamma)
        return out

    def sparsity(self) -> float:
        z = sum(unpack_ternary(p, self.cols).count(0) for p in self.packed)
        return z / (self.cols * len(self.packed))
