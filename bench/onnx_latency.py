"""Router-classifier latency through ONNX Runtime.

With MODEL_PATH/TOKENIZER_PATH set (e.g. a quantized MiniLM/BGE export) it times OnnxEmbedder.encode.
Otherwise it builds a SYNTHETIC random-weight embedding-bag + linear classifier: this only measures the
ONNX Runtime call floor, NOT real tokenization or transformer inference cost.
Needs: pip install onnx onnxruntime numpy
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, ".")


def pct(xs, p):
    return sorted(xs)[int(len(xs) * p) - 1] * 1e3


def run(fn, n=500):
    for _ in range(20):
        fn()
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t)
    print(f"  p50={pct(ts, .5):.3f} ms  p99={pct(ts, .99):.3f} ms")


if os.environ.get("MODEL_PATH"):
    from kinetix.retrieval import OnnxEmbedder
    emb = OnnxEmbedder(os.environ["MODEL_PATH"], os.environ["TOKENIZER_PATH"])
    print("real model OnnxEmbedder.encode:")
    run(lambda: emb.encode("compare our internal spec to the latest rfc"))
else:
    import onnx
    import onnxruntime as ort
    from onnx import TensorProto as T, helper as h, numpy_helper as nh

    V, D, C = 30522, 128, 3
    rng = np.random.default_rng(0)
    g = h.make_graph(
        [h.make_node("Gather", ["E", "ids"], ["g"]), h.make_node("ReduceMean", ["g"], ["m"], axes=[1], keepdims=0),
         h.make_node("MatMul", ["m", "W"], ["l"]), h.make_node("Softmax", ["l"], ["p"], axis=-1)],
        "router", [h.make_tensor_value_info("ids", T.INT64, [1, None])],
        [h.make_tensor_value_info("p", T.FLOAT, [1, C])],
        [nh.from_array(rng.standard_normal((V, D)).astype(np.float32), "E"),
         nh.from_array(rng.standard_normal((D, C)).astype(np.float32), "W")])
    m = h.make_model(g, opset_imports=[h.make_opsetid("", 13)])
    m.ir_version = 8
    sess = ort.InferenceSession(m.SerializeToString(), providers=["CPUExecutionProvider"])
    ids = np.array([[hash(w) % V for w in "compare our internal spec to the latest rfc".split()]], dtype=np.int64)
    print("SYNTHETIC embedding-bag classifier (ORT call floor only):")
    run(lambda: sess.run(None, {"ids": ids}))
