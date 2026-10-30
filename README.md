# Kinetix1.58
Hybrid AI Model

## Reference implementation

Stdlib-only Python modules in `kinetix/`: `router` (intent gatekeeper), `retrieval` (BM25+dense, parent-child dual store), `stepgate` (risk/uncertainty gate), `bitnet` (ternary BitLinear, 4-per-byte packing, N:M sparsity), `acespec` (edge-cloud speculative decoding simulation), `pipeline` (orchestration).

Run tests: `python -m pytest`

### Native kernels, gateway, embeddings

- `csrc/bitnet_kernels.cpp` (`make lib`, bound via `kinetix/native.py`): ternary packing, ternary GEMV, 2:4 sparse GEMV, XNOR/popcount dot. CPU only; no CUDA yet.
- `gateway/` (Go): Envoy `ext_proc` service that classifies requests and sets `x-kinetix-route` / `x-kinetix-engine`; `envoy.yaml` is a routing sketch. Run tests: `cd gateway && go test ./...`. The classifier is still keyword-based (no ONNX model yet).
- `kinetix/retrieval.py`: pluggable embedders. `HashEmbedder` is the default; `OnnxEmbedder` wraps an ONNX bi-encoder (needs `onnxruntime`, `tokenizers`, `numpy`; untested without a model).
