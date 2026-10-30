# Kinetix1.58
Hybrid AI Model

## Reference implementation

Stdlib-only Python modules in `kinetix/`: `router` (intent gatekeeper), `retrieval` (BM25+dense, parent-child dual store), `stepgate` (risk/uncertainty gate), `bitnet` (ternary BitLinear, 4-per-byte packing, N:M sparsity), `acespec` (edge-cloud speculative decoding simulation), `pipeline` (orchestration).

Run tests: `python -m pytest`
