# Deployment & Hardware Readiness

This repo is a prototype. This guide lists what runs today, what is unverified, and what is needed to take each part to production. Nothing here has been run on GPU hardware or with real model weights.

## 1. What works today (verified in dev container)

| Component | How to run | Verified |
|---|---|---|
| Python core (`kinetix/`) | `python -m pytest` | 18 tests pass |
| C++ kernels | `make lib` (needs g++) | parity vs Python via `tests/test_native.py` |
| Go gateway | `cd gateway && go test ./...` (Go >= 1.27.3) | unit + gRPC test |
| Envoy + ext_proc stack | `docker compose up --build` in `gateway/` | live routing checked with plain `docker run` (compose file itself not executed) |
| Benchmarks | `python bench/bench.py`, `python bench/onnx_latency.py` | CPU, synthetic model only |

## 2. Requirements for production

### 2.1 GPU / CUDA (kernels, local SLM)
- NVIDIA Ampere or newer (sm_80+) for 2:4 Sparse Tensor Cores; driver matching your CUDA toolkit.
- CUDA Toolkit with `nvcc` (>= 12.x). Build: `nvcc -O3 -arch=sm_80 -shared -Xcompiler -fPIC csrc/bitnet_kernels.cu -o libbitnet_cuda.so`.
- **Required before use:** run the CUDA kernels and compare bit-exactly with `bn_gemv_ternary` / `bn_gemv_sparse24`. They have only been compiled to PTX/SASS.
- Implement `mma.sp` per `docs/cuda_sparse_tensor_core.md`; measure before claiming speedups.
- Real ternary models: use a trained BitNet b1.58 checkpoint (e.g. via `bitnet.cpp`). The `bitnet.py` quantizer here is a reference, not a trainer; post-hoc quantizing a normal model degrades quality.

### 2.2 ONNX models (routing + retrieval)
- Export or download an INT8 ONNX classifier/bi-encoder (e.g. MiniLM or BGE) plus `tokenizer.json`.
- Python: `pip install onnxruntime tokenizers numpy`, then `MODEL_PATH=... TOKENIZER_PATH=... python bench/onnx_latency.py` and pass `OnnxEmbedder` to `DualStore(embedder=...)`.
- Gateway: replace the keyword classifier in `gateway/routing` with an ONNX Runtime call (e.g. `onnxruntime-go`, needs cgo and the ORT shared library in the image). Benchmark p99 end to end against the 5 ms budget; it is currently unmeasured.
- Train/calibrate the classifier on labeled public/private/hybrid queries and calibrate the uncertainty signal; `x-kinetix-uncertainty` is currently a client-supplied header, which must not be trusted from external callers (strip or overwrite it at the edge).

### 2.3 Serving (cloud + local)
- Replace the mock upstreams in `gateway/docker-compose.yml` with a vLLM (OpenAI-compatible) deployment for the cloud cluster and a local SLM server.
- AceSpec in `kinetix/acespec.py` is a token-list simulation. A real implementation needs a verifier API returning rejection position and target distribution, and a real KV cache on the edge; none exists yet.

### 2.4 Retrieval backends
- Parent-child logic is in-memory. Production needs Milvus (children), a KV store (parents), and optional Elasticsearch/Neo4j as in the blueprint; none are integrated.
- MinerU/Docling parsing and the web-scale index are out of scope for this repo.

## 3. Security & operations checklist
- Terminate TLS and authenticate at Envoy; the ext_proc gRPC port (50051) must not be exposed publicly.
- Sanitize/overwrite `x-kinetix-*` headers on ingress so clients cannot force routing.
- Private-route content must never be forwarded to external search; add tests/policy enforcing isolation before handling real documents.
- Set ext_proc `failure_mode_allow` and `message_timeout` deliberately; the gateway currently fails closed on timeout (504).
- Add metrics (route/engine counters, latency histograms) and health checks to the gateway.

## 4. Suggested order
1. Run `bench/` and the CUDA kernels on a GPU host; fix parity issues.
2. Obtain a real embedding/classifier model and measure latency.
3. Wire ONNX into the gateway; harden headers/auth.
4. Stand up vLLM + local SLM behind Envoy; replace mocks.
5. Integrate Milvus/KV stores, then build real speculative decoding.
