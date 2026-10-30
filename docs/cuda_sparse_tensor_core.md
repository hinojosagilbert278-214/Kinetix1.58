# CUDA path for BitNet 2:4 sparsity

## Status
- `csrc/bitnet_kernels.cu`: CUDA-core ternary GEMV and 2:4 sparse GEMV (one warp per row), same packed formats as the CPU kernels.
- Verified only as far as: `clang --cuda-device-only --cuda-gpu-arch=sm_80` -> PTX -> `ptxas -arch=sm_80` succeeds. **Never executed**: no GPU or nvcc in the dev environment. Treat numerical parity with the CPU kernels as untested on device.
- Not Tensor Core code.

## Plan for Sparse Tensor Cores (Ampere+)
1. Instruction: `mma.sp.sync.aligned.m16n8k64.row.col.s32.s8.s8.s32` (int8, 2:4 on operand A). Ternary weights {-1,0,+1} are stored as int8 and promoted from the 2-bit packing in shared memory; activations are int8 (W1.58A8).
2. Compressed weights: per 4-element group keep 2 values; metadata is 2 bits per kept value (index within the group), packed into 32-bit registers in the fragment layout required by `mma.sp` (differs from this repo's byte-per-group CPU format; a repacking step is needed).
3. Pruning must make every group of 4 contain <=2 non-zeros (`apply_nm_sparsity`); groups violating this must be re-trained/fine-tuned, not silently truncated.
4. Tiling: load packed tiles to shared memory, unpack to int8 fragments, `ldmatrix`-style loads for B, accumulate int32, apply `gamma * x_scale` in the epilogue.
5. Validation: bit-exact comparison against `bn_gemv_sparse24` (integer math, so exact), then benchmark vs. cuBLASLt int8 and cuSPARSELt 2:4. Claims such as the 1.30x Sparse-BitNet speedup must be measured on target hardware.
6. Batch-1 decoding is memory-bound: expect gains from 2-bit packing bandwidth, not from Tensor Core math; `mma.sp` pays off for batched/prefill GEMM.
