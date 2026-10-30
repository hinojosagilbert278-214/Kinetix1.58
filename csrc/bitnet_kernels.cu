// CUDA ports of the CPU kernels in bitnet_kernels.cpp (same packed formats). One warp per output row.
// CUDA-core implementations: they remove multiplies and cut memory traffic, but do NOT use Sparse Tensor
// Cores (mma.sp). See docs/cuda_sparse_tensor_core.md.
#include <cstdint>
#include <cuda_runtime.h>

__device__ __forceinline__ int32_t warp_sum(int32_t v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
    return v;
}

// packed: rows x ceil(cols/4) bytes, 2 bits/weight (value+1), LSB first. x: int8[cols]. y: int32[rows].
__global__ void bn_gemv_ternary(const uint8_t* __restrict__ packed, int rows, int cols,
                                const int8_t* __restrict__ x, int32_t* __restrict__ y) {
    int row = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    int lane = threadIdx.x & 31;
    if (row >= rows) return;
    const int stride = (cols + 3) / 4;
    const uint8_t* p = packed + (size_t)row * stride;
    int32_t acc = 0;
    for (int b = lane; b < stride; b += 32) {
        uint8_t byte = p[b];
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            int c = b * 4 + j;
            if (c >= cols) break;
            int w = ((byte >> (2 * j)) & 3) - 1;
            acc += w * (int)x[c];  // w in {-1,0,1}: compiler lowers to select/negate, no real multiply needed
        }
    }
    acc = warp_sum(acc);
    if (lane == 0) y[row] = acc;
}

// comp: rows x ceil(cols/4) bytes in the 2:4 format of bn_sparse24_compress.
__global__ void bn_gemv_sparse24(const uint8_t* __restrict__ comp, int rows, int cols,
                                 const int8_t* __restrict__ x, int32_t* __restrict__ y) {
    int row = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    int lane = threadIdx.x & 31;
    if (row >= rows) return;
    const int groups = (cols + 3) / 4;
    int32_t acc = 0;
    for (int g = lane; g < groups; g += 32) {
        uint8_t b = comp[(size_t)row * groups + g];
        #pragma unroll
        for (int k = 0; k < 2; ++k) {
            if (!((b >> (6 + k)) & 1)) continue;
            int32_t v = x[g * 4 + ((b >> (2 * k)) & 3)];
            acc += ((b >> (4 + k)) & 1) ? -v : v;
        }
    }
    acc = warp_sum(acc);
    if (lane == 0) y[row] = acc;
}

#ifndef KX_DEVICE_ONLY  // allows compile-checking kernels with clang --cuda-device-only
extern "C" cudaError_t bn_cuda_gemv_ternary(const uint8_t* d_packed, int rows, int cols, const int8_t* d_x,
                                            int32_t* d_y, cudaStream_t s) {
    const int warps = 8;
    bn_gemv_ternary<<<(rows + warps - 1) / warps, warps * 32, 0, s>>>(d_packed, rows, cols, d_x, d_y);
    return cudaGetLastError();
}

extern "C" cudaError_t bn_cuda_gemv_sparse24(const uint8_t* d_comp, int rows, int cols, const int8_t* d_x,
                                             int32_t* d_y, cudaStream_t s) {
    const int warps = 8;
    bn_gemv_sparse24<<<(rows + warps - 1) / warps, warps * 32, 0, s>>>(d_comp, rows, cols, d_x, d_y);
    return cudaGetLastError();
}
#endif
