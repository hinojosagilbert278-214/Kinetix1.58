// CPU reference kernels for BitNet b1.58: packing, ternary GEMV, 2:4 sparse GEMV, XNOR/popcount.
// Packing matches kinetix/bitnet.py: 2 bits per weight (value+1), 4 weights per byte, LSB first.
#include <cstdint>
#include <cstddef>

extern "C" {

void bn_pack(const int8_t* w, size_t n, uint8_t* out) {
    for (size_t i = 0; i < (n + 3) / 4; ++i) {
        uint8_t b = 0;
        for (size_t j = 0; j < 4 && i * 4 + j < n; ++j)
            b |= (uint8_t)((w[i * 4 + j] + 1) & 3) << (2 * j);
        // padding slots decode to -1 by convention of the Python packer (value 0 -> -1 is never read)
        out[i] = b;
    }
}

void bn_unpack(const uint8_t* p, size_t n, int8_t* out) {
    for (size_t i = 0; i < n; ++i) out[i] = (int8_t)(((p[i / 4] >> (2 * (i % 4))) & 3) - 1);
}

// y[r] = sum_c W[r][c] * x[c], W ternary packed row-major (row stride = ceil(cols/4) bytes), integer add/sub only.
void bn_gemv_ternary(const uint8_t* packed, size_t rows, size_t cols, const int8_t* x, int32_t* y) {
    const size_t stride = (cols + 3) / 4;
    for (size_t r = 0; r < rows; ++r) {
        const uint8_t* row = packed + r * stride;
        int32_t acc = 0;
        for (size_t c = 0; c < cols; ++c) {
            int w = ((row[c >> 2] >> (2 * (c & 3))) & 3) - 1;
            if (w == 1) acc += x[c];
            else if (w == -1) acc -= x[c];
        }
        y[r] = acc;
    }
}

// 2:4 compression of one ternary row. One byte per group of 4:
// bits0-1 idx0, bits2-3 idx1, bit4 neg0, bit5 neg1, bit6 valid0, bit7 valid1.
// Returns 0 on success, -1 if a group has more than 2 non-zeros (violates 2:4).
int bn_sparse24_compress(const int8_t* w, size_t cols, uint8_t* out) {
    for (size_t g = 0; g < (cols + 3) / 4; ++g) {
        uint8_t b = 0;
        int k = 0;
        for (size_t j = 0; j < 4 && g * 4 + j < cols; ++j) {
            int8_t v = w[g * 4 + j];
            if (v == 0) continue;
            if (k == 2) return -1;
            b |= (uint8_t)j << (2 * k);
            if (v < 0) b |= 1 << (4 + k);
            b |= 1 << (6 + k);
            ++k;
        }
        out[g] = b;
    }
    return 0;
}

void bn_gemv_sparse24(const uint8_t* comp, size_t rows, size_t cols, const int8_t* x, int32_t* y) {
    const size_t groups = (cols + 3) / 4;
    for (size_t r = 0; r < rows; ++r) {
        int32_t acc = 0;
        for (size_t g = 0; g < groups; ++g) {
            uint8_t b = comp[r * groups + g];
            for (int k = 0; k < 2; ++k) {
                if (!((b >> (6 + k)) & 1)) continue;
                int32_t v = x[g * 4 + ((b >> (2 * k)) & 3)];
                acc += ((b >> (4 + k)) & 1) ? -v : v;
            }
        }
        y[r] = acc;
    }
}

// Dot product of two +/-1 vectors packed as bits (1 = +1) via XNOR + popcount.
int32_t bn_xnor_dot(const uint64_t* a, const uint64_t* w, size_t nbits) {
    size_t words = (nbits + 63) / 64;
    int32_t pop = 0;
    for (size_t i = 0; i < words; ++i) {
        uint64_t m = ~(a[i] ^ w[i]);
        size_t rem = nbits - i * 64;
        if (rem < 64) m &= (1ULL << rem) - 1;
        pop += __builtin_popcountll(m);
    }
    return 2 * pop - (int32_t)nbits;
}

}  // extern "C"
