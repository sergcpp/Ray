// ASTC 4x4 fixed-layout encoder + spec-conformant scalar block decoder.
// Bitstream layout follows the Khronos ASTC spec (C.2.x); every ISE / unquant rule here was
// cross-checked against astcenc's tables - decode must stay bit-exact.
#include "TextureUtils.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <cstring>

namespace Ray {

// Block-emitter leaves with SIMD implementations in TextureUtilsASTC_SSE2.cpp /
// TextureUtilsASTC_NEON.cpp (byte-exact parity with the scalar reference leaves).
#if defined(_M_IX86) || defined(_M_X64) || defined(__i386__) || defined(__x86_64__)
#define RAY_ASTC_SIMD_SSE2 1
bool ASTCBlockIsFlat_SSE2(const uint8_t pixels[64]);
void ASTCGetMinMax_SSE2(const uint8_t pixels[64], uint8_t lo[4], uint8_t hi[4]);
void ASTCSelectWeights_SSE2(const uint8_t pixels[64], const uint8_t dec[4][8], int nch, uint8_t w[16], unsigned yshift);
#elif defined(__ARM_NEON__) || defined(__aarch64__) || defined(_M_ARM64)
#define RAY_ASTC_SIMD_NEON 1
bool ASTCBlockIsFlat_NEON(const uint8_t pixels[64]);
void ASTCGetMinMax_NEON(const uint8_t pixels[64], uint8_t lo[4], uint8_t hi[4]);
void ASTCSelectWeights_NEON(const uint8_t pixels[64], const uint8_t dec[4][8], int nch, uint8_t w[16], unsigned yshift);
#endif

namespace {
// Color endpoint modes: the 4-bit CEM field at bits [16:13] (astcenc endpoint_formats numbering).
enum eASTC_CEM : uint8_t {
    CEM_LUMINANCE = 0,       // LDR luminance, direct         (2 endpoint values)
    CEM_LUMINANCE_ALPHA = 4, // LDR luminance+alpha, direct   (4 endpoint values)
    CEM_RGBA = 12,           // LDR RGBA, direct              (8 endpoint values)
};

// Quantization methods (astcenc numbering; level counts must not be reordered).
enum eASTC_Quant : uint8_t {
    Q_2 = 0,
    Q_3,
    Q_4,
    Q_5,
    Q_6,
    Q_8,
    Q_10,
    Q_12,
    Q_16,
    Q_20,
    Q_24,
    Q_32,
    Q_40,
    Q_48,
    Q_64,
    Q_80,
    Q_96,
    Q_128,
    Q_160,
    Q_192,
    Q_256,
    Q_Count
};

constexpr int g_quant_levels[Q_Count] = {2,  3,  4,  5,  6,  8,  10,  12,  16,  20, 24,
                                         32, 40, 48, 64, 80, 96, 128, 160, 192, 256};
constexpr uint8_t g_quant_bits[Q_Count] = {1, 0, 2, 0, 1, 3, 1, 2, 4, 2, 3, 5, 3, 4, 6, 4, 5, 7, 5, 6, 8};
constexpr uint8_t g_quant_trits[Q_Count] = {0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0};
constexpr uint8_t g_quant_quints[Q_Count] = {0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0};

// ISE bitcount for 'count' values = (scale*count + round) / divisor (astcenc ise_sizes).
constexpr uint8_t g_ise_scale[Q_Count] = {1, 8, 2, 7, 13, 3, 10, 18, 4, 13, 23, 5, 16, 28, 6, 19, 33, 7, 22, 38, 8};
constexpr uint8_t g_ise_round[Q_Count] = {0, 4, 0, 2, 4, 0, 2, 4, 0, 2, 4, 0, 2, 4, 0, 2, 4, 0, 2, 4, 0};
constexpr uint8_t g_ise_div[Q_Count] = {1, 5, 1, 3, 5, 1, 3, 5, 1, 3, 5, 1, 3, 5, 1, 3, 5, 1, 3, 5, 1};

int ise_bitcount(const int count, const int method) {
    return (g_ise_scale[method] * count + g_ise_round[method]) / g_ise_div[method];
}

// Per-method (B addend per m-bit, constant C) for the T = D*C + B; T ^= A; T = (A&M)|(T>>2)
// chain; m bits are a=bit0..f=bit5, D = the trit/quint digit. Colors: 9-bit A/B/C, mask 0x80.
// B coefficients verified entry-by-entry against astcenc color_unquant_tables rows.
struct quant_unq_params {
    int16_t Bc[6];
    uint16_t C;
};
constexpr quant_unq_params g_color_uq[Q_Count] = {
    {{0, 0, 0, 0, 0, 0}, 0},        // Q_2 (bit-only)
    {{0, 0, 0, 0, 0, 0}, 0},        // Q_3 (special)
    {{0, 0, 0, 0, 0, 0}, 0},        // Q_4 (bit-only)
    {{0, 0, 0, 0, 0, 0}, 0},        // Q_5 (special)
    {{0, 0, 0, 0, 0, 0}, 204},      // Q_6   0..5:  B=0
    {{0, 0, 0, 0, 0, 0}, 0},        // Q_8 (bit-only)
    {{0, 0, 0, 0, 0, 0}, 113},      // Q_10  0..9:  B=0
    {{0, 278, 0, 0, 0, 0}, 93},     // Q_12  0..11: B=b*278
    {{0, 0, 0, 0, 0, 0}, 0},        // Q_16 (bit-only)
    {{0, 268, 0, 0, 0, 0}, 54},     // Q_20  0..19: B=b*268
    {{0, 133, 266, 0, 0, 0}, 44},   // Q_24  0..23: B=b*133 + c*266
    {{0, 0, 0, 0, 0, 0}, 0},        // Q_32 (bit-only)
    {{0, 130, 261, 0, 0, 0}, 26},   // Q_40  0..39: B=b*130 + c*261
    {{0, 64, 130, 260, 0, 0}, 22},  // Q_48  0..47 (Bc fit vs astcenc tables)
    {{0, 0, 0, 0, 0, 0}, 0},        // Q_64 (bit-only)
    {{0, 64, 129, 258, 0, 0}, 13},  // Q_80  0..79
    {{0, 32, 64, 129, 258, 0}, 11}, // Q_96 0..95
    {{0, 0, 0, 0, 0, 0}, 0},        // Q_128 (bit-only)
    {{0, 32, 64, 128, 256, 0}, 6},  // Q_160 0..159
    {{0, 16, 32, 64, 128, 256}, 4}, // Q_192 0..191
    {{0, 0, 0, 0, 0, 0}, 0},        // Q_256 (bit-only)
};
// Weight unquant rows (0..64 scale), astcenc quant_and_xfer_tables[] unquant fields verbatim.
// Modes above Q_32 can never fit the 96-bit weight budget (C.2.22), so no rows exist for them.
constexpr int g_weight_row_off[13] = {0, 2, 5, 9, 14, 20, 28, 38, 50, 66, 86, 110, 142}; // Q_2..Q_32
constexpr uint8_t g_weight_uq_rows[142] = {
    0,  64, 0,  32, 64, 0,  21, 43, 64, 0,  16, 32, 48, 64, 0,  64, 12, 52, 25, 39, 0,  9,  18, 27, 37, 46, 55, 64, 0,
    64, 7,  57, 14, 50, 21, 43, 28, 36, 0,  64, 17, 47, 5,  59, 23, 41, 11, 53, 28, 36, 0,  4,  8,  12, 17, 21, 25, 29,
    35, 39, 43, 47, 52, 56, 60, 64, 0,  64, 16, 48, 3,  61, 19, 45, 6,  58, 23, 41, 9,  55, 26, 38, 13, 51, 29, 35, 0,
    64, 8,  56, 16, 48, 24, 40, 2,  62, 11, 53, 19, 45, 27, 37, 5,  59, 13, 51, 22, 42, 30, 34, 0,  2,  4,  6,  8,  10,
    12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64};

bool weight_range_supported(const int q) {
    // Rows above (Q_32) would exceed the 96-bit weight budget in a valid block mode anyway;
    // decode_block_mode_2d's bit gate rejects them, this filter keeps them out early.
    return q <= Q_32;
}

uint8_t replicate_bits(const uint8_t v, const int n, const int out_bits) {
    int res = 0;
    for (int i = 0; i < out_bits; i++)
        res = (res << 1) | ((v >> (n - 1 - (i % n))) & 1);
    return uint8_t(res);
}

// Generate the color endpoint unquant table (0..255).
void build_color_unquant_table(uint8_t tab[Q_Count][256]) {
    constexpr int maskA = 0x80;
    constexpr int max_val = 255;
    for (int q = 0; q < Q_Count; q++) {
        const int bits = g_quant_bits[q], levels = g_quant_levels[q];
        const bool bq = (g_quant_trits[q] | g_quant_quints[q]) != 0;
        for (int v = 0; v < levels; v++) {
            int t;
            if (bq && bits == 0) {
                // trit/quint with no extra bits: spec Table C.2.13 (round UP steps).
                t = (max_val * v + levels - 2) / (levels - 1);
            } else if (!bq) { // bit-only: replicate MSB pattern
                t = replicate_bits(uint8_t(v), bits, 8);
            } else { // bit + trit/quint digit
                const int digit = v >> bits, m = v & ((1 << bits) - 1);
                const int a = m & 1;
                int A = 0, B = 0;
                for (int i = 0; i < 9; i++)
                    A = (A << 1) | a;
                for (int k = 0; k < 6; k++)
                    if ((m >> k) & 1)
                        B += g_color_uq[q].Bc[k];
                int T = digit * g_color_uq[q].C + B;
                T ^= A;
                T = (A & maskA) | (T >> 2);
                t = T;
            }
            tab[q][v] = uint8_t(t);
        }
    }
}

// Nearest unquant code for an 8-bit target (full scan: the spec bit-math scrambles table order).
int nearest_color_code(const int q, const uint8_t target, const uint8_t *tab) {
    const int levels = g_quant_levels[q];
    int best = 0, best_err = 256;
    for (int v = 0; v < levels; v++) {
        const int err = std::abs(int(tab[v]) - int(target));
        if (err < best_err) {
            best_err = err;
            best = v;
        }
    }
    return best;
}

struct ASTC_tables_t {
    uint8_t color[Q_Count][256];
    uint8_t enc[Q_Count][256];    // inverse of color[]: O(1) endpoint encode (same first-min tie as a scan)
    uint8_t weight[Q_Count][256]; // 0..64 grid scale, Q_2..Q_32 only

    ASTC_tables_t() {
        build_color_unquant_table(color);
        for (int q = 0; q < Q_Count; q++)
            for (int t = 0; t < 256; t++)
                enc[q][t] = uint8_t(nearest_color_code(q, uint8_t(t), color[q]));
        for (int q = 0; q <= Q_32; q++)
            memcpy(weight[q], &g_weight_uq_rows[g_weight_row_off[q]], size_t(g_quant_levels[q]));
    }
};

const ASTC_tables_t &tables() {
    static const ASTC_tables_t t;
    return t;
}

void write_bits(unsigned value, const unsigned bitcount, const unsigned bitoffset, uint8_t *data) {
    const unsigned mask = (1u << bitcount) - 1;
    value &= mask;
    uint8_t *ptr = data + (bitoffset >> 3);
    const unsigned off = bitoffset & 7;
    *ptr = uint8_t((*ptr & ~(mask << off)) | (value << off));
    if (off + bitcount > 8) {
        const unsigned rem = 8 - off;
        ptr[1] = uint8_t((ptr[1] & ~(mask >> rem)) | (value >> rem));
    }
}

[[maybe_unused]] unsigned read_bits(const unsigned bitcount, const unsigned bitoffset, const uint8_t *data) {
    const uint8_t *ptr = data + (bitoffset >> 3);
    const unsigned off = bitoffset & 7;
    unsigned res = unsigned(ptr[0]) >> off;
    if (off + bitcount > 8)
        res |= unsigned(ptr[1]) << (8 - off);
    return res & ((1u << bitcount) - 1);
}

struct BitRev8 {
    uint8_t t[256];
    constexpr BitRev8() : t() {
        for (int v = 0; v < 256; v++) {
            int r = 0;
            for (int i = 0; i < 8; i++)
                r = (r << 1) | ((v >> i) & 1);
            t[v] = uint8_t(r);
        }
    }
};
constexpr BitRev8 g_bitrev8{};

// Trit/quint group packing is NOT plain base-3/base-5 (spec C.2.12 scrambles the group values);
// the encode tables below are the exact inverses of the spec bit-manipulation decoders and were
// generated from astcenc's trits_of_integer[]/quints_of_integer[] reference tables.
constexpr uint8_t g_trit_group_of[243] = { // 5 trits (base-3 index) -> 8-bit group value T
    0,   1,   2,   4,   5,   6,   8,   9,   10,  16,  17,  18,  20,  21,  22,  24,  25,  26,  3,   7,   11,  19,  23,
    27,  12,  13,  14,  32,  33,  34,  36,  37,  38,  40,  41,  42,  48,  49,  50,  52,  53,  54,  56,  57,  58,  35,
    39,  43,  51,  55,  59,  44,  45,  46,  64,  65,  66,  68,  69,  70,  72,  73,  74,  80,  81,  82,  84,  85,  86,
    88,  89,  90,  67,  71,  75,  83,  87,  91,  76,  77,  78,  128, 129, 130, 132, 133, 134, 136, 137, 138, 144, 145,
    146, 148, 149, 150, 152, 153, 154, 131, 135, 139, 147, 151, 155, 140, 141, 142, 160, 161, 162, 164, 165, 166, 168,
    169, 170, 176, 177, 178, 180, 181, 182, 184, 185, 186, 163, 167, 171, 179, 183, 187, 172, 173, 174, 192, 193, 194,
    196, 197, 198, 200, 201, 202, 208, 209, 210, 212, 213, 214, 216, 217, 218, 195, 199, 203, 211, 215, 219, 204, 205,
    206, 96,  97,  98,  100, 101, 102, 104, 105, 106, 112, 113, 114, 116, 117, 118, 120, 121, 122, 99,  103, 107, 115,
    119, 123, 108, 109, 110, 224, 225, 226, 228, 229, 230, 232, 233, 234, 240, 241, 242, 244, 245, 246, 248, 249, 250,
    227, 231, 235, 243, 247, 251, 236, 237, 238, 28,  29,  30,  60,  61,  62,  92,  93,  94,  156, 157, 158, 188, 189,
    190, 220, 221, 222, 31,  63,  95,  159, 191, 223, 124, 125, 126};
constexpr uint8_t g_quint_group_of[125] = { // 3 quints (base-5 index) -> 7-bit group value Q
    0,   1,   2,   3,   4,   8,   9,   10,  11,  12,  16,  17,  18,  19,  20,  24, 25,  26,  27,  28,  5,
    13,  21,  29,  6,   32,  33,  34,  35,  36,  40,  41,  42,  43,  44,  48,  49, 50,  51,  52,  56,  57,
    58,  59,  60,  37,  45,  53,  61,  14,  64,  65,  66,  67,  68,  72,  73,  74, 75,  76,  80,  81,  82,
    83,  84,  88,  89,  90,  91,  92,  69,  77,  85,  93,  22,  96,  97,  98,  99, 100, 104, 105, 106, 107,
    108, 112, 113, 114, 115, 116, 120, 121, 122, 123, 124, 101, 109, 117, 125, 30, 102, 103, 70,  71,  38,
    110, 111, 78,  79,  46,  118, 119, 86,  87,  54,  126, 127, 94,  95,  62,  39, 47,  55,  63,  7};

// ISE encode of 'count' symbolic values at bit_offset (port of astcenc encode_ise()).
void encode_ise(const int method, const int count, const uint8_t *values, uint8_t *out, unsigned bit_offset) {
    const int bits = g_quant_bits[method], mask = (1 << bits) - 1;
    int i = 0;
    if (g_quant_trits[method]) {
        static const uint8_t tbits[5] = {2, 2, 1, 2, 1};
        static const uint8_t tshift[5] = {0, 2, 4, 5, 7};
        while (i < count) {
            const int group = i + 5 <= count ? 5 : count - i;
            int idx = 0, mul = 1; // base-3 tuple index (missing lanes pad with trit 0)
            for (int j = 0; j < group; j++, mul *= 3)
                idx += int(values[i + j] >> bits) * mul;
            const int T = g_trit_group_of[idx];
            for (int j = 0; j < group; j++, i++) {
                const unsigned pack =
                    unsigned(values[i] & mask) | (unsigned((T >> tshift[j]) & ((1 << tbits[j]) - 1)) << bits);
                write_bits(pack, unsigned(bits + tbits[j]), bit_offset, out);
                bit_offset += unsigned(bits) + tbits[j];
            }
        }
    } else if (g_quant_quints[method]) {
        static const uint8_t qbits[3] = {3, 2, 2};
        static const uint8_t qshift[3] = {0, 3, 5};
        while (i < count) {
            const int group = i + 3 <= count ? 3 : count - i;
            int idx = 0, mul = 1; // base-5 tuple index (missing lanes pad with quint 0)
            for (int j = 0; j < group; j++, mul *= 5)
                idx += int(values[i + j] >> bits) * mul;
            const int Q = g_quint_group_of[idx];
            for (int j = 0; j < group; j++, i++) {
                const unsigned pack =
                    unsigned(values[i] & mask) | (unsigned((Q >> qshift[j]) & ((1 << qbits[j]) - 1)) << bits);
                write_bits(pack, unsigned(bits + qbits[j]), bit_offset, out);
                bit_offset += unsigned(bits) + qbits[j];
            }
        }
    } else {
        for (; i < count; i++) {
            write_bits(values[i], unsigned(bits), bit_offset, out);
            bit_offset += unsigned(bits);
        }
    }
}

// Spec C.2.8 / astcenc decode_block_mode_2d: unpack the 11-bit block mode.
bool decode_block_mode_2d(const unsigned block_mode, int &x_weights, int &y_weights, bool &dual_plane,
                          int &weight_range) {
    unsigned base_q = (block_mode >> 4) & 1;
    unsigned H = (block_mode >> 9) & 1;
    unsigned D = (block_mode >> 10) & 1;
    const unsigned A = (block_mode >> 5) & 3;

    x_weights = y_weights = 0;

    if ((block_mode & 3) != 0) {
        base_q |= (block_mode & 3) << 1;
        unsigned B = (block_mode >> 7) & 3;
        switch ((block_mode >> 2) & 3) {
        case 0:
            x_weights = int(B + 4);
            y_weights = int(A + 2);
            break;
        case 1:
            x_weights = int(B + 8);
            y_weights = int(A + 2);
            break;
        case 2:
            x_weights = int(A + 2);
            y_weights = int(B + 8);
            break;
        case 3:
            B &= 1;
            if (block_mode & 0x100) {
                x_weights = int(B + 2);
                y_weights = int(A + 2);
            } else {
                x_weights = int(A + 2);
                y_weights = int(B + 6);
            }
            break;
        }
    } else {
        base_q |= ((block_mode >> 2) & 3) << 1;
        if (((block_mode >> 2) & 3) == 0)
            return false;
        unsigned B = (block_mode >> 9) & 3;
        switch ((block_mode >> 7) & 3) {
        case 0:
            x_weights = 12;
            y_weights = int(A + 2);
            break;
        case 1:
            x_weights = int(A + 2);
            y_weights = 12;
            break;
        case 2:
            x_weights = int(A + 6);
            y_weights = int(B + 6);
            D = 0;
            H = 0;
            break;
        case 3:
            switch ((block_mode >> 5) & 3) {
            case 0:
                x_weights = 6;
                y_weights = 10;
                break;
            case 1:
                x_weights = 10;
                y_weights = 6;
                break;
            default:
                return false;
            }
            break;
        }
    }

    const int weight_count = x_weights * y_weights * int(D + 1);
    weight_range = int(base_q - 2) + 6 * int(H);
    dual_plane = D != 0;

    if (weight_count > 64 || weight_range < 0 || weight_range >= Q_Count || !weight_range_supported(weight_range))
        return false;
    const int bits = ise_bitcount(weight_count, weight_range);
    return bits >= 24 && bits <= 96;
}

// Endpoint quant = richest method whose ISE stream fits the bits left by the header+weights.
int pick_endpoint_quant(const int num_vals, const int weight_bits) {
    const int color_bits = 128 - 17 - weight_bits;
    for (int q = Q_Count - 1; q >= 0; q--)
        if (ise_bitcount(num_vals, q) <= color_bits)
            return q;
    return -1;
}

struct ASTC4x4_mode_cand_t {
    uint16_t block_mode;
    uint8_t xw, yw;
    uint8_t wq; // weight quant method
    uint8_t eq; // endpoint quant method
};

// Fixed layout for the fast encoder (BC3-class): 4x4 grid (one weight per texel), 8 weight
// levels (3 bits). eq is re-derived by the decoder the same way, so it needs no storage.
const ASTC4x4_mode_cand_t &fixed_mode(const int num_endpoint_vals) {
    struct cache_t {
        ASTC4x4_mode_cand_t by_vals[3]; // slots: 2, 4, 8 endpoint values
        cache_t() {
            for (int slot = 0; slot < 3; slot++) {
                bool found = false;
                for (unsigned bm = 0; bm < 2048 && !found; bm++) {
                    int xw, yw, wq;
                    bool dual;
                    if (!decode_block_mode_2d(bm, xw, yw, dual, wq) || dual)
                        continue;
                    if (xw != 4 || yw != 4 || wq != Q_8)
                        continue;
                    by_vals[slot] = {uint16_t(bm), uint8_t(xw), uint8_t(yw), uint8_t(wq),
                                     uint8_t(pick_endpoint_quant(2 << slot, ise_bitcount(16, Q_8)))};
                    found = true;
                }
                assert(found);
            }
        }
    };
    static const cache_t cache;
    return cache.by_vals[num_endpoint_vals == 2 ? 0 : (num_endpoint_vals == 4 ? 1 : 2)];
}

// Extracts a 4x4 block into RGBA8 slots (mirrors the Ref extractors in TextureUtils.cpp,
// which are TU-local templates). Channels beyond SrcChannels keep the caller's seed.
template <int SrcChannels> void extract4x4_rgba(const uint8_t src[], const int stride, uint8_t dst[64]) {
    if constexpr (SrcChannels == 4) {
        for (int j = 0; j < 4; j++) {
            memcpy(dst, src, 16);
            dst += 16;
            src += stride;
        }
    } else {
        for (int j = 0; j < 4; j++) {
            for (int i = 0; i < 4; i++)
                for (int c = 0; c < SrcChannels; c++)
                    dst[i * 4 + c] = src[i * SrcChannels + c];
            dst += 16;
            src += stride;
        }
    }
}

template <int SrcChannels>
void extract4x4_rgba_partial(const uint8_t src[], const int stride, const int bw, const int bh, uint8_t dst[64]) {
    for (int j = 0; j < bh; j++) {
        for (int i = 0; i < bw; i++)
            for (int c = 0; c < SrcChannels; c++)
                dst[i * 4 + c] = src[i * SrcChannels + c];
        for (int i = bw; i < 4; i++)
            for (int c = 0; c < SrcChannels; c++)
                dst[i * 4 + c] = dst[(bw - 1) * 4 + c];
        dst += 16;
        src += stride;
    }
    uint8_t *dup = dst - 16;
    for (int j = bh; j < 4; j++) {
        memcpy(dst, dup, 16);
        dst += 16;
    }
}

// Scalar reference for the per-block hot leaves (SIMD twins live in the _SSE2/_NEON TUs).
// SelectWeights: a 4x4 grid makes infill identity, so every texel independently takes the
// nearest weight level (SSE over the channels carrying signal; first level wins ties).
struct RefLeaves {
    static bool IsFlat(const uint8_t pixels[64]) {
        for (int i = 1; i < 16; i++)
            if (memcmp(&pixels[i * 4], pixels, 4) != 0)
                return false;
        return true;
    }
    static void GetMinMax(const uint8_t pixels[64], uint8_t lo[4], uint8_t hi[4]) {
        for (int c = 0; c < 4; c++) {
            lo[c] = 255;
            hi[c] = 0;
        }
        for (int i = 0; i < 16; i++)
            for (int c = 0; c < 4; c++) {
                lo[c] = std::min(lo[c], pixels[i * 4 + c]);
                hi[c] = std::max(hi[c], pixels[i * 4 + c]);
            }
    }
    static void SelectWeights(const uint8_t pixels[64], const uint8_t dec[4][8], const int nch, uint8_t w[16],
                              const unsigned yshift) {
        for (int t = 0; t < 16; t++) {
            int best_v = 0;
            unsigned best_e = ~0u;
            for (int v = 0; v < 8; v++) {
                unsigned te = 0;
                for (int ch = 0; ch < nch; ch++) {
                    const int d = int(dec[ch][v]) - int(pixels[t * 4 + ch]);
                    te += (ch == 3 && yshift) ? (unsigned(d * d) << yshift) : unsigned(d * d);
                }
                if (te < best_e) {
                    best_e = te;
                    best_v = v;
                }
            }
            w[t] = uint8_t(best_v);
        }
    }
};

#ifdef RAY_ASTC_SIMD_SSE2
struct SimdLeaves {
    static bool IsFlat(const uint8_t pixels[64]) { return ASTCBlockIsFlat_SSE2(pixels); }
    static void GetMinMax(const uint8_t pixels[64], uint8_t lo[4], uint8_t hi[4]) {
        ASTCGetMinMax_SSE2(pixels, lo, hi);
    }
    static void SelectWeights(const uint8_t pixels[64], const uint8_t dec[4][8], const int nch, uint8_t w[16],
                              const unsigned yshift) {
        ASTCSelectWeights_SSE2(pixels, dec, nch, w, yshift);
    }
};
#elif defined(RAY_ASTC_SIMD_NEON)
struct SimdLeaves {
    static bool IsFlat(const uint8_t pixels[64]) { return ASTCBlockIsFlat_NEON(pixels); }
    static void GetMinMax(const uint8_t pixels[64], uint8_t lo[4], uint8_t hi[4]) {
        ASTCGetMinMax_NEON(pixels, lo, hi);
    }
    static void SelectWeights(const uint8_t pixels[64], const uint8_t dec[4][8], const int nch, uint8_t w[16],
                              const unsigned yshift) {
        ASTCSelectWeights_NEON(pixels, dec, nch, w, yshift);
    }
};
#else
using SimdLeaves = RefLeaves; // no SIMD form for this target
#endif

// Encode one 4x4 block of RGBA8 texels, writing exactly 16 bytes. Fixed BC3-class layout:
// one weight per texel (4x4 grid, 8 levels) + min/max endpoints. CEM is RGBA (8 endpoint
// values, alpha from the data) or luminance (2 values from R; decode forces alpha 255).
// cocg = input is YCoCg-packed (Co,Cg,0,Y): the constant B channel doubles as the decoder's
// chroma-scale exponent (same trick and same byte encoding as the BC path's ScaleYCoCg).
template <class LV>
void emit_astc_block_impl(const uint8_t pixels[64], const int cem, const int nch, const bool cocg, uint8_t *&out_data) {
    const int n_e = cem == CEM_RGBA ? 8 : 2;

    // Void-extent fast path: all 16 texels identical.
    if (cem == CEM_RGBA && LV::IsFlat(pixels)) {
        uint8_t blk[16] = {0xFC, 0xFD, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF};
        for (int ch = 0; ch < 4; ch++) {
            blk[8 + 2 * ch] = pixels[ch];
            blk[9 + 2 * ch] = pixels[ch];
        }
        memcpy(out_data, blk, 16);
        out_data += 16;
        return;
    }

    // Endpoint targets: per-channel 8-bit min/max over the block.
    uint8_t lo[4], hi[4];
    LV::GetMinMax(pixels, lo, hi);

    const ASTC4x4_mode_cand_t &cand = fixed_mode(n_e);
    // nch = channels carrying signal for weight selection (RG = 2: the constant B/A seed channels
    // add a level-independent SSE term, so dropping them cannot reorder argmin/ties).
    // n_c = endpoint-coded channels; B/A pairs encode from lo==hi, their swap is a no-op.
    const int n_c = cem == CEM_RGBA ? 4 : 1;
    assert(nch >= 1 && nch <= n_c);

    // Chroma amplification (YCoCg blocks): (Co,Cg-128)*s+128 with s in {1,2,3}, B channel
    // carrying (s-1)*8 for YCoCg_to_RGB's divide-out (same as the BC path's ScaleYCoCg).
    // Endpoints then ride an s-times finer effective chroma grid; the weight pick stays in the
    // unscaled final domain (picking on amplified levels overweights chroma by s^2: measured
    // -2.3 dB on Fabric), which is pick-identical to the un-scaled path (monotonic mapping).
    int cocg_s = 1;
    uint8_t lo_u[4], hi_u[4];
    memcpy(lo_u, lo, 4);
    memcpy(hi_u, hi, 4);
    if (cocg && cem == CEM_RGBA && lo[2] == hi[2]) {
        int m = 0;
        for (int ch2 = 0; ch2 < 2; ch2++) {
            m = std::max(m, std::abs(int(lo[ch2]) - 128));
            m = std::max(m, std::abs(int(hi[ch2]) - 128));
        }
        cocg_s = m <= 31 ? 3 : (m <= 63 ? 2 : 1);
        if (cocg_s > 1) {
            for (int ch = 0; ch < 2; ch++) {
                lo[ch] = uint8_t((int(lo_u[ch]) - 128) * cocg_s + 128);
                hi[ch] = uint8_t((int(hi_u[ch]) - 128) * cocg_s + 128);
            }
            lo[2] = hi[2] = uint8_t((cocg_s - 1) * 8);
        }
    }

    // Endpoints from per-channel min/max; order the pairs so the decoder's blue-uncontract
    // + swap branch stays inert (same guard as its uncontraction).
    bool swapped = false;
    const uint8_t *enc_tab = tables().enc[cand.eq];
    uint8_t code[8] = {};
    for (int ch = 0; ch < n_c; ch++) {
        code[2 * ch] = enc_tab[lo[ch]];
        code[2 * ch + 1] = enc_tab[hi[ch]];
    }
    {
        auto unq = [&](int idx) { return int(tables().color[cand.eq][code[idx]]); };
        const int sum0 = cem == CEM_RGBA ? unq(0) + unq(2) + unq(4) : unq(0);
        const int sum1 = cem == CEM_RGBA ? unq(1) + unq(3) + unq(5) : unq(1);
        swapped = sum0 > sum1;
        if (swapped)
            for (int ch = 0; ch < n_c; ch++)
                std::swap(code[2 * ch], code[2 * ch + 1]);
    }

    int e0[4], e1[4];
    for (int ch = 0; ch < 4; ch++) {
        if (cem == CEM_RGBA) {
            e0[ch] = int(tables().color[cand.eq][code[2 * ch]]);
            e1[ch] = int(tables().color[cand.eq][code[2 * ch + 1]]);
        } else if (ch < 3) {
            e0[ch] = int(tables().color[cand.eq][code[0]]);
            e1[ch] = int(tables().color[cand.eq][code[1]]);
        } else {
            e0[ch] = e1[ch] = 255;
        }
    }

    // Exact 8-bit decoder color per weight level (16-bit lerp + UNORM8 take), small table.
    const uint8_t *wtab = tables().weight[cand.wq];
    assert(cand.wq == Q_8);
    uint8_t dec[4][8] = {};
    for (int ch = 0; ch < nch; ch++) {
        const int E0 = (e0[ch] << 8) | e0[ch], E1 = (e1[ch] << 8) | e1[ch];
        for (int v = 0; v < 8; v++)
            dec[ch][v] = uint8_t((E0 * (64 - int(wtab[v])) + E1 * int(wtab[v]) + 32) >> 14);
    }

    // BC-style index pick (leaf: scalar or SIMD). For amplified blocks pick against the
    // unscaled level grid (final-domain luma/chroma balance); the same weights decode on the
    // amplified grid, with the pick's endpoint order following the emitted pair order.
    uint8_t w[16];
    // YCoCg: weight Y x4 in the pick - scene noise metrics track luminance (specular), and
    // the shared plane otherwise lets chroma outvote luma (measured: BC3 beats ASTC's Y by
    // 0.3-6.5 dB per texture while ASTC wins chroma by 8-21 dB).
    const unsigned yshift = cocg ? 3u : 0u;
    if (cocg_s > 1) {
        uint8_t udec[4][8] = {};
        for (int ch = 0; ch < nch; ch++) {
            const int E0 = int(tables().color[cand.eq][enc_tab[swapped ? hi_u[ch] : lo_u[ch]]]);
            const int E1 = int(tables().color[cand.eq][enc_tab[swapped ? lo_u[ch] : hi_u[ch]]]);
            for (int v = 0; v < 8; v++)
                udec[ch][v] =
                    uint8_t(((((E0 << 8) | E0) * (64 - int(wtab[v])) + ((E1 << 8) | E1) * int(wtab[v]) + 32) >> 14));
        }
        LV::SelectWeights(pixels, udec, nch, w, yshift);
    } else {
        LV::SelectWeights(pixels, dec, nch, w, yshift);
    }

    // Pack (layout mirrors astcenc symbolic_to_physical): weights are ISE-encoded, then
    // stored bit-reversed within bytes and byte-reversed at the block top. Q_8 weights are
    // bit-only ISE: the whole 48-bit stream packs into one 64-bit register.
    uint64_t wbits = 0;
    for (int i = 0; i < 16; i++)
        wbits |= uint64_t(w[i]) << (3 * i);
    uint8_t blk[16] = {};
    for (int k = 0; k < 6; k++)
        blk[10 + k] = g_bitrev8.t[uint8_t(wbits >> (8 * (5 - k)))];

    write_bits(cand.block_mode, 11, 0, blk);
    write_bits(0, 2, 11, blk);             // partitions-1 = 0
    write_bits(unsigned(cem), 4, 13, blk); // CEM
    encode_ise(cand.eq, n_e, code, blk, 17);

    memcpy(out_data, blk, 16);
    out_data += 16;
}

void emit_astc_block(const uint8_t pixels[64], const int cem, const int nch, const bool cocg, uint8_t *&out_data) {
    emit_astc_block_impl<SimdLeaves>(pixels, cem, nch, cocg, out_data);
}
} // anonymous namespace

static int astc_round_up(const int v, const int align) { return align == 0 ? v : align * ((v + align - 1) / align); }

int GetRequiredMemory_ASTC4x4(const int w, const int h, const int pitch_align) {
    return astc_round_up(16 * ((w + 3) / 4), pitch_align) * ((h + 3) / 4);
}

template <int SrcChannels>
void CompressImage_ASTC4x4_RGBA(const uint8_t img_src[], const int w, const int h, uint8_t img_dst[],
                                const int dst_pitch, const bool is_ycocg) {
    alignas(16) uint8_t block[64];
    uint8_t *p_out = img_dst;

    const int w_aligned = w - (w % 4);
    const int h_aligned = h - (h % 4);
    const int pitch_pad = dst_pitch == 0 ? 0 : dst_pitch - 16 * ((w + 3) / 4);

    for (int j = 0; j < h_aligned; j += 4, img_src += 4 * w * SrcChannels) {
        for (int i = 0; i < w_aligned; i += 4) {
            extract4x4_rgba<SrcChannels>(&img_src[i * SrcChannels], w * SrcChannels, block);
            emit_astc_block(block, CEM_RGBA, 4, is_ycocg, p_out);
        }
        if (w_aligned != w) {
            extract4x4_rgba_partial<SrcChannels>(&img_src[w_aligned * SrcChannels], w * SrcChannels, w % 4, 4, block);
            emit_astc_block(block, CEM_RGBA, 4, is_ycocg, p_out);
        }
        p_out += pitch_pad;
    }
    for (int i = 0; i < w && h_aligned != h; i += 4) {
        extract4x4_rgba_partial<SrcChannels>(&img_src[i * SrcChannels], w * SrcChannels, std::min(4, w - i), h % 4,
                                             block);
        emit_astc_block(block, CEM_RGBA, 4, is_ycocg, p_out);
    }
}

// RG (normal-map style) source into CEM_RGBA with constant B/A endpoints; shared 3-bit weights on R,G.
template <int SrcChannels>
void CompressImage_ASTC4x4_RG(const uint8_t img_src[], const int w, const int h, uint8_t img_dst[],
                              const int dst_pitch) {
    alignas(16) uint8_t block[64];
    for (int i = 0; i < 16; i++) {
        block[i * 4 + 2] = 0;
        block[i * 4 + 3] = 255;
    }
    uint8_t *p_out = img_dst;

    const int w_aligned = w - (w % 4);
    const int h_aligned = h - (h % 4);
    const int pitch_pad = dst_pitch == 0 ? 0 : dst_pitch - 16 * ((w + 3) / 4);

    for (int j = 0; j < h_aligned; j += 4, img_src += 4 * w * SrcChannels) {
        for (int i = 0; i < w_aligned; i += 4) {
            extract4x4_rgba<SrcChannels>(&img_src[i * SrcChannels], w * SrcChannels, block);
            emit_astc_block(block, CEM_RGBA, 2, false, p_out);
        }
        if (w_aligned != w) {
            extract4x4_rgba_partial<SrcChannels>(&img_src[w_aligned * SrcChannels], w * SrcChannels, w % 4, 4, block);
            emit_astc_block(block, CEM_RGBA, 2, false, p_out);
        }
        p_out += pitch_pad;
    }
    for (int i = 0; i < w && h_aligned != h; i += 4) {
        extract4x4_rgba_partial<SrcChannels>(&img_src[i * SrcChannels], w * SrcChannels, std::min(4, w - i), h % 4,
                                             block);
        emit_astc_block(block, CEM_RGBA, 2, false, p_out);
    }
}

template <int SrcChannels>
void CompressImage_ASTC4x4_R8(const uint8_t img_src[], const int w, const int h, uint8_t img_dst[],
                              const int dst_pitch) {
    alignas(16) uint8_t block[64];
    // Seed G/B/A once; 1-channel extracts overwrite R only. Alpha SSE bias is constant.
    for (int i = 0; i < 16; i++) {
        block[i * 4 + 1] = 0;
        block[i * 4 + 2] = 0;
        block[i * 4 + 3] = 255;
    }
    uint8_t *p_out = img_dst;

    const int w_aligned = w - (w % 4);
    const int h_aligned = h - (h % 4);
    const int pitch_pad = dst_pitch == 0 ? 0 : dst_pitch - 16 * ((w + 3) / 4);

    for (int j = 0; j < h_aligned; j += 4, img_src += 4 * w * SrcChannels) {
        for (int i = 0; i < w_aligned; i += 4) {
            extract4x4_rgba<SrcChannels>(&img_src[i * SrcChannels], w * SrcChannels, block);
            emit_astc_block(block, CEM_LUMINANCE, 1, false, p_out);
        }
        if (w_aligned != w) {
            extract4x4_rgba_partial<SrcChannels>(&img_src[w_aligned * SrcChannels], w * SrcChannels, w % 4, 4, block);
            emit_astc_block(block, CEM_LUMINANCE, 1, false, p_out);
        }
        p_out += pitch_pad;
    }
    for (int i = 0; i < w && h_aligned != h; i += 4) {
        extract4x4_rgba_partial<SrcChannels>(&img_src[i * SrcChannels], w * SrcChannels, std::min(4, w - i), h % 4,
                                             block);
        emit_astc_block(block, CEM_LUMINANCE, 1, false, p_out);
    }
}

template void CompressImage_ASTC4x4_RGBA<4>(const uint8_t[], int, int, uint8_t[], int, bool);
template void CompressImage_ASTC4x4_RGBA<3>(const uint8_t[], int, int, uint8_t[], int, bool);
template void CompressImage_ASTC4x4_R8<1>(const uint8_t[], int, int, uint8_t[], int);
template void CompressImage_ASTC4x4_RG<2>(const uint8_t[], int, int, uint8_t[], int);

} // namespace Ray
