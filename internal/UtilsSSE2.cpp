#if defined(_M_IX86) || defined(_M_X64) || defined(__i386__) || defined(__x86_64__)
#include "Utils.h"

#ifdef __GNUC__
#pragma GCC push_options
#pragma GCC target("sse2")
#endif

#include <cassert>
#include <cstring>

#include <immintrin.h>
#include <xmmintrin.h>

#ifdef __linux__
//#define _mm_storeu_si32(mem_addr, a) _mm_store_ss((float *)mem_addr, _mm_castsi128_ps(a));
#define _mm_loadu_si32(mem_addr) _mm_castps_si128(_mm_load_ss((float const *)mem_addr))
#endif

#define _ABS(x) ((x) < 0 ? -(x) : (x))

//
// Internal functions
//

// https://developer.download.nvidia.com/whitepapers/2007/Real-Time-YCoCg-DXT-Compression/Real-Time%20YCoCg-DXT%20Compression.pdf

// WARNING: Reads 4 bytes outside of block in 3 channels case!
namespace Ray {
// clang-format off
// Copy rgb values and zero out alpha
static const __m128i RGB_to_RGBA = _mm_set_epi8(-1 /* Insert zero */, 11, 10, 9,
    -1 /* Insert zero */, 8, 7, 6,
    -1 /* Insert zero */, 5, 4, 3,
    -1 /* Insert zero */, 2, 1, 0);
// clang-format on

#ifdef __GNUC__
#pragma GCC push_options
#pragma GCC target("avx")
#endif
#ifdef __clang__
#pragma clang attribute push(__attribute__((target("ssse3"))), apply_to = function)
#endif

template <int Channels> void Extract4x4Block_SSSE3(const uint8_t src[], const int stride, uint8_t dst[64]) {
    for (int j = 0; j < 4; j++) {
        __m128i rgba;
        if (Channels == 4) {
            rgba = _mm_loadu_si128(reinterpret_cast<const __m128i *>(src));
        } else if (Channels == 3) {
            const __m128i rgb = _mm_loadu_si128(reinterpret_cast<const __m128i *>(src));
            rgba = _mm_shuffle_epi8(rgb, RGB_to_RGBA);
        }

        _mm_store_si128(reinterpret_cast<__m128i *>(dst), rgba);

        src += stride;
        dst += 4 * 4;
    }
}

template void Extract4x4Block_SSSE3<4 /* Channels */>(const uint8_t src[], const int stride, uint8_t dst[64]);
template void Extract4x4Block_SSSE3<3 /* Channels */>(const uint8_t src[], const int stride, uint8_t dst[64]);

#ifdef __GNUC__
#pragma GCC pop_options
#endif
#ifdef __clang__
#pragma clang attribute pop
#endif

static const __m128i CoCgInsetMul = _mm_set_epi16(1, 2, 2, 2, 1, 2, 2, 2);

template <bool UseAlpha, bool Is_YCoCg>
void GetMinMaxColorByBBox_SSE2(const uint8_t block[64], uint8_t min_color[4], uint8_t max_color[4]) {
    __m128i min_col = _mm_set1_epi8(-1 /* 255 */);
    __m128i max_col = _mm_set1_epi8(0);

    const auto *_4px_lines = reinterpret_cast<const __m128i *>(block);

    for (int i = 0; i < 4; i++) {
        min_col = _mm_min_epu8(min_col, _4px_lines[i]);
        max_col = _mm_max_epu8(max_col, _4px_lines[i]);
    }

    // Find horizontal min/max values
    min_col = _mm_min_epu8(min_col, _mm_srli_si128(min_col, 8));
    min_col = _mm_min_epu8(min_col, _mm_srli_si128(min_col, 4));

    max_col = _mm_max_epu8(max_col, _mm_srli_si128(max_col, 8));
    max_col = _mm_max_epu8(max_col, _mm_srli_si128(max_col, 4));

    if (!Is_YCoCg) {
        __m128i min_col_16 = _mm_unpacklo_epi8(min_col, _mm_setzero_si128());
        __m128i max_col_16 = _mm_unpacklo_epi8(max_col, _mm_setzero_si128());
        __m128i inset = _mm_sub_epi16(max_col_16, min_col_16);
        if (!UseAlpha) {
            inset = _mm_srli_epi16(inset, 4);
        } else {
            inset = _mm_mullo_epi16(inset, CoCgInsetMul);
            inset = _mm_srli_epi16(inset, 5);
        }
        min_col_16 = _mm_add_epi16(min_col_16, inset);
        max_col_16 = _mm_sub_epi16(max_col_16, inset);
        min_col = _mm_packus_epi16(min_col_16, min_col_16);
        max_col = _mm_packus_epi16(max_col_16, max_col_16);
    }

    _mm_storeu_si32(min_color, min_col);
    _mm_storeu_si32(max_color, max_col);
}

template void GetMinMaxColorByBBox_SSE2<false /* UseAlpha */, false /* Is_YCoCg */>(const uint8_t block[64],
                                                                                    uint8_t min_color[4],
                                                                                    uint8_t max_color[4]);
template void GetMinMaxColorByBBox_SSE2<true /* UseAlpha */, false /* Is_YCoCg */>(const uint8_t block[64],
                                                                                   uint8_t min_color[4],
                                                                                   uint8_t max_color[4]);
template void GetMinMaxColorByBBox_SSE2<true /* UseAlpha */, true /* Is_YCoCg */>(const uint8_t block[64],
                                                                                  uint8_t min_color[4],
                                                                                  uint8_t max_color[4]);

static const __m128i YCoCgScaleBias =
    _mm_set_epi8(0, 0, -128, -128, 0, 0, -128, -128, 0, 0, -128, -128, 0, 0, -128, -128);

void ScaleYCoCg_SSE2(uint8_t block[64], uint8_t min_color[3], uint8_t max_color[3]) {
    int m0 = _ABS(min_color[0] - 128);
    int m1 = _ABS(min_color[1] - 128);
    int m2 = _ABS(max_color[0] - 128);
    int m3 = _ABS(max_color[1] - 128);

    // clang-format off
    if (m1 > m0) m0 = m1;
    if (m3 > m2) m2 = m3;
    if (m2 > m0) m0 = m2;
    // clang-format on

    static const int s0 = 128 / 2 - 1;
    static const int s1 = 128 / 4 - 1;

    const int mask0 = -(m0 <= s0);
    const int mask1 = -(m0 <= s1);
    const int scale = 1 + (1 & mask0) + (2 & mask1);

    min_color[0] = (min_color[0] - 128) * scale + 128;
    min_color[1] = (min_color[1] - 128) * scale + 128;
    min_color[2] = (scale - 1) * 8;

    max_color[0] = (max_color[0] - 128) * scale + 128;
    max_color[1] = (max_color[1] - 128) * scale + 128;
    max_color[2] = (scale - 1) * 8;

    const __m128i _scale = _mm_set_epi16(1, scale, 1, scale, 1, scale, 1, scale);

    const __m128i _mask = _mm_set_epi8(-1, -1, ~(scale - 1), ~(scale - 1), -1, -1, ~(scale - 1), ~(scale - 1), -1, -1,
                                       ~(scale - 1), ~(scale - 1), -1, -1, ~(scale - 1), ~(scale - 1));

    auto *_4px_lines = reinterpret_cast<__m128i *>(block);

    for (int i = 0; i < 4; i++) {
        __m128i cur_col = _mm_load_si128(&_4px_lines[i]);

        cur_col = _mm_add_epi8(cur_col, YCoCgScaleBias);
        cur_col = _mm_mullo_epi16(cur_col, _scale);
        cur_col = _mm_and_si128(cur_col, _mask);
        cur_col = _mm_sub_epi8(cur_col, YCoCgScaleBias);

        _mm_store_si128(&_4px_lines[i], cur_col);
    }
}

static const int InsetColorShift = 4;
static const int InsetAlphaShift = 5;

static const __m128i InsetYCoCgRound =
    _mm_set_epi16(0, 0, 0, 0, (1 << (InsetAlphaShift - 1)) - 1, (1 << (InsetColorShift - 1)) - 1,
                  (1 << (InsetColorShift - 1)) - 1, (1 << (InsetColorShift - 1)) - 1);
static const __m128i InsetYCoCgMask = _mm_set_epi16(-1, 0, -1, -1, -1, 0, -1, -1);

static const __m128i InsetShiftUp =
    _mm_set_epi16(0, 0, 0, 0, 1 << InsetAlphaShift, 1 << InsetColorShift, 1 << InsetColorShift, 1 << InsetColorShift);
static const __m128i InsetShiftDown =
    _mm_set_epi16(0, 0, 0, 0, 1 << (16 - InsetAlphaShift), 1 << (16 - InsetColorShift), 1 << (16 - InsetColorShift),
                  1 << (16 - InsetColorShift));

static const __m128i Inset565Mask =
    _mm_set_epi16(0xff, 0b11111000, 0b11111100, 0b11111000, 0xff, 0b11111000, 0b11111100, 0b11111000);
static const __m128i Inset565Rep =
    _mm_set_epi16(0, 1 << (16 - 5), 1 << (16 - 6), 1 << (16 - 5), 0, 1 << (16 - 5), 1 << (16 - 6), 1 << (16 - 5));

void InsetYCoCgBBox_SSE2(uint8_t min_color[4], uint8_t max_color[4]) {
    __m128i min_col, max_col;

    min_col = _mm_loadu_si32(min_color);
    max_col = _mm_loadu_si32(max_color);

    min_col = _mm_unpacklo_epi8(min_col, _mm_setzero_si128());
    max_col = _mm_unpacklo_epi8(max_col, _mm_setzero_si128());

    __m128i inset = _mm_sub_epi16(max_col, min_col);
    inset = _mm_sub_epi16(inset, InsetYCoCgRound);
    inset = _mm_and_si128(inset, InsetYCoCgMask);

    min_col = _mm_mullo_epi16(min_col, InsetShiftUp);
    max_col = _mm_mullo_epi16(max_col, InsetShiftUp);

    min_col = _mm_add_epi16(min_col, inset);
    max_col = _mm_sub_epi16(max_col, inset);

    min_col = _mm_mulhi_epi16(min_col, InsetShiftDown);
    max_col = _mm_mulhi_epi16(max_col, InsetShiftDown);

    min_col = _mm_max_epi16(min_col, _mm_setzero_si128());
    max_col = _mm_max_epi16(max_col, _mm_setzero_si128());

    min_col = _mm_and_si128(min_col, Inset565Mask);
    max_col = _mm_and_si128(max_col, Inset565Mask);

    const __m128i temp0 = _mm_mulhi_epi16(min_col, Inset565Rep);
    const __m128i temp1 = _mm_mulhi_epi16(max_col, Inset565Rep);

    min_col = _mm_or_si128(min_col, temp0);
    max_col = _mm_or_si128(max_col, temp1);

    min_col = _mm_packus_epi16(min_col, min_col);
    max_col = _mm_packus_epi16(max_col, max_col);

    _mm_storeu_si32(min_color, min_col);
    _mm_storeu_si32(max_color, max_col);
}

static const __m128i CoCgMask = _mm_set_epi16(0, -1, 0, -1, 0, -1, 0, -1);
static const __m128i CoCgDiagonalMask = _mm_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0);

static const __m128i Zeroes_128 = _mm_setzero_si128();
static const __m128i Ones_i16 = _mm_set1_epi16(1);
static const __m128i Twos_i16 = _mm_set1_epi16(2);
static const __m128i Eights_i16 = _mm_set1_epi16(8);

void SelectYCoCgDiagonal_SSE2(const uint8_t block[64], uint8_t min_color[3], uint8_t max_color[3]) {
    // load block
    __m128i line0 = _mm_load_si128(reinterpret_cast<const __m128i *>(block));
    __m128i line1 = _mm_load_si128(reinterpret_cast<const __m128i *>(block + 16));
    __m128i line2 = _mm_load_si128(reinterpret_cast<const __m128i *>(block + 32));
    __m128i line3 = _mm_load_si128(reinterpret_cast<const __m128i *>(block + 48));

    // mask out everything except CoCg channels
    __m128i line0_CoCg = _mm_and_si128(line0, CoCgMask);
    __m128i line1_CoCg = _mm_and_si128(line1, CoCgMask);
    __m128i line2_CoCg = _mm_and_si128(line2, CoCgMask);
    __m128i line3_CoCg = _mm_and_si128(line3, CoCgMask);

    // merge pairs of CoCg channels
    __m128i line01_CoCg = _mm_or_si128(line0_CoCg, _mm_slli_si128(line1_CoCg, 2));
    __m128i line23_CoCg = _mm_or_si128(line2_CoCg, _mm_slli_si128(line3_CoCg, 2));

    __m128i min_col = _mm_loadu_si32(min_color);
    __m128i max_col = _mm_loadu_si32(max_color);

    __m128i mid = _mm_avg_epu8(min_col, max_col);
    mid = _mm_shufflelo_epi16(mid, _MM_SHUFFLE(0, 0, 0, 0));
    mid = _mm_shuffle_epi32(mid, _MM_SHUFFLE(0, 0, 0, 0));

    __m128i tmp1 = _mm_max_epu8(mid, line01_CoCg);
    __m128i tmp3 = _mm_max_epu8(mid, line23_CoCg);
    tmp1 = _mm_cmpeq_epi8(tmp1, line01_CoCg);
    tmp3 = _mm_cmpeq_epi8(tmp3, line23_CoCg);

    __m128i tmp0 = _mm_srli_si128(tmp1, 1);
    __m128i tmp2 = _mm_srli_si128(tmp3, 1);

    tmp0 = _mm_xor_si128(tmp0, tmp1);
    tmp2 = _mm_xor_si128(tmp2, tmp3);
    tmp0 = _mm_and_si128(tmp0, Ones_i16);
    tmp2 = _mm_and_si128(tmp2, Ones_i16);

    tmp0 = _mm_add_epi16(tmp0, tmp2);
    tmp0 = _mm_sad_epu8(tmp0, Zeroes_128);
    tmp1 = _mm_shuffle_epi32(tmp0, _MM_SHUFFLE(1, 0, 3, 2));

    tmp0 = _mm_add_epi16(tmp0, tmp1);
    tmp0 = _mm_cmpgt_epi16(tmp0, Eights_i16);
    tmp0 = _mm_and_si128(tmp0, CoCgDiagonalMask);

    min_col = _mm_xor_si128(min_col, max_col);
    tmp0 = _mm_and_si128(tmp0, min_col);
    max_col = _mm_xor_si128(max_col, tmp0);
    min_col = _mm_xor_si128(min_col, max_col);

    _mm_storeu_si32(min_color, min_col);
    _mm_storeu_si32(max_color, max_col);
}

static const __m128i RGB565Mask = _mm_set_epi8(0, 0, 0, 0, 0, char(0b11111000), char(0b11111100), char(0b11111000), 0,
                                               0, 0, 0, 0, char(0b11111000), char(0b11111100), char(0b11111000));
// multiplier used to emulate division by 3
static const __m128i DivBy3_i16 = _mm_set1_epi16((1 << 16) / 3 + 1);

void EmitColorIndices_SSE2(const uint8_t block[64], const uint8_t min_color[4], const uint8_t max_color[4],
                           uint8_t *&out_data) {
    __m128i result = _mm_setzero_si128();

    // Find 4 colors on the line through min - max color
    // compute color0 (max_color)
    __m128i color0 = _mm_loadu_si32(max_color);
    color0 = _mm_and_si128(color0, RGB565Mask);
    color0 = _mm_unpacklo_epi8(color0, _mm_setzero_si128());
    __m128i rb = _mm_shufflelo_epi16(color0, _MM_SHUFFLE(3, 2, 3, 0));
    __m128i g = _mm_shufflelo_epi16(color0, _MM_SHUFFLE(3, 3, 1, 3));
    rb = _mm_srli_epi16(rb, 5);
    g = _mm_srli_epi16(g, 6);
    color0 = _mm_or_si128(color0, rb);
    color0 = _mm_or_si128(color0, g);

    // compute color1 (min_color)
    __m128i color1 = _mm_loadu_si32(min_color);
    color1 = _mm_and_si128(color1, RGB565Mask);
    color1 = _mm_unpacklo_epi8(color1, _mm_setzero_si128());
    rb = _mm_shufflelo_epi16(color1, _MM_SHUFFLE(3, 2, 3, 0));
    g = _mm_shufflelo_epi16(color1, _MM_SHUFFLE(3, 3, 1, 3));
    rb = _mm_srli_epi16(rb, 5);
    g = _mm_srli_epi16(g, 6);
    color1 = _mm_or_si128(color1, rb);
    color1 = _mm_or_si128(color1, g);

    // compute and pack color3
    __m128i color3 = _mm_add_epi16(color1, color1);
    color3 = _mm_add_epi16(color0, color3);
    color3 = _mm_mulhi_epi16(color3, DivBy3_i16);
    color3 = _mm_packus_epi16(color3, _mm_setzero_si128());
    color3 = _mm_shuffle_epi32(color3, _MM_SHUFFLE(1, 0, 1, 0));

    // compute and pack color2
    __m128i color2 = _mm_add_epi16(color0, color0);
    color2 = _mm_add_epi16(color2, color1);
    color2 = _mm_mulhi_epi16(color2, DivBy3_i16);
    color2 = _mm_packus_epi16(color2, _mm_setzero_si128());
    color2 = _mm_shuffle_epi32(color2, _MM_SHUFFLE(1, 0, 1, 0));

    // pack color1
    color1 = _mm_packus_epi16(color1, _mm_setzero_si128());
    color1 = _mm_shuffle_epi32(color1, _MM_SHUFFLE(1, 0, 1, 0));

    // pack color0
    color0 = _mm_packus_epi16(color0, _mm_setzero_si128());
    color0 = _mm_shuffle_epi32(color0, _MM_SHUFFLE(1, 0, 1, 0));

    for (int i = 32; i >= 0; i -= 32) {
        // load 4 colors
        __m128i color_hi = _mm_loadu_si64(block + i + 0);
        color_hi = _mm_shuffle_epi32(color_hi, _MM_SHUFFLE(3, 1, 2, 0));
        __m128i color_lo = _mm_loadu_si64(block + i + 8);
        color_lo = _mm_shuffle_epi32(color_lo, _MM_SHUFFLE(3, 1, 2, 0));

        // compute the sum of abs diff for each color
        __m128i d_hi = _mm_sad_epu8(color_hi, color0);
        __m128i d_lo = _mm_sad_epu8(color_lo, color0);
        __m128i d0 = _mm_packs_epi32(d_hi, d_lo);
        d_hi = _mm_sad_epu8(color_hi, color1);
        d_lo = _mm_sad_epu8(color_lo, color1);
        __m128i d1 = _mm_packs_epi32(d_hi, d_lo);
        d_hi = _mm_sad_epu8(color_hi, color2);
        d_lo = _mm_sad_epu8(color_lo, color2);
        __m128i d2 = _mm_packs_epi32(d_hi, d_lo);
        d_hi = _mm_sad_epu8(color_hi, color3);
        d_lo = _mm_sad_epu8(color_lo, color3);
        __m128i d3 = _mm_packs_epi32(d_hi, d_lo);

        // load next 4 colors
        color_hi = _mm_loadu_si64(block + i + 16);
        color_hi = _mm_shuffle_epi32(color_hi, _MM_SHUFFLE(3, 1, 2, 0));
        color_lo = _mm_loadu_si64(block + i + 24);
        color_lo = _mm_shuffle_epi32(color_lo, _MM_SHUFFLE(3, 1, 2, 0));

        // compute the sum of abs diff for each color and combine with prev result
        d_hi = _mm_sad_epu8(color_hi, color0);
        d_lo = _mm_sad_epu8(color_lo, color0);
        d_lo = _mm_packs_epi32(d_hi, d_lo);
        d0 = _mm_packs_epi32(d0, d_lo);
        d_hi = _mm_sad_epu8(color_hi, color1);
        d_lo = _mm_sad_epu8(color_lo, color1);
        d_lo = _mm_packs_epi32(d_hi, d_lo);
        d1 = _mm_packs_epi32(d1, d_lo);
        d_hi = _mm_sad_epu8(color_hi, color2);
        d_lo = _mm_sad_epu8(color_lo, color2);
        d_lo = _mm_packs_epi32(d_hi, d_lo);
        d2 = _mm_packs_epi32(d2, d_lo);
        d_hi = _mm_sad_epu8(color_hi, color3);
        d_lo = _mm_sad_epu8(color_lo, color3);
        d_lo = _mm_packs_epi32(d_hi, d_lo);
        d3 = _mm_packs_epi32(d3, d_lo);

        // compare the distances
        __m128i b0 = _mm_cmpgt_epi16(d0, d3);
        __m128i b1 = _mm_cmpgt_epi16(d1, d2);
        __m128i b2 = _mm_cmpgt_epi16(d0, d2);
        __m128i b3 = _mm_cmpgt_epi16(d1, d3);
        __m128i b4 = _mm_cmpgt_epi16(d2, d3);

        // compute color index
        __m128i x0 = _mm_and_si128(b2, b1);
        __m128i x1 = _mm_and_si128(b3, b0);
        __m128i x2 = _mm_and_si128(b4, b0);
        __m128i index_bit0 = _mm_or_si128(x0, x1);
        index_bit0 = _mm_and_si128(index_bit0, Twos_i16);
        __m128i index_bit1 = _mm_and_si128(x2, Ones_i16);
        __m128i index = _mm_or_si128(index_bit1, index_bit0);

        // pack index into result
        __m128i index_hi = _mm_shuffle_epi32(index, _MM_SHUFFLE(1, 0, 3, 2));
        index_hi = _mm_unpacklo_epi16(index_hi, Zeroes_128);
        index_hi = _mm_slli_epi32(index_hi, 8);
        __m128i index_lo = _mm_unpacklo_epi16(index, Zeroes_128);
        result = _mm_slli_epi32(result, 16);
        result = _mm_or_si128(result, index_hi);
        result = _mm_or_si128(result, index_lo);
    }

    // pack 16 2-bit color indices into a single 32-bit value
    __m128i result1 = _mm_shuffle_epi32(result, _MM_SHUFFLE(0, 3, 2, 1));
    __m128i result2 = _mm_shuffle_epi32(result, _MM_SHUFFLE(1, 0, 3, 2));
    __m128i result3 = _mm_shuffle_epi32(result, _MM_SHUFFLE(2, 1, 0, 3));
    result1 = _mm_slli_epi32(result1, 2);
    result2 = _mm_slli_epi32(result2, 4);
    result3 = _mm_slli_epi32(result3, 6);
    result = _mm_or_si128(result, result1);
    result = _mm_or_si128(result, result2);
    result = _mm_or_si128(result, result3);

    _mm_storeu_si32(out_data, result);
    out_data += 4;
}

// multiplier used to emulate division by 7
static const __m128i DivBy7_i16 = _mm_set1_epi16((1 << 16) / 7 + 1);
// multiplier used to emulate division by 14
static const __m128i DivBy14_i16 = _mm_set1_epi16((1 << 16) / 14 + 1);
static const __m128i ScaleBy_66554400_i16 = _mm_setr_epi16(6, 6, 5, 5, 4, 4, 0, 0);
static const __m128i ScaleBy_11223300_i16 = _mm_setr_epi16(1, 1, 2, 2, 3, 3, 0, 0);

static const __m128i Ones_i8 = _mm_set1_epi8(1);
static const __m128i Twos_i8 = _mm_set1_epi8(2);
static const __m128i Sevens_i8 = _mm_set1_epi8(7);

static const __m128i AlphaMask0 = _mm_setr_epi32(7 << 0, 0, 7 << 0, 0);
static const __m128i AlphaMask1 = _mm_setr_epi32(7 << 3, 0, 7 << 3, 0);
static const __m128i AlphaMask2 = _mm_setr_epi32(7 << 6, 0, 7 << 6, 0);
static const __m128i AlphaMask3 = _mm_setr_epi32(7 << 9, 0, 7 << 9, 0);
static const __m128i AlphaMask4 = _mm_setr_epi32(7 << 12, 0, 7 << 12, 0);
static const __m128i AlphaMask5 = _mm_setr_epi32(7 << 15, 0, 7 << 15, 0);
static const __m128i AlphaMask6 = _mm_setr_epi32(7 << 18, 0, 7 << 18, 0);
static const __m128i AlphaMask7 = _mm_setr_epi32(7 << 21, 0, 7 << 21, 0);

void EmitAlphaIndices_SSE2(const uint8_t block[64], const uint8_t min_alpha, const uint8_t max_alpha,
                           uint8_t *&out_data) {
    __m128i line0 = _mm_load_si128(reinterpret_cast<const __m128i *>(block));
    __m128i line1 = _mm_load_si128(reinterpret_cast<const __m128i *>(block + 16));
    __m128i line2 = _mm_load_si128(reinterpret_cast<const __m128i *>(block + 32));
    __m128i line3 = _mm_load_si128(reinterpret_cast<const __m128i *>(block + 48));

    line0 = _mm_srli_epi32(line0, 24);
    line1 = _mm_srli_epi32(line1, 24);
    line2 = _mm_srli_epi32(line2, 24);
    line3 = _mm_srli_epi32(line3, 24);

    __m128i line01 = _mm_packus_epi16(line0, line1);
    __m128i line23 = _mm_packus_epi16(line2, line3);

    // pack all 16 alpha values
    __m128i alpha = _mm_packus_epi16(line01, line23);

    __m128i max = _mm_set1_epi16(max_alpha);
    __m128i min = _mm_set1_epi16(min_alpha);

    // compute midpoint offset between any two interpolated alpha values
    __m128i mid = _mm_sub_epi16(max, min);
    mid = _mm_mulhi_epi16(mid, DivBy14_i16);

    // compute first midpoint
    __m128i ab1 = min;
    ab1 = _mm_add_epi16(ab1, mid);
    ab1 = _mm_packus_epi16(ab1, ab1);

    // compute the next three midpoints
    __m128i max456 = _mm_mullo_epi16(max, ScaleBy_66554400_i16);
    __m128i min123 = _mm_mullo_epi16(min, ScaleBy_11223300_i16);
    __m128i ab234 = _mm_add_epi16(max456, min123);
    ab234 = _mm_mulhi_epi16(ab234, DivBy7_i16);
    ab234 = _mm_add_epi16(ab234, mid);
    __m128i ab2 = _mm_shuffle_epi32(ab234, _MM_SHUFFLE(0, 0, 0, 0));
    ab2 = _mm_packus_epi16(ab2, ab2);
    __m128i ab3 = _mm_shuffle_epi32(ab234, _MM_SHUFFLE(1, 1, 1, 1));
    ab3 = _mm_packus_epi16(ab3, ab3);
    __m128i ab4 = _mm_shuffle_epi32(ab234, _MM_SHUFFLE(2, 2, 2, 2));
    ab4 = _mm_packus_epi16(ab4, ab4);

    // compute the last three midpoints
    __m128i max123 = _mm_mullo_epi16(max, ScaleBy_11223300_i16);
    __m128i min456 = _mm_mullo_epi16(min, ScaleBy_66554400_i16);
    __m128i ab567 = _mm_add_epi16(max123, min456);
    ab567 = _mm_mulhi_epi16(ab567, DivBy7_i16);
    ab567 = _mm_add_epi16(ab567, mid);
    __m128i ab5 = _mm_shuffle_epi32(ab567, _MM_SHUFFLE(2, 2, 2, 2));
    ab5 = _mm_packus_epi16(ab5, ab5);
    __m128i ab6 = _mm_shuffle_epi32(ab567, _MM_SHUFFLE(1, 1, 1, 1));
    ab6 = _mm_packus_epi16(ab6, ab6);
    __m128i ab7 = _mm_shuffle_epi32(ab567, _MM_SHUFFLE(0, 0, 0, 0));
    ab7 = _mm_packus_epi16(ab7, ab7);

    // compare the alpha values to the midpoints
    __m128i b1 = _mm_min_epu8(ab1, alpha);
    b1 = _mm_cmpeq_epi8(b1, alpha);
    b1 = _mm_and_si128(b1, Ones_i8);
    __m128i b2 = _mm_min_epu8(ab2, alpha);
    b2 = _mm_cmpeq_epi8(b2, alpha);
    b2 = _mm_and_si128(b2, Ones_i8);
    __m128i b3 = _mm_min_epu8(ab3, alpha);
    b3 = _mm_cmpeq_epi8(b3, alpha);
    b3 = _mm_and_si128(b3, Ones_i8);
    __m128i b4 = _mm_min_epu8(ab4, alpha);
    b4 = _mm_cmpeq_epi8(b4, alpha);
    b4 = _mm_and_si128(b4, Ones_i8);
    __m128i b5 = _mm_min_epu8(ab5, alpha);
    b5 = _mm_cmpeq_epi8(b5, alpha);
    b5 = _mm_and_si128(b5, Ones_i8);
    __m128i b6 = _mm_min_epu8(ab6, alpha);
    b6 = _mm_cmpeq_epi8(b6, alpha);
    b6 = _mm_and_si128(b6, Ones_i8);
    __m128i b7 = _mm_min_epu8(ab7, alpha);
    b7 = _mm_cmpeq_epi8(b7, alpha);
    b7 = _mm_and_si128(b7, Ones_i8);

    // compute alpha indices
    __m128i index = _mm_adds_epu8(b1, b2);
    index = _mm_adds_epu8(index, b3);
    index = _mm_adds_epu8(index, b4);
    index = _mm_adds_epu8(index, b5);
    index = _mm_adds_epu8(index, b6);
    index = _mm_adds_epu8(index, b7);

    // convert natural index ordering to DXT index ordering
    index = _mm_adds_epu8(index, Ones_i8);
    index = _mm_and_si128(index, Sevens_i8);
    __m128i swapMinMax = _mm_cmpgt_epi8(Twos_i8, index);
    swapMinMax = _mm_and_si128(swapMinMax, Ones_i8);
    index = _mm_xor_si128(index, swapMinMax);

    // pack the 16 3-bit indices into 6 bytes
    __m128i index0 = _mm_and_si128(index, AlphaMask0);
    __m128i index1 = _mm_srli_epi64(index, 8 - 3);
    index1 = _mm_and_si128(index1, AlphaMask1);
    __m128i index2 = _mm_srli_epi64(index, 16 - 6);
    index2 = _mm_and_si128(index2, AlphaMask2);
    __m128i index3 = _mm_srli_epi64(index, 24 - 9);
    index3 = _mm_and_si128(index3, AlphaMask3);
    __m128i index4 = _mm_srli_epi64(index, 32 - 12);
    index4 = _mm_and_si128(index4, AlphaMask4);
    __m128i index5 = _mm_srli_epi64(index, 40 - 15);
    index5 = _mm_and_si128(index5, AlphaMask5);
    __m128i index6 = _mm_srli_epi64(index, 48 - 18);
    index6 = _mm_and_si128(index6, AlphaMask6);
    __m128i index7 = _mm_srli_epi64(index, 56 - 21);
    index7 = _mm_and_si128(index7, AlphaMask7);
    index = _mm_or_si128(index0, index1);
    index = _mm_or_si128(index, index2);
    index = _mm_or_si128(index, index3);
    index = _mm_or_si128(index, index4);
    index = _mm_or_si128(index, index5);
    index = _mm_or_si128(index, index6);
    index = _mm_or_si128(index, index7);

    _mm_storeu_si32(out_data, index);
    index = _mm_shuffle_epi32(index, _MM_SHUFFLE(1, 0, 3, 2));
    _mm_storeu_si32(out_data + 3, index);

    out_data += 6;
}

} // namespace Ren

#undef _ABS

#ifdef __GNUC__
#pragma GCC pop_options
#endif

#endif // defined(_M_IX86) || defined(_M_X64) || defined(__i386__) || defined(__x86_64__)