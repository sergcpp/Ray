#if defined(__ARM_NEON__) || defined(__arm__) || defined(__aarch64__) || defined(_M_ARM) || defined(_M_ARM64)
#include "TextureUtils.h"

#include <arm_neon.h>

#ifdef __GNUC__
#define force_inline __attribute__((always_inline)) inline
#endif
#ifdef _MSC_VER
#define force_inline __forceinline
#endif

#define _ABS(x) ((x) < 0 ? -(x) : (x))

namespace Ray {
// clang-format off
// Copy rgb values and zero out alpha
static const int8_t RGB_to_RGBA[] = {-1 /* Insert zero */, 11, 10, 9,
                                     -1 /* Insert zero */, 8, 7, 6,
                                     -1 /* Insert zero */, 5, 4, 3,
                                     -1 /* Insert zero */, 2, 1, 0};
// clang-format on

force_inline int16x8_t _mm_mulhi_epi16(int16x8_t a, int16x8_t b) {
    int16x4_t a3210 = vget_low_s16(a);
    int16x4_t b3210 = vget_low_s16(b);
    int32x4_t ab3210 = vmull_s16(a3210, b3210); // 3333222211110000
    int16x4_t a7654 = vget_high_s16(a);
    int16x4_t b7654 = vget_high_s16(b);
    int32x4_t ab7654 = vmull_s16(a7654, b7654); // 7777666655554444
    uint16x8x2_t r = vuzpq_u16(vreinterpretq_u16_s32(ab3210), vreinterpretq_u16_s32(ab7654));
    return r.val[1];
}

template <int imm> force_inline int16x8_t _mm_shufflelo_epi16(int16x8_t a) {
    int16x8_t ret = a;
    int16x4_t lowBits = vget_low_s16(ret);
    ret = vsetq_lane_s16(vget_lane_s16(lowBits, (imm) & (0x3)), ret, 0);
    ret = vsetq_lane_s16(vget_lane_s16(lowBits, ((imm) >> 2) & 0x3), ret, 1);
    ret = vsetq_lane_s16(vget_lane_s16(lowBits, ((imm) >> 4) & 0x3), ret, 2);
    ret = vsetq_lane_s16(vget_lane_s16(lowBits, ((imm) >> 6) & 0x3), ret, 3);
    return ret;
}

template <int imm> force_inline int32x4_t _mm_shuffle_epi32(int32x4_t a) {
    return vsetq_lane_s32(vgetq_lane_s32(a, ((imm) >> 6) & 0x3),
                          vsetq_lane_s32(vgetq_lane_s32(a, ((imm) >> 4) & 0x3),
                                         vsetq_lane_s32(vgetq_lane_s32(a, ((imm) >> 2) & 0x3),
                                                        vmovq_n_s32(vgetq_lane_s32(a, (imm) & (0x3))), 1),
                                         2),
                          3);
}

force_inline int8x16_t _mm_sad_epu8(int8x16_t a, int8x16_t b) {
    uint16x8_t t = vpaddlq_u8(vabdq_u8((uint8x16_t)a, (uint8x16_t)b));
    return vpaddlq_u32(vpaddlq_u16(t));
}

force_inline int32x4_t _mm_packs_epi32(int32x4_t a, int32x4_t b) { return vcombine_s16(vqmovn_s32(a), vqmovn_s32(b)); }

#if !defined(__aarch64__) && !defined(_M_ARM64)
force_inline int8x16_t vzip1q_s8(int8x16_t a, int8x16_t b) {
    int8x8_t a1 = vget_low_s16(a);
    int8x8_t b1 = vget_low_s16(b);
    int8x8x2_t result = vzip_s8(a1, b1);
    return vcombine_s8(result.val[0], result.val[1]);
}

force_inline int16x8_t vzip1q_s16(int16x8_t a, int16x8_t b) {
    int16x4_t a1 = vget_low_s16(a);
    int16x4_t b1 = vget_low_s16(b);
    int16x4x2_t result = vzip_s16(a1, b1);
    return vcombine_s16(result.val[0], result.val[1]);
}
#endif

#ifndef _MM_SHUFFLE
#define _MM_SHUFFLE(fp3, fp2, fp1, fp0) (((fp3) << 6) | ((fp2) << 4) | ((fp1) << 2) | ((fp0)))
#endif

template <int Channels> void Extract4x4Block_NEON(const uint8_t src[], const int stride, uint8_t dst[64]) {
    for (int j = 0; j < 4; j++) {
        int32x4_t rgba;
        if (Channels == 4) {
            rgba = vld1q_s32(reinterpret_cast<const int32_t *>(src));
        } else if (Channels == 3) {
#if defined(__aarch64__) || defined(_M_ARM64)
            const int32x4_t rgb = vld1q_u8(src);
            rgba = vqtbl1q_s8(rgb, vld1q_s8(RGB_to_RGBA));
#else
            const int32x2_t rgb_lo = vld1_u8(src);
            const int32x2_t rgb_hi = vld1_u8(src + 6);

            int8x8_t index = vld1_s8(RGB_to_RGBA + 2);

            int8x8_t res_hi = vtbl1_s8(rgb_hi, index);
            int8x8_t res_lo = vtbl1_s8(rgb_lo, index);

            rgba = vcombine_s32(res_hi, res_lo);
#endif
        }

        vst1q_s32(reinterpret_cast<int32_t *>(dst), rgba);

        src += stride;
        dst += 4 * 4;
    }
}

template void Extract4x4Block_NEON<4 /* Channels */>(const uint8_t src[], const int stride, uint8_t dst[64]);
template void Extract4x4Block_NEON<3 /* Channels */>(const uint8_t src[], const int stride, uint8_t dst[64]);

static const int16_t CoCgInsetMul[] = {1, 2, 2, 2, 1, 2, 2, 2};

template <bool UseAlpha, bool Is_YCoCg>
void GetMinMaxColorByBBox_NEON(const uint8_t block[64], uint8_t min_color[4], uint8_t max_color[4]) {
    int8x16_t min_col = vdupq_n_s8(-1 /* 255 */);
    int8x16_t max_col = vdupq_n_s8(0);

    const auto *_4px_lines = reinterpret_cast<const int8x16_t *>(block);

    for (int i = 0; i < 4; i++) {
        min_col = vminq_u8(min_col, _4px_lines[i]);
        max_col = vmaxq_u8(max_col, _4px_lines[i]);
    }

    // Find horizontal min/max values
    min_col = vminq_u8(min_col, vextq_u8(min_col, vdupq_n_u8(0), 8));
    min_col = vminq_u8(min_col, vextq_u8(min_col, vdupq_n_u8(0), 4));

    max_col = vmaxq_u8(max_col, vextq_u8(max_col, vdupq_n_u8(0), 8));
    max_col = vmaxq_u8(max_col, vextq_u8(max_col, vdupq_n_u8(0), 4));

    if (!Is_YCoCg) {
        int16x8_t min_col_16 = vzip1q_s8(min_col, vdupq_n_s32(0));
        int16x8_t max_col_16 = vzip1q_s8(max_col, vdupq_n_s32(0));
        int16x8_t inset = vsubq_s16(max_col_16, min_col_16);
        if (!UseAlpha) {
            inset = vshlq_u16(inset, vdupq_n_s16(-4));
        } else {
            inset = vmulq_s16(inset, vld1q_s16(CoCgInsetMul));
            inset = vshlq_u16(inset, vdupq_n_s16(-5));
        }
        min_col_16 = vaddq_s16(min_col_16, inset);
        max_col_16 = vsubq_s16(max_col_16, inset);

        min_col = vcombine_u8(vqmovun_s16(min_col_16), vqmovun_s16(min_col_16));
        max_col = vcombine_u8(vqmovun_s16(max_col_16), vqmovun_s16(max_col_16));
    }

    vst1q_lane_s32(reinterpret_cast<int32_t *>(min_color), min_col, 0);
    vst1q_lane_s32(reinterpret_cast<int32_t *>(max_color), max_col, 0);
}

template void GetMinMaxColorByBBox_NEON<false /* UseAlpha */, false /* Is_YCoCg */>(const uint8_t block[64],
                                                                                    uint8_t min_color[4],
                                                                                    uint8_t max_color[4]);
template void GetMinMaxColorByBBox_NEON<true /* UseAlpha */, false /* Is_YCoCg */>(const uint8_t block[64],
                                                                                   uint8_t min_color[4],
                                                                                   uint8_t max_color[4]);
template void GetMinMaxColorByBBox_NEON<true /* UseAlpha */, true /* Is_YCoCg */>(const uint8_t block[64],
                                                                                  uint8_t min_color[4],
                                                                                  uint8_t max_color[4]);

void GetMinMaxAlphaByBBox_NEON(const uint8_t block[16], uint8_t &min_alpha, uint8_t &max_alpha) {
    uint8x16_t min_col = vld1q_u8(block);
    uint8x16_t max_col = min_col;

    // Find horizontal min/max values
    min_col = vminq_u8(min_col, vextq_u8(min_col, vdupq_n_u8(0), 8));
    min_col = vminq_u8(min_col, vextq_u8(min_col, vdupq_n_u8(0), 4));
    min_col = vminq_u8(min_col, vextq_u8(min_col, vdupq_n_u8(0), 2));
    min_col = vminq_u8(min_col, vextq_u8(min_col, vdupq_n_u8(0), 1));

    max_col = vmaxq_u8(max_col, vextq_u8(max_col, vdupq_n_u8(0), 8));
    max_col = vmaxq_u8(max_col, vextq_u8(max_col, vdupq_n_u8(0), 4));
    max_col = vmaxq_u8(max_col, vextq_u8(max_col, vdupq_n_u8(0), 2));
    max_col = vmaxq_u8(max_col, vextq_u8(max_col, vdupq_n_u8(0), 1));

    min_alpha = vgetq_lane_u8(min_col, 0);
    max_alpha = vgetq_lane_u8(max_col, 0);
}

alignas(16) static const int8_t YCoCgScaleBias[] = {-128, -128, 0, 0, -128, -128, 0, 0,
                                                    -128, -128, 0, 0, -128, -128, 0, 0};

void ScaleYCoCg_NEON(uint8_t block[64], uint8_t min_color[3], uint8_t max_color[3]) {
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

    alignas(16) const int16_t temp[] = {int16_t(scale), 1, int16_t(scale), 1, int16_t(scale), 1, int16_t(scale), 1};
    const int16x8_t _scale = vld1q_s16(temp);

    alignas(16) const int8_t temp2[] = {
        int8_t(~(scale - 1)), int8_t(~(scale - 1)), -1, -1, int8_t(~(scale - 1)), int8_t(~(scale - 1)), -1, -1,
        int8_t(~(scale - 1)), int8_t(~(scale - 1)), -1, -1, int8_t(~(scale - 1)), int8_t(~(scale - 1)), -1, -1};
    const int8x16_t _mask = vld1q_s8(temp2);

    auto *_4px_lines = reinterpret_cast<int8x16_t *>(block);

    for (int i = 0; i < 4; i++) {
        int8x16_t cur_col = _4px_lines[i];

        cur_col = vaddq_s8(cur_col, vld1q_s8(YCoCgScaleBias));
        cur_col = vmulq_s16(cur_col, _scale);
        cur_col = vandq_s32(cur_col, _mask);
        cur_col = vsubq_s8(cur_col, vld1q_s8(YCoCgScaleBias));

        vst1q_s32(reinterpret_cast<int32_t *>(&_4px_lines[i]), cur_col);
    }
}

static const int InsetColorShift = 4;
static const int InsetAlphaShift = 5;

alignas(16) static const int16_t InsetYCoCgRound[] = {(1 << (InsetColorShift - 1)) - 1,
                                                      (1 << (InsetColorShift - 1)) - 1,
                                                      (1 << (InsetColorShift - 1)) - 1,
                                                      (1 << (InsetAlphaShift - 1)) - 1,
                                                      0,
                                                      0,
                                                      0,
                                                      0};
alignas(16) static const int16_t InsetYCoCgMask[] = {-1, -1, 0, -1, -1, -1, 0, -1};

alignas(16) static const int16_t InsetShiftUp[] = {
    1 << InsetColorShift, 1 << InsetColorShift, 1 << InsetColorShift, 1 << InsetAlphaShift, 0, 0, 0, 0};
alignas(16) static const int16_t InsetShiftDown[] = {
    1 << (16 - InsetColorShift),
    1 << (16 - InsetColorShift),
    1 << (16 - InsetColorShift),
    1 << (16 - InsetAlphaShift),
    0,
    0,
    0,
    0,
};

alignas(16) static const int16_t Inset565Mask[] = {0b11111000, 0b11111100, 0b11111000, 0xff,
                                                   0b11111000, 0b11111100, 0b11111000, 0xff};
alignas(16) static const int16_t Inset565Rep[] = {1 << (16 - 5), 1 << (16 - 6), 1 << (16 - 5), 0,
                                                  1 << (16 - 5), 1 << (16 - 6), 1 << (16 - 5), 0};

void InsetYCoCgBBox_NEON(uint8_t min_color[4], uint8_t max_color[4]) {
    int8x16_t min_col, max_col;

    min_col = vld1q_u8(min_color);
    max_col = vld1q_u8(max_color);

    min_col = vzip1q_s8(min_col, vdupq_n_u8(0));
    max_col = vzip1q_s8(max_col, vdupq_n_u8(0));

    int16x8_t inset = vsubq_s16(max_col, min_col);
    inset = vsubq_s16(inset, vld1q_s16(InsetYCoCgRound));
    inset = vandq_s32(inset, vld1q_s16(InsetYCoCgMask));

    min_col = vmulq_s16(min_col, vld1q_s16(InsetShiftUp));
    max_col = vmulq_s16(max_col, vld1q_s16(InsetShiftUp));

    min_col = vaddq_s16(min_col, inset);
    max_col = vsubq_s16(max_col, inset);

    min_col = _mm_mulhi_epi16(min_col, vld1q_s16(InsetShiftDown));
    max_col = _mm_mulhi_epi16(max_col, vld1q_s16(InsetShiftDown));

    min_col = vmaxq_s16(min_col, vdupq_n_s16(0));
    max_col = vmaxq_s16(max_col, vdupq_n_s16(0));

    min_col = vandq_s32(min_col, vld1q_s16(Inset565Mask));
    max_col = vandq_s32(max_col, vld1q_s16(Inset565Mask));

    const int8x16_t temp0 = _mm_mulhi_epi16(min_col, vld1q_s16(Inset565Rep));
    const int8x16_t temp1 = _mm_mulhi_epi16(max_col, vld1q_s16(Inset565Rep));

    min_col = vorrq_s32(min_col, temp0);
    max_col = vorrq_s32(max_col, temp1);

    min_col = vcombine_u8(vqmovun_s16(min_col), vqmovun_s16(min_col));
    max_col = vcombine_u8(vqmovun_s16(max_col), vqmovun_s16(max_col));

    vst1q_lane_s32(reinterpret_cast<int32_t *>(min_color), min_col, 0);
    vst1q_lane_s32(reinterpret_cast<int32_t *>(max_color), max_col, 0);
}

alignas(16) static const int16_t CoCgMask[] = {-1, 0, -1, 0, -1, 0, -1, 0};
alignas(16) static const int8_t CoCgDiagonalMask[] = {0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

alignas(16) static const int16x8_t Zeroes_128 = vdupq_n_s16(0);
alignas(16) static const int16x8_t Ones_i16 = vdupq_n_s16(1);
alignas(16) static const int16x8_t Twos_i16 = vdupq_n_s16(2);
alignas(16) static const int16x8_t Eights_i16 = vdupq_n_s16(8);

void SelectYCoCgDiagonal_NEON(const uint8_t block[64], uint8_t min_color[3], uint8_t max_color[3]) {
    // load block
    int8x16_t line0 = vld1q_u8(block);
    int8x16_t line1 = vld1q_u8(block + 16);
    int8x16_t line2 = vld1q_u8(block + 32);
    int8x16_t line3 = vld1q_u8(block + 48);

    // mask out everything except CoCg channels
    int8x16_t line0_CoCg = vandq_s32(line0, vld1q_s16(CoCgMask));
    int8x16_t line1_CoCg = vandq_s32(line1, vld1q_s16(CoCgMask));
    int8x16_t line2_CoCg = vandq_s32(line2, vld1q_s16(CoCgMask));
    int8x16_t line3_CoCg = vandq_s32(line3, vld1q_s16(CoCgMask));

    // merge pairs of CoCg channels
    int8x16_t line01_CoCg = vorrq_s32(line0_CoCg, vextq_s8(vdupq_n_s8(0), line1_CoCg, (16 - 2)));
    int8x16_t line23_CoCg = vorrq_s32(line2_CoCg, vextq_s8(vdupq_n_s8(0), line3_CoCg, (16 - 2)));

    int8x16_t min_col = vsetq_lane_s32(*reinterpret_cast<const int32_t *>(min_color), vdupq_n_s32(0), 0);
    int8x16_t max_col = vsetq_lane_s32(*reinterpret_cast<const int32_t *>(max_color), vdupq_n_s32(0), 0);

    int8x16_t mid = vrhaddq_u8(min_col, max_col);
    mid = vdupq_lane_s16(vget_low_s16(mid), 0);

    int8x16_t tmp1 = vmaxq_u8(mid, line01_CoCg);
    int8x16_t tmp3 = vmaxq_u8(mid, line23_CoCg);
    tmp1 = vceqq_s8(tmp1, line01_CoCg);
    tmp3 = vceqq_s8(tmp3, line23_CoCg);

    int8x16_t tmp0 = vextq_s8(tmp1, vdupq_n_s8(0), 1);
    int8x16_t tmp2 = vextq_s8(tmp3, vdupq_n_s8(0), 1);

    tmp0 = veorq_s32(tmp0, tmp1);
    tmp2 = veorq_s32(tmp2, tmp3);
    tmp0 = vandq_s32(tmp0, Ones_i16);
    tmp2 = vandq_s32(tmp2, Ones_i16);

    tmp0 = vaddq_s16(tmp0, tmp2);
    uint16x8_t t = vpaddlq_u8(vabdq_u8(tmp0, Zeroes_128));
    tmp0 = vpaddlq_u32(vpaddlq_u16(t));

    int32x2_t a32 = vget_high_s32(tmp0);
    int32x2_t a10 = vget_low_s32(tmp0);
    tmp1 = vcombine_s32(a32, a10);

    tmp0 = vaddq_s16(tmp0, tmp1);
    tmp0 = vcgtq_s16(tmp0, Eights_i16);
    tmp0 = vandq_s32(tmp0, vld1q_s8(CoCgDiagonalMask));

    min_col = veorq_s32(min_col, max_col);
    tmp0 = vandq_s32(tmp0, min_col);
    max_col = veorq_s32(max_col, tmp0);
    min_col = veorq_s32(min_col, max_col);

    vst1q_lane_s32(reinterpret_cast<int32_t *>(min_color), min_col, 0);
    vst1q_lane_s32(reinterpret_cast<int32_t *>(max_color), max_col, 0);
}

alignas(16) static const int8_t RGB565Mask[] = {
    int8_t(0b11111000), int8_t(0b11111100), int8_t(0b11111000), 0, 0, 0, 0, 0,
    int8_t(0b11111000), int8_t(0b11111100), int8_t(0b11111000), 0, 0, 0, 0, 0,
};
// multiplier used to emulate division by 3
static const int16x8_t DivBy3_i16 = vdupq_n_s16((1 << 16) / 3 + 1);

void EmitColorIndices_NEON(const uint8_t block[64], const uint8_t min_color[4], const uint8_t max_color[4],
                           uint8_t *&out_data) {
    int16x8_t result = vdupq_n_s16(0);

    // Find 4 colors on the line through min - max color
    // compute color0 (max_color)
    int16x8_t color0 = vsetq_lane_s32(*reinterpret_cast<const int32_t *>(max_color), vdupq_n_s32(0), 0);
    color0 = vandq_s32(color0, vld1q_s8(RGB565Mask));
    color0 = vzip1q_s8(color0, vdupq_n_s16(0));

    int16x8_t rb = _mm_shufflelo_epi16<_MM_SHUFFLE(3, 2, 3, 0)>(color0);
    int16x8_t g = _mm_shufflelo_epi16<_MM_SHUFFLE(3, 3, 1, 3)>(color0);

    rb = vshlq_u16(rb, vdupq_n_s16(-5));
    g = vshlq_u16(g, vdupq_n_s16(-6));
    color0 = vorrq_s32(color0, rb);
    color0 = vorrq_s32(color0, g);

    // compute color1 (min_color)
    int32x4_t color1 = vsetq_lane_s32(*reinterpret_cast<const int32_t *>(min_color), vdupq_n_s32(0), 0);
    color1 = vandq_s32(color1, vld1q_s8(RGB565Mask));
    color1 = vzip1q_s8(color1, vdupq_n_s16(0));
    rb = _mm_shufflelo_epi16<_MM_SHUFFLE(3, 2, 3, 0)>(color1);
    g = _mm_shufflelo_epi16<_MM_SHUFFLE(3, 3, 1, 3)>(color1);
    rb = vshlq_u16(rb, vdupq_n_s16(-5));
    g = vshlq_u16(g, vdupq_n_s16(-6));
    color1 = vorrq_s32(color1, rb);
    color1 = vorrq_s32(color1, g);

    // compute and pack color3
    int16x8_t color3 = vaddq_s16(color1, color1);
    color3 = vaddq_s16(color0, color3);
    color3 = _mm_mulhi_epi16(color3, DivBy3_i16);

    color3 = vcombine_u8(vqmovun_s16(color3), vqmovun_s16(vdupq_n_s16(0)));
    int32x2_t a10 = vget_low_s32(color3);
    color3 = vcombine_s32(a10, a10);

    // compute and pack color2
    int16x8_t color2 = vaddq_s16(color0, color0);
    color2 = vaddq_s16(color2, color1);
    color2 = _mm_mulhi_epi16(color2, DivBy3_i16);

    color2 = vcombine_u8(vqmovun_s16(color2), vqmovun_s16(vdupq_n_s16(0)));
    a10 = vget_low_s32(color2);
    color2 = vcombine_s32(a10, a10);

    // pack color1
    color1 = vcombine_u8(vqmovun_s16(color1), vqmovun_s16(vdupq_n_s16(0)));
    a10 = vget_low_s32(color1);
    color1 = vcombine_s32(a10, a10);

    // pack color0
    color0 = vcombine_u8(vqmovun_s16(color0), vqmovun_s16(vdupq_n_s16(0)));
    a10 = vget_low_s32(color0);
    color0 = vcombine_s32(a10, a10);

    for (int i = 32; i >= 0; i -= 32) {
        // load 4 colors
        int32x4_t color_hi = vcombine_s64(vld1_s64(reinterpret_cast<const int64_t *>(block + i + 0)), vdup_n_s64(0));
        color_hi = _mm_shuffle_epi32<_MM_SHUFFLE(3, 1, 2, 0)>(color_hi);
        int32x4_t color_lo = vcombine_s64(vld1_s64(reinterpret_cast<const int64_t *>(block + i + 8)), vdup_n_s64(0));
        color_lo = _mm_shuffle_epi32<_MM_SHUFFLE(3, 1, 2, 0)>(color_lo);

        // compute the sum of abs diff for each color
        int32x4_t d_hi = _mm_sad_epu8(color_hi, color0);
        int32x4_t d_lo = _mm_sad_epu8(color_lo, color0);
        int32x4_t d0 = _mm_packs_epi32(d_hi, d_lo);
        d_hi = _mm_sad_epu8(color_hi, color1);
        d_lo = _mm_sad_epu8(color_lo, color1);
        int32x4_t d1 = _mm_packs_epi32(d_hi, d_lo);
        d_hi = _mm_sad_epu8(color_hi, color2);
        d_lo = _mm_sad_epu8(color_lo, color2);
        int32x4_t d2 = _mm_packs_epi32(d_hi, d_lo);
        d_hi = _mm_sad_epu8(color_hi, color3);
        d_lo = _mm_sad_epu8(color_lo, color3);
        int32x4_t d3 = _mm_packs_epi32(d_hi, d_lo);

        // load next 4 colors
        color_hi = vcombine_s64(vld1_s64(reinterpret_cast<const int64_t *>(block + i + 16)), vdup_n_s64(0));
        color_hi = _mm_shuffle_epi32<_MM_SHUFFLE(3, 1, 2, 0)>(color_hi);
        color_lo = vcombine_s64(vld1_s64(reinterpret_cast<const int64_t *>(block + i + 24)), vdup_n_s64(0));
        color_lo = _mm_shuffle_epi32<_MM_SHUFFLE(3, 1, 2, 0)>(color_lo);

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
        int32x4_t b0 = vcgtq_s16(d0, d3);
        int32x4_t b1 = vcgtq_s16(d1, d2);
        int32x4_t b2 = vcgtq_s16(d0, d2);
        int32x4_t b3 = vcgtq_s16(d1, d3);
        int32x4_t b4 = vcgtq_s16(d2, d3);

        // compute color index
        int32x4_t x0 = vandq_s32(b2, b1);
        int32x4_t x1 = vandq_s32(b3, b0);
        int32x4_t x2 = vandq_s32(b4, b0);
        int32x4_t index_bit0 = vorrq_s32(x0, x1);
        index_bit0 = vandq_s32(index_bit0, Twos_i16);
        int32x4_t index_bit1 = vandq_s32(x2, Ones_i16);
        int32x4_t index = vorrq_s32(index_bit1, index_bit0);

        // pack index into result
        int32x2_t a32 = vget_high_s32(index);
        int32x2_t a10 = vget_low_s32(index);
        int32x4_t index_hi = vcombine_s32(a32, a10);
        index_hi = vzip1q_s16(index_hi, Zeroes_128);
        index_hi = vshlq_s32(index_hi, vdupq_n_s32(8));
        int32x4_t index_lo = vzip1q_s16(index, Zeroes_128);
        result = vshlq_s32(result, vdupq_n_s32(16));
        result = vorrq_s32(result, index_hi);
        result = vorrq_s32(result, index_lo);
    }

    // pack 16 2-bit color indices into a single 32-bit value
    int32x4_t result1 = _mm_shuffle_epi32<_MM_SHUFFLE(0, 3, 2, 1)>(result);
    int32x4_t result2 = _mm_shuffle_epi32<_MM_SHUFFLE(1, 0, 3, 2)>(result);
    int32x4_t result3 = _mm_shuffle_epi32<_MM_SHUFFLE(2, 1, 0, 3)>(result);
    result1 = vshlq_s32(result1, vdupq_n_s32(2));
    result2 = vshlq_s32(result2, vdupq_n_s32(4));
    result3 = vshlq_s32(result3, vdupq_n_s32(6));
    result = vorrq_s32(result, result1);
    result = vorrq_s32(result, result2);
    result = vorrq_s32(result, result3);

    vst1q_lane_s32(reinterpret_cast<int32_t *>(out_data), result, 0);
    out_data += 4;
}

// multiplier used to emulate division by 7
static const int16x8_t DivBy7_i16 = vdupq_n_s16((1 << 16) / 7 + 1);
// multiplier used to emulate division by 14
static const int16x8_t DivBy14_i16 = vdupq_n_s16((1 << 16) / 14 + 1);
static const int16_t ScaleBy_66554400_i16[] = {6, 6, 5, 5, 4, 4, 0, 0};
static const int16_t ScaleBy_11223300_i16[] = {1, 1, 2, 2, 3, 3, 0, 0};

static const int16x8_t Ones_i8 = vdupq_n_s8(1);
static const int16x8_t Twos_i8 = vdupq_n_s8(2);
static const int16x8_t Sevens_i8 = vdupq_n_s8(7);

alignas(16) static const int32_t AlphaMask0[] = {7 << 0, 0, 7 << 0, 0};
alignas(16) static const int32_t AlphaMask1[] = {7 << 3, 0, 7 << 3, 0};
alignas(16) static const int32_t AlphaMask2[] = {7 << 6, 0, 7 << 6, 0};
alignas(16) static const int32_t AlphaMask3[] = {7 << 9, 0, 7 << 9, 0};
alignas(16) static const int32_t AlphaMask4[] = {7 << 12, 0, 7 << 12, 0};
alignas(16) static const int32_t AlphaMask5[] = {7 << 15, 0, 7 << 15, 0};
alignas(16) static const int32_t AlphaMask6[] = {7 << 18, 0, 7 << 18, 0};
alignas(16) static const int32_t AlphaMask7[] = {7 << 21, 0, 7 << 21, 0};

void EmitAlphaIndicesInternal_NEON(uint8x16_t alpha, const uint8_t min_alpha, const uint8_t max_alpha,
                                   uint8_t *&out_data) {
    int16x8_t max = vdupq_n_s16(max_alpha);
    int16x8_t min = vdupq_n_s16(min_alpha);

    // compute midpoint offset between any two interpolated alpha values
    int16x8_t mid = vsubq_s16(max, min);
    mid = _mm_mulhi_epi16(mid, DivBy14_i16);

    // compute first midpoint
    int16x8_t ab1 = min;
    ab1 = vaddq_s16(ab1, mid);
    ab1 = vcombine_u8(vqmovun_s16(ab1), vqmovun_s16(ab1));

    // compute the next three midpoints
    int16x8_t max456 = vmulq_s16(max, vld1q_s16(ScaleBy_66554400_i16));
    int16x8_t min123 = vmulq_s16(min, vld1q_s16(ScaleBy_11223300_i16));
    int16x8_t ab234 = vaddq_s16(max456, min123);
    ab234 = _mm_mulhi_epi16(ab234, DivBy7_i16);
    ab234 = vaddq_s16(ab234, mid);
    int32x4_t ab2 = vdupq_lane_s32(vget_low_s32(ab234), 0);
    ab2 = vcombine_u8(vqmovun_s16(ab2), vqmovun_s16(ab2));
    int32x4_t ab3 = vdupq_lane_s32(vget_low_s32(ab234), 1);
    ab3 = vcombine_u8(vqmovun_s16(ab3), vqmovun_s16(ab3));
    int32x4_t ab4 = vdupq_lane_s32(vget_high_s32(ab234), 0);
    ab4 = vcombine_u8(vqmovun_s16(ab4), vqmovun_s16(ab4));

    // compute the last three midpoints
    int16x8_t max123 = vmulq_s16(max, vld1q_s16(ScaleBy_11223300_i16));
    int16x8_t min456 = vmulq_s16(min, vld1q_s16(ScaleBy_66554400_i16));
    int16x8_t ab567 = vaddq_s16(max123, min456);
    ab567 = _mm_mulhi_epi16(ab567, DivBy7_i16);
    ab567 = vaddq_s16(ab567, mid);
    int32x4_t ab5 = vdupq_lane_s32(vget_high_s32(ab567), 0);
    ab5 = vcombine_u8(vqmovun_s16(ab5), vqmovun_s16(ab5));
    int32x4_t ab6 = vdupq_lane_s32(vget_low_s32(ab567), 1);
    ab6 = vcombine_u8(vqmovun_s16(ab6), vqmovun_s16(ab6));
    int32x4_t ab7 = vdupq_lane_s32(vget_low_s32(ab567), 0);
    ab7 = vcombine_u8(vqmovun_s16(ab7), vqmovun_s16(ab7));

    // compare the alpha values to the midpoints
    uint8x16_t b1 = vminq_u8(ab1, alpha);
    b1 = vceqq_s8(b1, alpha);
    b1 = vandq_s32(b1, Ones_i8);
    uint8x16_t b2 = vminq_u8(ab2, alpha);
    b2 = vceqq_s8(b2, alpha);
    b2 = vandq_s32(b2, Ones_i8);
    uint8x16_t b3 = vminq_u8(ab3, alpha);
    b3 = vceqq_s8(b3, alpha);
    b3 = vandq_s32(b3, Ones_i8);
    uint8x16_t b4 = vminq_u8(ab4, alpha);
    b4 = vceqq_s8(b4, alpha);
    b4 = vandq_s32(b4, Ones_i8);
    uint8x16_t b5 = vminq_u8(ab5, alpha);
    b5 = vceqq_s8(b5, alpha);
    b5 = vandq_s32(b5, Ones_i8);
    uint8x16_t b6 = vminq_u8(ab6, alpha);
    b6 = vceqq_s8(b6, alpha);
    b6 = vandq_s32(b6, Ones_i8);
    uint8x16_t b7 = vminq_u8(ab7, alpha);
    b7 = vceqq_s8(b7, alpha);
    b7 = vandq_s32(b7, Ones_i8);

    // compute alpha indices
    uint8x16_t index = vqaddq_u8(b1, b2);
    index = vqaddq_u8(index, b3);
    index = vqaddq_u8(index, b4);
    index = vqaddq_u8(index, b5);
    index = vqaddq_u8(index, b6);
    index = vqaddq_u8(index, b7);

    // convert natural index ordering to DXT index ordering
    index = vqaddq_u8(index, Ones_i8);
    index = vandq_s32(index, Sevens_i8);
    uint8x16_t swapMinMax = vcgtq_s8(Twos_i8, index);
    swapMinMax = vandq_s32(swapMinMax, Ones_i8);
    index = veorq_s32(index, swapMinMax);

    // pack the 16 3-bit indices into 6 bytes
    int32x4_t index0 = vandq_s32(index, vld1q_s32(AlphaMask0));
    int32x4_t index1 = vshlq_u64(index, vdupq_n_s64(-(8 - 3)));
    index1 = vandq_s32(index1, vld1q_s32(AlphaMask1));
    int32x4_t index2 = vshlq_u64(index, vdupq_n_s64(-(16 - 6)));
    index2 = vandq_s32(index2, vld1q_s32(AlphaMask2));
    int32x4_t index3 = vshlq_u64(index, vdupq_n_s64(-(24 - 9)));
    index3 = vandq_s32(index3, vld1q_s32(AlphaMask3));
    int32x4_t index4 = vshlq_u64(index, vdupq_n_s64(-(32 - 12)));
    index4 = vandq_s32(index4, vld1q_s32(AlphaMask4));
    int32x4_t index5 = vshlq_u64(index, vdupq_n_s64(-(40 - 15)));
    index5 = vandq_s32(index5, vld1q_s32(AlphaMask5));
    int32x4_t index6 = vshlq_u64(index, vdupq_n_s64(-(48 - 18)));
    index6 = vandq_s32(index6, vld1q_s32(AlphaMask6));
    int32x4_t index7 = vshlq_u64(index, vdupq_n_s64(-(56 - 21)));
    index7 = vandq_s32(index7, vld1q_s32(AlphaMask7));
    index = vorrq_s32(index0, index1);
    index = vorrq_s32(index, index2);
    index = vorrq_s32(index, index3);
    index = vorrq_s32(index, index4);
    index = vorrq_s32(index, index5);
    index = vorrq_s32(index, index6);
    index = vorrq_s32(index, index7);

    vst1q_lane_s32(reinterpret_cast<int32_t *>(out_data), index, 0);
    vst1q_lane_s32(reinterpret_cast<int32_t *>(out_data + 3), index, 2);

    out_data += 6;
}

void EmitAlphaIndices_NEON(const uint8_t block[64], const uint8_t min_alpha, const uint8_t max_alpha,
                           uint8_t *&out_data) {
    int32x4_t line0 = vld1q_u8(block);
    int32x4_t line1 = vld1q_u8(block + 16);
    int32x4_t line2 = vld1q_u8(block + 32);
    int32x4_t line3 = vld1q_u8(block + 48);

    line0 = vshlq_u32(line0, vdupq_n_s32(-24));
    line1 = vshlq_u32(line1, vdupq_n_s32(-24));
    line2 = vshlq_u32(line2, vdupq_n_s32(-24));
    line3 = vshlq_u32(line3, vdupq_n_s32(-24));

    int32x4_t line01 = vcombine_u8(vqmovun_s16(line0), vqmovun_s16(line1));
    int32x4_t line23 = vcombine_u8(vqmovun_s16(line2), vqmovun_s16(line3));

    // pack all 16 alpha values
    int32x4_t alpha = vcombine_u8(vqmovun_s16(line01), vqmovun_s16(line23));

    EmitAlphaIndicesInternal_NEON(alpha, min_alpha, max_alpha, out_data);
}

void EmitAlphaOnlyIndices_NEON(const uint8_t block[16], const uint8_t min_alpha, const uint8_t max_alpha,
                               uint8_t *&out_data) {
    int32x4_t alpha = vld1q_s32(reinterpret_cast<const int32_t *>(block));

    EmitAlphaIndicesInternal_NEON(alpha, min_alpha, max_alpha, out_data);
}

} // namespace Ray

#undef _ABS

#endif // defined(__ARM_NEON__) || defined(__arm__) || defined(__aarch64__) || defined(_M_ARM) || defined(_M_ARM64)