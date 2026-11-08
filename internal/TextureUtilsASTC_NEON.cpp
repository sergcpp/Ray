// NEON block-emitter leaves for the ASTC 4x4 fixed-layout encoder (see TextureUtilsASTC.cpp).
// Byte-exact parity with the scalar reference leaves: int32-exact SSE math, first-min tie
// (lowest level index) via the level scan after the vector sum.
#if defined(__ARM_NEON__) || defined(__arm__) || defined(__aarch64__) || defined(_M_ARM) || defined(_M_ARM64)
#include <cstdint>
#include <cstring>

#include <arm_neon.h>

namespace Ray {

bool ASTCBlockIsFlat_NEON(const uint8_t pixels[64]) {
    const uint8x16_t r0 = vld1q_u8(pixels);
    const uint32x2_t dws = vreinterpret_u32_u8(vget_low_u8(r0));
    const uint32x4_t splat = vcombine_u32(vdup_lane_u32(dws, 0), vdup_lane_u32(dws, 0));
    uint32x4_t eq = vceqq_u32(vreinterpretq_u32_u8(r0), splat);
    for (int i = 1; i < 4; i++)
        eq = vandq_u32(eq, vceqq_u32(vreinterpretq_u32_u8(vld1q_u8(&pixels[i * 16])), splat));
    // vpmin (not vpmax): the texel0-vs-itself lane is always all-ones, only a min reduce
    // can detect a differing texel anywhere in the block.
    const uint32x2_t p2 = vpmin_u32(vget_low_u32(eq), vget_high_u32(eq));
    return vget_lane_u32(vpmin_u32(p2, p2), 0) == 0xFFFFFFFFu;
}

void ASTCGetMinMax_NEON(const uint8_t pixels[64], uint8_t lo[4], uint8_t hi[4]) {
    uint8x16_t mn = vld1q_u8(pixels), mx = mn;
    for (int i = 1; i < 4; i++) {
        const uint8x16_t r = vld1q_u8(&pixels[i * 16]);
        mn = vminq_u8(mn, r);
        mx = vmaxq_u8(mx, r);
    }
    // Fold the 4 texel columns; 4/8-byte rotations keep byte classes = channels.
    mn = vminq_u8(mn, vextq_u8(mn, mn, 8));
    mn = vminq_u8(mn, vextq_u8(mn, mn, 4));
    mx = vmaxq_u8(mx, vextq_u8(mx, mx, 8));
    mx = vmaxq_u8(mx, vextq_u8(mx, mx, 4));
    const uint32_t vmin = vget_lane_u32(vreinterpret_u32_u8(vget_low_u8(mn)), 0);
    const uint32_t vmax = vget_lane_u32(vreinterpret_u32_u8(vget_low_u8(mx)), 0);
    memcpy(lo, &vmin, 4);
    memcpy(hi, &vmax, 4);
}

void ASTCSelectWeights_NEON(const uint8_t pixels[64], const uint8_t dec[4][8], const int nch, uint8_t w[16],
                            const unsigned yshift) {
    int16x8_t d8[4]; // dec levels 0..7, i16
    for (int ch = 0; ch < nch; ch++)
        d8[ch] = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(&dec[ch][0])));
    for (int t = 0; t < 16; t++) {
        int32x4_t alo = vdupq_n_s32(0), ahi = alo; // per-level SSE (i32) for levels 0..3 / 4..7
        for (int ch = 0; ch < nch; ch++) {
            const int16x8_t pb = vdupq_n_s16(int16_t(pixels[t * 4 + ch]));
            const int16x8_t d = vsubq_s16(d8[ch], pb);
            int32x4_t lo = vmull_s16(vget_low_s16(d), vget_low_s16(d));
            int32x4_t hi = vmull_s16(vget_high_s16(d), vget_high_s16(d));
            if (ch == 3) { // YCoCg luma weighting (power of 2, exact doubling adds)
                for (unsigned s = 0; s < yshift; s++) {
                    lo = vaddq_s32(lo, lo);
                    hi = vaddq_s32(hi, hi);
                }
            }
            alo = vaddq_s32(alo, lo);
            ahi = vaddq_s32(ahi, hi);
        }
        int32_t e[8];
        vst1q_s32(&e[0], alo);
        vst1q_s32(&e[4], ahi);
        int best = 0;
        for (int v = 1; v < 8; v++)
            if (e[v] < e[best])
                best = v;
        w[t] = uint8_t(best);
    }
}

} // namespace Ray
#endif
