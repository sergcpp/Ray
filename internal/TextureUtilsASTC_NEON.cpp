// NEON block-emitter leaves for the ASTC 4x4 fixed-layout encoder (see TextureUtilsASTC.cpp).
// Byte-exact parity with the scalar reference leaves: int32-exact NEON math, first-min tie
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

void ASTCSelectWeights_NEON(const uint8_t pixels[64], const uint8_t dec[4][8], const int nch, uint8_t w[16]) {
    int16x8_t d8[4]; // dec levels 0..7, i16
    for (int ch = 0; ch < nch; ch++)
        d8[ch] = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(&dec[ch][0])));
    for (int t = 0; t < 16; t++) {
        int32x4_t alo = vdupq_n_s32(0), ahi = alo; // per-level SSE (i32) for levels 0..3 / 4..7
        for (int ch = 0; ch < nch; ch++) {
            const int16x8_t pb = vdupq_n_s16(int16_t(pixels[t * 4 + ch]));
            const int16x8_t d = vsubq_s16(d8[ch], pb);
            const int32x4_t lo = vmull_s16(vget_low_s16(d), vget_low_s16(d));
            const int32x4_t hi = vmull_s16(vget_high_s16(d), vget_high_s16(d));
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

// Dual-plane YCoCg leaves (mirroring the BC ScaleYCoCg/SelectYCoCgDiagonal pair in TextureUtilsNEON.cpp).

// Chroma scale (decoder's byte/8 + 1, sb = scale*8): exact i16 math, (v-128)*sb is |p|<=4096 and
// (p + 4 - (p<0)) >> 3 is lround(p/8) - round half away from zero, matching scalar scale_cocg().
void ASTCScaleCoCg_NEON(const uint8_t pixels[64], const int sb, uint8_t out[64]) {
    const int16x8_t c128 = vdupq_n_s16(128), sbv = vdupq_n_s16(int16_t(sb));
    const int16x8_t z = vdupq_n_s16(0), c255 = vdupq_n_s16(255);
    alignas(16) static const int16_t cocg_lanes[8] = {-1, -1, 0, 0, -1, -1, 0, 0}; // unpack lanes = Co,Cg,B,Y
    const int16x8_t sel = vld1q_s16(cocg_lanes);
    for (int i = 0; i < 4; i++) {
        const uint8x16_t row = vld1q_u8(&pixels[i * 16]);
        const int16x8_t halves[2] = {vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(row))),
                                     vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(row)))};
        int16x8_t scaled[2];
        for (int h = 0; h < 2; h++) {
            const int16x8_t p = vmulq_s16(vsubq_s16(halves[h], c128), sbv);
            const int16x8_t b = vaddq_s16(vshrq_n_s16(p, 15), vdupq_n_s16(4));
            const int16x8_t r = vaddq_s16(c128, vshrq_n_s16(vaddq_s16(p, b), 3));
            scaled[h] = vbslq_s16(sel, vminq_s16(vmaxq_s16(r, z), c255), halves[h]);
        }
        vst1q_u8(&out[i * 16],
                 vcombine_u8(vqmovun_s16(scaled[0]), vqmovun_s16(scaled[1]))); // values pre-clamped: exact narrow
    }
}

// Per-texel (Co >= mid0) XOR (Cg >= mid1) count over the scaled block; the Cg-lane mask result is
// shifted one lane down to XOR into the Co lane, kept bits then summed by the final lane read.
int ASTCDiagonalSide_NEON(const uint8_t scaled[64], const int mid0, const int mid1) {
    const int16x8_t z = vdupq_n_s16(0);
    const int16x8_t m0 = vdupq_n_s16(int16_t(mid0)), m1 = vdupq_n_s16(int16_t(mid1));
    alignas(16) static const int16_t lane_co[8] = {-1, 0, 0, 0, -1, 0, 0, 0}; // lanes 0, 4
    alignas(16) static const int16_t lane_cg[8] = {0, -1, 0, 0, 0, -1, 0, 0}; // lanes 1, 5
    alignas(16) static const int16_t bit_co[8] = {1, 0, 0, 0, 1, 0, 0, 0};    // bit0 of the Co lanes
    int16x8_t acc = z;
    for (int i = 0; i < 4; i++) {
        const uint8x16_t row = vld1q_u8(&scaled[i * 16]);
        const int16x8_t halves[2] = {vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(row))),
                                     vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(row)))};
        for (int h = 0; h < 2; h++) {
            const int16x8_t t0 = vandq_s16(vreinterpretq_s16_u16(vcgeq_s16(halves[h], m0)), vld1q_s16(lane_co));
            const int16x8_t t1 = vandq_s16(vreinterpretq_s16_u16(vcgeq_s16(halves[h], m1)), vld1q_s16(lane_cg));
            const int16x8_t xr = veorq_s16(t0, vextq_s16(t1, z, 1)); // Cg-lane result moves down to the Co lane
            acc = vaddq_s16(acc, vandq_s16(xr, vld1q_s16(bit_co)));
        }
    }
    return int(vgetq_lane_s16(acc, 0)) + int(vgetq_lane_s16(acc, 4));
}

// Plane 0 picks the nearest of the 4 decoded (d0,d1) chroma levels, plane 1 the nearest Y level;
// squared diffs are exact in i16, sums in i32, first-min ties resolved by the level scan.
uint64_t ASTCSelectWeightsDual_NEON(const uint8_t pixels[64], const uint8_t scaled[64], const uint8_t d0[4],
                                    const uint8_t d1[4], const uint8_t dy[4]) {
    alignas(16) uint8_t l08[8] = {}, ly8[8] = {};
    memcpy(&l08[0], d0, 4);
    memcpy(&l08[4], d1, 4);
    memcpy(&ly8[0], dy, 4);
    const int16x8_t d01 = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(&l08[0]))); // d0[0..3], d1[0..3]
    const int16x8_t dyv = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(&ly8[0]))); // dy[0..3]
    uint64_t wbits = 0;
    for (int t = 0; t < 16; t++) {
        const int co = scaled[t * 4 + 0], cg = scaled[t * 4 + 1], y = pixels[t * 4 + 3];
        const int16x8_t pix = vcombine_s16(vdup_n_s16(int16_t(co)), vdup_n_s16(int16_t(cg)));
        const int16x8_t dif = vsubq_s16(d01, pix);
        const int32x4_t e0v = vaddq_s32(vmull_s16(vget_low_s16(dif), vget_low_s16(dif)),
                                        vmull_s16(vget_high_s16(dif), vget_high_s16(dif)));
        const int16x4_t dyd = vsub_s16(vget_low_s16(dyv), vdup_n_s16(int16_t(y)));
        const int32x4_t e1v = vmull_s16(dyd, dyd);
        int32_t e0[4], e1[4];
        vst1q_s32(&e0[0], e0v);
        vst1q_s32(&e1[0], e1v);
        int b0 = 0, b1 = 0;
        for (int v = 1; v < 4; v++) {
            if (e0[v] < e0[b0])
                b0 = v;
            if (e1[v] < e1[b1])
                b1 = v;
        }
        wbits |= uint64_t(b0 | (b1 << 2)) << (4 * t);
    }
    return wbits;
}

} // namespace Ray
#endif
