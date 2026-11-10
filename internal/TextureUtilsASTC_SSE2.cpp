// SSE2 block-emitter leaves for the ASTC 4x4 fixed-layout encoder (see TextureUtilsASTC.cpp).
// Byte-exact parity with the scalar reference leaves: int32-exact SSE math, first-min tie
// (lowest level index) via the level scan after the vector sum.
#if defined(_M_IX86) || defined(_M_X64) || defined(__i386__) || defined(__x86_64__)
#include <cstdint>
#include <cstring>

#include <immintrin.h>

namespace Ray {

bool ASTCBlockIsFlat_SSE2(const uint8_t pixels[64]) {
    const __m128i r0 = _mm_loadu_si128((const __m128i *)&pixels[0]);
    const __m128i splat = _mm_shuffle_epi32(r0, _MM_SHUFFLE(0, 0, 0, 0));
    __m128i eq = _mm_cmpeq_epi32(r0, splat);
    for (int i = 1; i < 4; i++)
        eq = _mm_and_si128(eq, _mm_cmpeq_epi32(_mm_loadu_si128((const __m128i *)&pixels[i * 16]), splat));
    return _mm_movemask_epi8(eq) == 0xFFFF;
}

void ASTCGetMinMax_SSE2(const uint8_t pixels[64], uint8_t lo[4], uint8_t hi[4]) {
    __m128i mn = _mm_loadu_si128((const __m128i *)&pixels[0]);
    __m128i mx = mn;
    for (int i = 1; i < 4; i++) {
        const __m128i r = _mm_loadu_si128((const __m128i *)&pixels[i * 16]);
        mn = _mm_min_epu8(mn, r);
        mx = _mm_max_epu8(mx, r);
    }
    // Fold the 4 texel columns; 4/8-byte shifts keep byte classes = channels.
    mn = _mm_min_epu8(mn, _mm_srli_si128(mn, 8));
    mn = _mm_min_epu8(mn, _mm_srli_si128(mn, 4));
    mx = _mm_max_epu8(mx, _mm_srli_si128(mx, 8));
    mx = _mm_max_epu8(mx, _mm_srli_si128(mx, 4));
    const int vmin = _mm_cvtsi128_si32(mn), vmax = _mm_cvtsi128_si32(mx); // LE: bytes 0..3 = RGBA
    memcpy(lo, &vmin, 4);
    memcpy(hi, &vmax, 4);
}

void ASTCSelectWeights_SSE2(const uint8_t pixels[64], const uint8_t dec[4][8], const int nch, uint8_t w[16]) {
    alignas(16) uint8_t dtab[4][8];
    memcpy(dtab, dec, sizeof(dtab));
    const __m128i z = _mm_setzero_si128();
    __m128i d8[4]; // dec levels 0..7, i16 (punpcklbw covers all 8 source bytes)
    for (int ch = 0; ch < nch; ch++) {
        const __m128i d = _mm_loadl_epi64((const __m128i *)&dtab[ch][0]);
        d8[ch] = _mm_unpacklo_epi8(d, z);
    }
    for (int t = 0; t < 16; t++) {
        __m128i alo = z, ahi = z; // per-level SSE (i32) for levels 0..3 / 4..7
        for (int ch = 0; ch < nch; ch++) {
            const __m128i pb = _mm_set1_epi16(int16_t(pixels[t * 4 + ch]));
            const __m128i dl = _mm_sub_epi16(d8[ch], pb);
            const __m128i sq = _mm_mullo_epi16(dl, dl);          // d^2 <= 65025: exact mod 2^16, sign-agnostic
            alo = _mm_add_epi32(alo, _mm_unpacklo_epi16(sq, z)); // levels 0..3
            ahi = _mm_add_epi32(ahi, _mm_unpackhi_epi16(sq, z)); // levels 4..7
        }
        alignas(16) int e[8];
        _mm_store_si128((__m128i *)&e[0], alo);
        _mm_store_si128((__m128i *)&e[4], ahi);
        int best = 0;
        for (int v = 1; v < 8; v++)
            if (e[v] < e[best])
                best = v;
        w[t] = uint8_t(best);
    }
}

// Dual-plane YCoCg leaves (mirroring the BC ScaleYCoCg/SelectYCoCgDiagonal pair in TextureUtilsSSE2.cpp).

// Chroma scale (decoder's byte/8 + 1, sb = scale*8): exact i16 math, (v-128)*sb is |p|<=4096 and
// (p + 4 - (p<0)) >> 3 is lround(p/8) - round half away from zero, matching scalar scale_cocg().
void ASTCScaleCoCg_SSE2(const uint8_t pixels[64], const int sb, uint8_t out[64]) {
    const __m128i z = _mm_setzero_si128();
    const __m128i c128 = _mm_set1_epi16(128), sbv = _mm_set1_epi16(int16_t(sb)), c255 = _mm_set1_epi16(255);
    // unpack lanes are Co,Cg,B,Y; keep scaled values only on the Co/Cg lanes, B/Y pass through.
    const __m128i sel = _mm_set_epi16(0, 0, -1, -1, 0, 0, -1, -1);
    const __m128i nsel = _mm_xor_si128(sel, _mm_set1_epi16(-1));
    for (int i = 0; i < 4; i++) {
        const __m128i row = _mm_loadu_si128((const __m128i *)&pixels[i * 16]);
        const __m128i halves[2] = {_mm_unpacklo_epi8(row, z), _mm_unpackhi_epi8(row, z)};
        __m128i scaled[2];
        for (int h = 0; h < 2; h++) {
            const __m128i p = _mm_mullo_epi16(_mm_sub_epi16(halves[h], c128), sbv);
            const __m128i b = _mm_add_epi16(_mm_set1_epi16(4), _mm_srai_epi16(p, 15));
            const __m128i r = _mm_add_epi16(c128, _mm_srai_epi16(_mm_add_epi16(p, b), 3));
            scaled[h] = _mm_or_si128(_mm_and_si128(_mm_min_epi16(_mm_max_epi16(r, z), c255), sel),
                                     _mm_and_si128(halves[h], nsel));
        }
        _mm_storeu_si128((__m128i *)&out[i * 16], _mm_packus_epi16(scaled[0], scaled[1]));
    }
}

// Per-texel (Co >= mid0) XOR (Cg >= mid1) count over the scaled block; the Cg-lane mask result is
// shifted one lane down (srli) to XOR into the Co lane, kept bits then summed by the final scan.
int ASTCDiagonalSide_SSE2(const uint8_t scaled[64], const int mid0, const int mid1) {
    const __m128i z = _mm_setzero_si128();
    const __m128i m0 = _mm_set1_epi16(int16_t(mid0 - 1)), m1 = _mm_set1_epi16(int16_t(mid1 - 1));
    const __m128i lane_co = _mm_set_epi16(0, 0, 0, -1, 0, 0, 0, -1); // lanes 0, 4
    const __m128i lane_cg = _mm_set_epi16(0, 0, -1, 0, 0, 0, -1, 0); // lanes 1, 5
    const __m128i bit_co = _mm_set_epi16(0, 0, 0, 1, 0, 0, 0, 1);    // bit0 of the Co lanes
    __m128i acc = z;
    for (int i = 0; i < 4; i++) {
        const __m128i row = _mm_loadu_si128((const __m128i *)&scaled[i * 16]);
        const __m128i halves[2] = {_mm_unpacklo_epi8(row, z), _mm_unpackhi_epi8(row, z)};
        for (int h = 0; h < 2; h++) {
            const __m128i t0 = _mm_and_si128(_mm_cmpgt_epi16(halves[h], m0), lane_co);
            const __m128i t1 = _mm_and_si128(_mm_cmpgt_epi16(halves[h], m1), lane_cg);
            acc = _mm_add_epi16(acc, _mm_and_si128(_mm_xor_si128(t0, _mm_srli_si128(t1, 2)), bit_co));
        }
    }
    alignas(16) int16_t s[8];
    _mm_store_si128((__m128i *)&s[0], acc);
    return int(s[0]) + int(s[4]);
}

// Plane 0 picks the nearest of the 4 decoded (d0,d1) chroma levels, plane 1 the nearest Y level;
// squared diffs are exact in i16, sums in i32, first-min ties resolved by the level scan.
uint64_t ASTCSelectWeightsDual_SSE2(const uint8_t pixels[64], const uint8_t scaled[64], const uint8_t d0[4],
                                    const uint8_t d1[4], const uint8_t dy[4]) {
    alignas(16) uint8_t l08[8] = {}, ly8[8] = {};
    memcpy(&l08[0], d0, 4);
    memcpy(&l08[4], d1, 4);
    memcpy(&ly8[0], dy, 4);
    const __m128i z = _mm_setzero_si128();
    const __m128i d01 = _mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i *)&l08[0]), z);
    const __m128i dyv = _mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i *)&ly8[0]), z);
    const __m128i lo64 = _mm_set_epi64x(0, ~int64_t(0)), hi64 = _mm_set_epi64x(~int64_t(0), 0);
    uint64_t wbits = 0;
    for (int t = 0; t < 16; t++) {
        const int co = scaled[t * 4 + 0], cg = scaled[t * 4 + 1], y = pixels[t * 4 + 3];
        const __m128i pix = _mm_or_si128(_mm_and_si128(_mm_set1_epi32(co * 0x00010001), lo64),
                                         _mm_and_si128(_mm_set1_epi32(cg * 0x00010001), hi64));
        const __m128i dif = _mm_sub_epi16(d01, pix);
        const __m128i sq = _mm_mullo_epi16(dif, dif);
        const __m128i e0v = _mm_add_epi32(_mm_unpacklo_epi16(sq, z), _mm_unpackhi_epi16(sq, z));
        const __m128i dyd = _mm_sub_epi16(dyv, _mm_set1_epi16(int16_t(y)));
        const __m128i e1v = _mm_unpacklo_epi16(_mm_mullo_epi16(dyd, dyd), z); // garbage top lanes, unpacklo only
        alignas(16) int e0[4], e1[4];
        _mm_store_si128((__m128i *)&e0[0], e0v);
        _mm_store_si128((__m128i *)&e1[0], e1v);
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
