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

void ASTCSelectWeights_SSE2(const uint8_t pixels[64], const uint8_t dec[4][8], const int nch, uint8_t w[16],
                            const unsigned yshift) {
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
            __m128i sq = _mm_mullo_epi16(dl, dl);     // d^2 <= 65025: exact mod 2^16, sign-agnostic
            __m128i lo32 = _mm_unpacklo_epi16(sq, z); // levels 0..3
            __m128i hi32 = _mm_unpackhi_epi16(sq, z); // levels 4..7
            if (ch == 3) {                            // YCoCg luma weighting (power of 2, exact doubling adds)
                for (unsigned s = 0; s < yshift; s++) {
                    lo32 = _mm_add_epi32(lo32, lo32);
                    hi32 = _mm_add_epi32(hi32, hi32);
                }
            }
            alo = _mm_add_epi32(alo, lo32);
            ahi = _mm_add_epi32(ahi, hi32);
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

} // namespace Ray
#endif
