// #pragma once

#include "fp16.h"
#include "simd_avx.h"

#include <immintrin.h>

#define _mm512_cmp_ps(a, b, c) _mm512_castsi512_ps(_mm512_movm_epi32(_mm512_cmp_ps_mask(a, b, c)))

#define _mm512_blendv_ps(a, b, m)                                                                                      \
    _mm512_castsi512_ps(_mm512_ternarylogic_epi32(_mm512_castps_si512(a), _mm512_castps_si512(b),                      \
                                                  _mm512_srai_epi32(_mm512_castps_si512(m), 31), 0xd8))

#define _mm512_movemask_epi32(a)                                                                                       \
    int(_mm512_cmpneq_epi32_mask(_mm512_setzero_si512(), _mm512_and_si512(_mm512_set1_epi32(0x80000000U), a)))
#define _mm512_movemask_epi8(a)                                                                                        \
    int(_mm512_cmpneq_epi8_mask(_mm512_setzero_si512(), _mm512_and_si512(a, _mm512_set1_epi8((char)0x80))))

// https://adms-conf.org/2020-camera-ready/ADMS20_05.pdf
#define _mm512_slli_si512(x, k) _mm512_alignr_epi32(x, _mm512_setzero_si512(), 16 - k)

#ifndef NDEBUG
#define validate_mask(m) __assert_valid_mask(m)
#else
#define validate_mask(m) ((void)m)
#endif

#pragma warning(push)
#pragma warning(disable : 4752)

namespace Ray {
namespace NS {

template <> force_inline __m512 _mm_cast(__m512i x) { return _mm512_castsi512_ps(x); }
template <> force_inline __m512i _mm_cast(__m512 x) { return _mm512_castps_si512(x); }

template <> class fixed_size_simd<int, 16>;
template <> class fixed_size_simd<unsigned, 16>;
#ifdef USE_AVX512_FP16
template <> class fixed_size_simd<float16, 8>;
template <> class fixed_size_simd<float16, 16>;
template <> class fixed_size_simd<float16, 32>;
#endif

template <> class fixed_size_simd<float, 16> {
    union {
        __m512 vec_;
        float comp_[16];
    };

    friend class fixed_size_simd<int, 16>;
    friend class fixed_size_simd<unsigned, 16>;
#ifdef USE_AVX512_FP16
    friend class fixed_size_simd<float16, 32>;
#endif

    force_inline fixed_size_simd(const __m512 vec) : vec_(vec) {}

  public:
    force_inline fixed_size_simd() = default;
    force_inline fixed_size_simd(const float f) { vec_ = _mm512_set1_ps(f); }
    force_inline fixed_size_simd(const float f0, const float f1, const float f2, const float f3, const float f4,
                                 const float f5, const float f6, const float f7, const float f8, const float f9,
                                 const float f10, const float f11, const float f12, const float f13, const float f14,
                                 const float f15) {
        vec_ = _mm512_setr_ps(f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15);
    }
    force_inline explicit fixed_size_simd(const float *f) { vec_ = _mm512_loadu_ps(f); }
    force_inline fixed_size_simd(const float *f, vector_aligned_tag) { vec_ = _mm512_load_ps(f); }
    // F16C half-float load/store (nearest-even) for fp16-packed UNet tensors - this class only
    // exists under USE_AVX512, which implies F16C (see detect.cpp, AVX2 gate)
    static force_inline fixed_size_simd vectorcall load_f16(const unsigned short *p) {
        return fixed_size_simd{_mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)p))};
    }
    force_inline void vectorcall store_f16(unsigned short *p) const {
        _mm256_storeu_si256((__m256i *)p, _mm512_cvtps_ph(vec_, _MM_FROUND_TO_NEAREST_INT));
    }

    force_inline float operator[](const int i) const {
        __m512 temp = _mm512_maskz_compress_ps(__mmask16(1u << i), vec_);
        return _mm512_cvtss_f32(temp);
    }

    force_inline float operator[](const long i) const { return operator[](int(i)); }

    template <int i> force_inline float get() const {
        __m128 temp = _mm512_extractf32x4_ps(vec_, (i & 15) / 4);
        const int ndx = (i & 15) % 4;
        return _mm_cvtss_f32(_mm_shuffle_ps(temp, temp, _MM_SHUFFLE(ndx, ndx, ndx, ndx)));
    }
    template <int i> force_inline void set(const float v) {
        // TODO: find more optimal implementation (with compile-time index)
        vec_ = _mm512_mask_broadcastss_ps(vec_, __mmask16(1u << (i & 15)), _mm_set_ss(v));
    }
    force_inline void set(const int i, const float v) {
        vec_ = _mm512_mask_broadcastss_ps(vec_, __mmask16(1u << i), _mm_set_ss(v));
    }

    force_inline fixed_size_simd<float, 16> &vectorcall operator+=(const fixed_size_simd<float, 16> rhs) {
        vec_ = _mm512_add_ps(vec_, rhs.vec_);
        return *this;
    }

    force_inline fixed_size_simd<float, 16> &vectorcall operator-=(const fixed_size_simd<float, 16> rhs) {
        vec_ = _mm512_sub_ps(vec_, rhs.vec_);
        return *this;
    }

    force_inline fixed_size_simd<float, 16> &vectorcall operator*=(const fixed_size_simd<float, 16> rhs) {
        vec_ = _mm512_mul_ps(vec_, rhs.vec_);
        return *this;
    }

    force_inline fixed_size_simd<float, 16> &vectorcall operator/=(const fixed_size_simd<float, 16> rhs) {
        vec_ = _mm512_div_ps(vec_, rhs.vec_);
        return *this;
    }

    force_inline fixed_size_simd<float, 16> &vectorcall operator|=(const fixed_size_simd<float, 16> rhs) {
        vec_ = _mm512_or_ps(vec_, rhs.vec_);
        return *this;
    }

    force_inline fixed_size_simd<float, 16> &vectorcall operator&=(const fixed_size_simd<float, 16> rhs) {
        vec_ = _mm512_and_ps(vec_, rhs.vec_);
        return *this;
    }

    force_inline fixed_size_simd<float, 16> operator~() const;
    force_inline fixed_size_simd<float, 16> operator-() const;
    force_inline explicit operator fixed_size_simd<int, 16>() const;
    force_inline explicit operator fixed_size_simd<unsigned, 16>() const;

    force_inline fixed_size_simd<float, 16> sqrt() const;
    force_inline fixed_size_simd<float, 16> log() const;

    force_inline float length() const { return sqrtf(length2()); }

    float length2() const {
        const __m512 squared = _mm512_mul_ps(vec_, vec_);
        return _mm512_reduce_add_ps(squared);
    }

    force_inline float hsum() const { return _mm512_reduce_add_ps(vec_); }

    force_inline void store_to(float *f) const { _mm512_storeu_ps(f, vec_); }
    force_inline void store_to(float *f, vector_aligned_tag) const { _mm512_store_ps(f, vec_); }

    force_inline void vectorcall blend_to(const fixed_size_simd<float, 16> mask, const fixed_size_simd<float, 16> v1) {
        validate_mask(mask);
        vec_ = _mm512_blendv_ps(vec_, v1.vec_, mask.vec_);
    }

    force_inline void vectorcall blend_inv_to(const fixed_size_simd<float, 16> mask,
                                              const fixed_size_simd<float, 16> v1) {
        validate_mask(mask);
        vec_ = _mm512_blendv_ps(v1.vec_, vec_, mask.vec_);
    }

    friend force_inline fixed_size_simd<float, 16> vectorcall min(fixed_size_simd<float, 16> v1,
                                                                  fixed_size_simd<float, 16> v2);
    friend force_inline fixed_size_simd<float, 16> vectorcall max(fixed_size_simd<float, 16> v1,
                                                                  fixed_size_simd<float, 16> v2);

    friend force_inline fixed_size_simd<float, 16> vectorcall and_not(fixed_size_simd<float, 16> v1,
                                                                      fixed_size_simd<float, 16> v2);
    friend force_inline fixed_size_simd<float, 16> vectorcall floor(fixed_size_simd<float, 16> v1);
    friend force_inline fixed_size_simd<float, 16> vectorcall ceil(fixed_size_simd<float, 16> v1);

    friend force_inline fixed_size_simd<float, 16> vectorcall operator&(fixed_size_simd<float, 16> v1,
                                                                        fixed_size_simd<float, 16> v2);
    friend force_inline fixed_size_simd<float, 16> vectorcall operator|(fixed_size_simd<float, 16> v1,
                                                                        fixed_size_simd<float, 16> v2);
    friend force_inline fixed_size_simd<float, 16> vectorcall operator^(fixed_size_simd<float, 16> v1,
                                                                        fixed_size_simd<float, 16> v2);
    friend force_inline fixed_size_simd<float, 16> vectorcall operator+(fixed_size_simd<float, 16> v1,
                                                                        fixed_size_simd<float, 16> v2);
    friend force_inline fixed_size_simd<float, 16> vectorcall operator-(fixed_size_simd<float, 16> v1,
                                                                        fixed_size_simd<float, 16> v2);
    friend force_inline fixed_size_simd<float, 16> vectorcall operator*(fixed_size_simd<float, 16> v1,
                                                                        fixed_size_simd<float, 16> v2);
    friend force_inline fixed_size_simd<float, 16> vectorcall operator/(fixed_size_simd<float, 16> v1,
                                                                        fixed_size_simd<float, 16> v2);

    friend force_inline fixed_size_simd<float, 16> vectorcall operator<(fixed_size_simd<float, 16> v1,
                                                                        fixed_size_simd<float, 16> v2);
    friend force_inline fixed_size_simd<float, 16> vectorcall operator<=(fixed_size_simd<float, 16> v1,
                                                                         fixed_size_simd<float, 16> v2);
    friend force_inline fixed_size_simd<float, 16> vectorcall operator>(fixed_size_simd<float, 16> v1,
                                                                        fixed_size_simd<float, 16> v2);
    friend force_inline fixed_size_simd<float, 16> vectorcall operator>=(fixed_size_simd<float, 16> v1,
                                                                         fixed_size_simd<float, 16> v2);
    friend force_inline fixed_size_simd<float, 16> vectorcall operator==(fixed_size_simd<float, 16> v1,
                                                                         fixed_size_simd<float, 16> v2);
    friend force_inline fixed_size_simd<float, 16> vectorcall operator!=(fixed_size_simd<float, 16> v1,
                                                                         fixed_size_simd<float, 16> v2);

    friend force_inline fixed_size_simd<float, 16>
        vectorcall clamp(fixed_size_simd<float, 16> v1, fixed_size_simd<float, 16> min, fixed_size_simd<float, 16> max);
    // friend force_inline fixed_size_simd<float, 16> vectorcall clamp(fixed_size_simd<float, 16> v1, float min, float
    // max);
    friend force_inline fixed_size_simd<float, 16> vectorcall saturate(const fixed_size_simd<float, 16> v1) {
        return clamp(v1, 0.0f, 1.0f);
    }
    friend force_inline fixed_size_simd<float, 16> vectorcall pow(fixed_size_simd<float, 16> v1,
                                                                  fixed_size_simd<float, 16> v2);
    friend force_inline fixed_size_simd<float, 16> vectorcall exp(fixed_size_simd<float, 16> v1);
    friend force_inline fixed_size_simd<float, 16> vectorcall normalize(fixed_size_simd<float, 16> v1);
    friend force_inline fixed_size_simd<float, 16> vectorcall normalize_len(fixed_size_simd<float, 16> v1,
                                                                            float &out_len);
    friend force_inline fixed_size_simd<float, 16> vectorcall inclusive_scan(fixed_size_simd<float, 16> v1);
    friend force_inline fixed_size_simd<float, 16> vectorcall copysign(const fixed_size_simd<float, 16> val,
                                                                       const fixed_size_simd<float, 16> sign);

    friend force_inline fixed_size_simd<float, 16>
        vectorcall fmadd(fixed_size_simd<float, 16> a, fixed_size_simd<float, 16> b, fixed_size_simd<float, 16> c);
    friend force_inline fixed_size_simd<float, 16>
        vectorcall fmsub(fixed_size_simd<float, 16> a, fixed_size_simd<float, 16> b, fixed_size_simd<float, 16> c);

    friend force_inline fixed_size_simd<float, 16> vectorcall gather(const float *base_addr,
                                                                     fixed_size_simd<int, 16> vindex);

    friend force_inline void vectorcall scatter(float *base_addr, fixed_size_simd<int, 16> vindex,
                                                fixed_size_simd<float, 16> v);
    friend force_inline void vectorcall scatter(float *base_addr, fixed_size_simd<int, 16> mask,
                                                fixed_size_simd<int, 16> vindex, fixed_size_simd<float, 16> v);

    template <typename U>
    friend force_inline fixed_size_simd<float, 16> vectorcall select(const fixed_size_simd<U, 16> mask,
                                                                     const fixed_size_simd<float, 16> vec1,
                                                                     const fixed_size_simd<float, 16> vec2);
    template <typename U>
    friend force_inline fixed_size_simd<int, 16> vectorcall select(const fixed_size_simd<U, 16> mask,
                                                                   const fixed_size_simd<int, 16> vec1,
                                                                   const fixed_size_simd<int, 16> vec2);
    template <typename U>
    friend force_inline fixed_size_simd<unsigned, 16> vectorcall select(const fixed_size_simd<U, 16> mask,
                                                                        const fixed_size_simd<unsigned, 16> vec1,
                                                                        const fixed_size_simd<unsigned, 16> vec2);

#ifndef NDEBUG
    friend void vectorcall __assert_valid_mask(const fixed_size_simd<float, 16> mask) {
        UNROLLED_FOR(i, 16, {
            const float val = mask.get<i>();
            assert(reinterpret_cast<const uint32_t &>(val) == 0 ||
                   reinterpret_cast<const uint32_t &>(val) == 0xffffffff);
        })
    }
#endif

    friend force_inline const float *value_ptr(const fixed_size_simd<float, 16> &v1) {
        return reinterpret_cast<const float *>(&v1.vec_);
    }
    friend force_inline float *value_ptr(fixed_size_simd<float, 16> &v1) { return reinterpret_cast<float *>(&v1.vec_); }

    static int size() { return 16; }
    static bool is_native() { return true; }
};

template <> class fixed_size_simd<int, 16> {
    union {
        __m512i vec_;
        int comp_[16];
    };

    friend class fixed_size_simd<float, 16>;
    friend class fixed_size_simd<unsigned, 16>;

    force_inline fixed_size_simd(const __m512i vec) : vec_(vec) {}

  public:
    force_inline fixed_size_simd() = default;
    force_inline fixed_size_simd(const int f) { vec_ = _mm512_set1_epi32(f); }
    force_inline fixed_size_simd(const int i0, const int i1, const int i2, const int i3, const int i4, const int i5,
                                 const int i6, const int i7, const int i8, const int i9, const int i10, const int i11,
                                 const int i12, const int i13, const int i14, const int i15) {
        vec_ = _mm512_setr_epi32(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15);
    }
    force_inline explicit fixed_size_simd(const int *f) { vec_ = _mm512_loadu_si512((const __m512i *)f); }
    force_inline fixed_size_simd(const int *f, vector_aligned_tag) { vec_ = _mm512_load_si512((const __m512i *)f); }

    force_inline int operator[](const int i) const {
        __m512i temp = _mm512_maskz_compress_epi32(__mmask16(1u << (i & 15)), vec_);
        return _mm512_cvtsi512_si32(temp);
    }

    force_inline int operator[](const long i) const { return operator[](int(i)); }

    template <int i> force_inline int get() const {
        __m128i temp = _mm512_extracti32x4_epi32(vec_, (i & 15) / 4);
        return _mm_extract_epi32(temp, (i & 15) % 4);
    }
    template <int i> force_inline void set(const int v) {
        //  TODO: find more optimal implementation (with compile-time index)
        vec_ = _mm512_mask_set1_epi32(vec_, __mmask16(1u << (i & 15)), v);
    }
    force_inline void set(const int i, const int v) {
        vec_ = _mm512_mask_set1_epi32(vec_, __mmask16(1u << (i & 15)), v);
    }

    force_inline fixed_size_simd<int, 16> &vectorcall operator+=(const fixed_size_simd<int, 16> rhs) {
        vec_ = _mm512_add_epi32(vec_, rhs.vec_);
        return *this;
    }

    force_inline fixed_size_simd<int, 16> &vectorcall operator-=(const fixed_size_simd<int, 16> rhs) {
        vec_ = _mm512_sub_epi32(vec_, rhs.vec_);
        return *this;
    }

    force_inline fixed_size_simd<int, 16> &vectorcall operator*=(const fixed_size_simd<int, 16> rhs) {
        vec_ = _mm512_mullo_epi32(vec_, rhs.vec_);
        return *this;
    }

    fixed_size_simd<int, 16> &vectorcall operator/=(const fixed_size_simd<int, 16> rhs) {
        UNROLLED_FOR(i, 16, { comp_[i] /= rhs.comp_[i]; })
        return *this;
    }

    force_inline fixed_size_simd<int, 16> &vectorcall operator|=(const fixed_size_simd<int, 16> rhs) {
        vec_ = _mm512_or_si512(vec_, rhs.vec_);
        return *this;
    }

    force_inline fixed_size_simd<int, 16> &vectorcall operator^=(const fixed_size_simd<int, 16> rhs) {
        vec_ = _mm512_xor_epi32(vec_, rhs.vec_);
        return *this;
    }

    force_inline fixed_size_simd<int, 16> operator-() const { return _mm512_sub_epi32(_mm512_setzero_si512(), vec_); }

    force_inline fixed_size_simd<int, 16> vectorcall operator==(const fixed_size_simd<int, 16> rhs) const {
        return _mm512_movm_epi32(_mm512_cmpeq_epi32_mask(vec_, rhs.vec_));
    }

    force_inline fixed_size_simd<int, 16> vectorcall operator!=(const fixed_size_simd<int, 16> rhs) const {
        return _mm512_andnot_si512(_mm512_movm_epi32(_mm512_cmpeq_epi32_mask(vec_, rhs.vec_)), _mm512_set1_epi32(~0));
    }

    force_inline fixed_size_simd<int, 16> &vectorcall operator&=(const fixed_size_simd<int, 16> rhs) {
        vec_ = _mm512_and_si512(vec_, rhs.vec_);
        return *this;
    }

    force_inline explicit operator fixed_size_simd<float, 16>() const { return _mm512_cvtepi32_ps(vec_); }

    // Load 8 bytes and expand each into a separate 32-bit lane, upper lanes are zeroed
    static force_inline fixed_size_simd<int, 16> vectorcall load_u8x8(const uint8_t *p) {
        fixed_size_simd<int, 16> ret;
        // NOTE: _mm512_castsi256_si512() leaves the upper lanes undefined - go through the
        // 512-bit widening convert of a zero-extended 64-bit load instead
        ret.vec_ = _mm512_cvtepu8_epi32(_mm_loadl_epi64((const __m128i *)p));
        return ret;
    }

    // Load 8 16-bit values and expand each into a separate 32-bit lane, upper lanes are zeroed
    static force_inline fixed_size_simd<int, 16> vectorcall load_u16x8(const uint16_t *p) {
        fixed_size_simd<int, 16> ret;
        ret.vec_ = _mm512_cvtepu16_epi32(_mm256_zextsi128_si256(_mm_loadu_si128((const __m128i *)p)));
        return ret;
    }

    force_inline explicit operator fixed_size_simd<unsigned, 16>() const;

    force_inline int hsum() const { return _mm512_reduce_add_epi32(vec_); }

    force_inline void store_to(int *f) const { _mm512_storeu_si512((__m512i *)f, vec_); }
    force_inline void store_to(int *f, vector_aligned_tag) const { _mm512_store_si512((__m512i *)f, vec_); }

    force_inline void vectorcall blend_to(const fixed_size_simd<int, 16> mask, const fixed_size_simd<int, 16> v1) {
        validate_mask(mask);
        vec_ = _mm512_ternarylogic_epi32(vec_, v1.vec_, _mm512_srai_epi32(mask.vec_, 31), 0xd8);
    }

    force_inline void vectorcall blend_inv_to(const fixed_size_simd<int, 16> mask, const fixed_size_simd<int, 16> v1) {
        validate_mask(mask);
        vec_ = _mm512_ternarylogic_epi32(v1.vec_, vec_, _mm512_srai_epi32(mask.vec_, 31), 0xd8);
    }

    force_inline int movemask() const { return _mm512_movemask_epi32(vec_); }
    force_inline int movemask8() const { return _mm512_movemask_epi8(vec_); }

    force_inline bool vectorcall all_zeros() const {
        return _mm512_cmpeq_epi32_mask(vec_, _mm512_setzero_si512()) == 0xFFFF;
    }

    force_inline bool vectorcall all_zeros(const fixed_size_simd<int, 16> mask) const {
        return _mm512_cmpeq_epi32_mask(_mm512_and_si512(vec_, mask.vec_), _mm512_setzero_si512()) == 0xFFFF;
    }

    force_inline bool not_all_zeros() const { return !all_zeros(); }

    friend force_inline fixed_size_simd<int, 16> vectorcall min(const fixed_size_simd<int, 16> v1,
                                                                const fixed_size_simd<int, 16> v2) {
        return _mm512_min_epi32(v1.vec_, v2.vec_);
    }

    friend force_inline fixed_size_simd<int, 16> vectorcall max(const fixed_size_simd<int, 16> v1,
                                                                const fixed_size_simd<int, 16> v2) {
        return _mm512_max_epi32(v1.vec_, v2.vec_);
    }

    friend force_inline fixed_size_simd<int, 16> vectorcall clamp(const fixed_size_simd<int, 16> v1,
                                                                  const fixed_size_simd<int, 16> _min,
                                                                  const fixed_size_simd<int, 16> _max) {
        return max(_min, min(v1, _max));
    }

    friend force_inline fixed_size_simd<int, 16> vectorcall cmpeq8(const fixed_size_simd<int, 16> &v1,
                                                                   const fixed_size_simd<int, 16> &v2) {
        fixed_size_simd<int, 16> ret;
        ret.vec_ = _mm512_movm_epi8(_mm512_cmpeq_epi8_mask(v1.vec_, v2.vec_));
        return ret;
    }

    force_inline static fixed_size_simd<int, 16> vectorcall and_not(const fixed_size_simd<int, 16> v1,
                                                                    const fixed_size_simd<int, 16> v2) {
        return _mm512_andnot_si512(v1.vec_, v2.vec_);
    }

    friend force_inline fixed_size_simd<int, 16> vectorcall operator&(const fixed_size_simd<int, 16> v1,
                                                                      const fixed_size_simd<int, 16> v2) {
        return _mm512_and_si512(v1.vec_, v2.vec_);
    }

    friend force_inline fixed_size_simd<int, 16> vectorcall operator|(const fixed_size_simd<int, 16> v1,
                                                                      const fixed_size_simd<int, 16> v2) {
        return _mm512_or_si512(v1.vec_, v2.vec_);
    }

    friend force_inline fixed_size_simd<int, 16> vectorcall operator^(const fixed_size_simd<int, 16> v1,
                                                                      const fixed_size_simd<int, 16> v2) {
        return _mm512_xor_si512(v1.vec_, v2.vec_);
    }

    friend force_inline fixed_size_simd<int, 16> vectorcall operator+(const fixed_size_simd<int, 16> v1,
                                                                      const fixed_size_simd<int, 16> v2) {
        return _mm512_add_epi32(v1.vec_, v2.vec_);
    }

    friend force_inline fixed_size_simd<int, 16> vectorcall operator-(const fixed_size_simd<int, 16> v1,
                                                                      const fixed_size_simd<int, 16> v2) {
        return _mm512_sub_epi32(v1.vec_, v2.vec_);
    }

    friend fixed_size_simd<int, 16> vectorcall operator*(const fixed_size_simd<int, 16> v1,
                                                         const fixed_size_simd<int, 16> v2) {
        return _mm512_mullo_epi32(v1.vec_, v2.vec_);
    }

    friend fixed_size_simd<int, 16> vectorcall operator/(const fixed_size_simd<int, 16> v1,
                                                         const fixed_size_simd<int, 16> v2) {
        fixed_size_simd<int, 16> ret;
        UNROLLED_FOR(i, 16, { ret.comp_[i] = v1.comp_[i] / v2.comp_[i]; })
        return ret;
    }

    friend force_inline fixed_size_simd<int, 16> vectorcall operator<(const fixed_size_simd<int, 16> v1,
                                                                      const fixed_size_simd<int, 16> v2) {
        return _mm512_movm_epi32(_mm512_cmpgt_epi32_mask(v2.vec_, v1.vec_));
    }

    friend force_inline fixed_size_simd<int, 16> vectorcall operator>(const fixed_size_simd<int, 16> v1,
                                                                      const fixed_size_simd<int, 16> v2) {
        return _mm512_movm_epi32(_mm512_cmpgt_epi32_mask(v1.vec_, v2.vec_));
    }

    friend force_inline fixed_size_simd<int, 16> vectorcall operator>=(const fixed_size_simd<int, 16> v1,
                                                                       const fixed_size_simd<int, 16> v2) {
        return _mm512_movm_epi32(_mm512_cmpge_epi32_mask(v1.vec_, v2.vec_));
    }

    friend force_inline fixed_size_simd<int, 16> vectorcall operator>>(const fixed_size_simd<int, 16> v1,
                                                                       const fixed_size_simd<int, 16> v2) {
        return _mm512_srlv_epi32(v1.vec_, v2.vec_);
    }

    friend force_inline fixed_size_simd<int, 16> vectorcall operator>>(const fixed_size_simd<int, 16> v1,
                                                                       const int v2) {
        return _mm512_srli_epi32(v1.vec_, v2);
    }

    friend force_inline fixed_size_simd<int, 16> vectorcall operator<<(const fixed_size_simd<int, 16> v1,
                                                                       const fixed_size_simd<int, 16> v2) {
        return _mm512_sllv_epi32(v1.vec_, v2.vec_);
    }

    friend force_inline fixed_size_simd<int, 16> vectorcall operator<<(const fixed_size_simd<int, 16> v1,
                                                                       const int v2) {
        return _mm512_slli_epi32(v1.vec_, v2);
    }

    force_inline fixed_size_simd<int, 16> operator~() const { return _mm512_andnot_si512(vec_, _mm512_set1_epi32(~0)); }

    friend force_inline fixed_size_simd<int, 16> vectorcall srai(const fixed_size_simd<int, 16> v1, const int v2) {
        return _mm512_srai_epi32(v1.vec_, v2);
    }

    friend force_inline fixed_size_simd<int, 16> vectorcall srli(const fixed_size_simd<int, 16> v1, const int v2) {
        return _mm512_srli_epi32(v1.vec_, v2);
    }

    friend force_inline bool vectorcall is_equal(const fixed_size_simd<int, 16> v1, const fixed_size_simd<int, 16> v2) {
        return _mm512_cmpeq_epi32_mask(v1.vec_, v2.vec_) == 0xFFFF;
    }

    friend fixed_size_simd<int, 16> vectorcall inclusive_scan(fixed_size_simd<int, 16> v1);

    friend force_inline fixed_size_simd<float, 16> vectorcall gather(const float *base_addr,
                                                                     fixed_size_simd<int, 16> vindex);
    friend force_inline fixed_size_simd<int, 16> vectorcall gather(const int *base_addr,
                                                                   fixed_size_simd<int, 16> vindex);
    friend force_inline fixed_size_simd<unsigned, 16> vectorcall gather(const unsigned *base_addr,
                                                                        fixed_size_simd<int, 16> vindex);

    friend force_inline void vectorcall scatter(float *base_addr, fixed_size_simd<int, 16> vindex,
                                                fixed_size_simd<float, 16> v);
    friend force_inline void vectorcall scatter(float *base_addr, fixed_size_simd<int, 16> vindex, const float v) {
        scatter(base_addr, vindex, fixed_size_simd<float, 16>{v});
    }
    friend force_inline void vectorcall scatter(float *base_addr, fixed_size_simd<int, 16> mask,
                                                fixed_size_simd<int, 16> vindex, fixed_size_simd<float, 16> v);
    friend force_inline void vectorcall scatter(float *base_addr, fixed_size_simd<int, 16> mask,
                                                fixed_size_simd<int, 16> vindex, const float v) {
        scatter(base_addr, mask, vindex, fixed_size_simd<float, 16>{v});
    }
    friend force_inline void vectorcall scatter(int *base_addr, fixed_size_simd<int, 16> vindex,
                                                fixed_size_simd<int, 16> v);
    friend force_inline void vectorcall scatter(int *base_addr, fixed_size_simd<int, 16> vindex, const int v) {
        scatter(base_addr, vindex, fixed_size_simd<int, 16>{v});
    }
    friend force_inline void vectorcall scatter(int *base_addr, fixed_size_simd<int, 16> mask,
                                                fixed_size_simd<int, 16> vindex, fixed_size_simd<int, 16> v);
    friend force_inline void vectorcall scatter(int *base_addr, fixed_size_simd<int, 16> mask,
                                                fixed_size_simd<int, 16> vindex, const int v) {
        scatter(base_addr, mask, vindex, fixed_size_simd<int, 16>{v});
    }
    friend force_inline void vectorcall scatter(unsigned *base_addr, fixed_size_simd<int, 16> vindex,
                                                fixed_size_simd<unsigned, 16> v);
    friend force_inline void vectorcall scatter(unsigned *base_addr, fixed_size_simd<int, 16> mask,
                                                fixed_size_simd<int, 16> vindex, fixed_size_simd<unsigned, 16> v);

    template <typename U>
    friend force_inline fixed_size_simd<float, 16> vectorcall select(const fixed_size_simd<U, 16> mask,
                                                                     const fixed_size_simd<float, 16> vec1,
                                                                     const fixed_size_simd<float, 16> vec2);
    template <typename U>
    friend force_inline fixed_size_simd<int, 16> vectorcall select(const fixed_size_simd<U, 16> mask,
                                                                   const fixed_size_simd<int, 16> vec1,
                                                                   const fixed_size_simd<int, 16> vec2);
    template <typename U>
    friend force_inline fixed_size_simd<unsigned, 16> vectorcall select(const fixed_size_simd<U, 16> mask,
                                                                        const fixed_size_simd<unsigned, 16> vec1,
                                                                        const fixed_size_simd<unsigned, 16> vec2);

#ifndef NDEBUG
    friend void vectorcall __assert_valid_mask(const fixed_size_simd<int, 16> mask) {
        UNROLLED_FOR(i, 16, {
            const int val = mask.get<i>();
            assert(val == 0 || val == -1);
        })
    }
#endif

    friend force_inline const int *value_ptr(const fixed_size_simd<int, 16> &v1) {
        return reinterpret_cast<const int *>(&v1.vec_);
    }
    friend force_inline int *value_ptr(fixed_size_simd<int, 16> &v1) { return reinterpret_cast<int *>(&v1.vec_); }

    static int size() { return 16; }
    static bool is_native() { return true; }
};

template <> class fixed_size_simd<unsigned, 16> {
    union {
        __m512i vec_;
        unsigned comp_[16];
    };

    friend class fixed_size_simd<float, 16>;
    friend class fixed_size_simd<int, 16>;

    force_inline fixed_size_simd(const __m512i vec) : vec_(vec) {}

  public:
    force_inline fixed_size_simd() = default;
    force_inline fixed_size_simd(const unsigned f) { vec_ = _mm512_set1_epi32(f); }
    force_inline fixed_size_simd(const unsigned i0, const unsigned i1, const unsigned i2, const unsigned i3,
                                 const unsigned i4, const unsigned i5, const unsigned i6, const unsigned i7,
                                 const unsigned i8, const unsigned i9, const unsigned i10, const unsigned i11,
                                 const unsigned i12, const unsigned i13, const unsigned i14, const unsigned i15) {
        vec_ = _mm512_setr_epi32(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15);
    }
    force_inline explicit fixed_size_simd(const unsigned *f) { vec_ = _mm512_loadu_si512((const __m512i *)f); }
    force_inline fixed_size_simd(const unsigned *f, vector_aligned_tag) {
        vec_ = _mm512_load_si512((const __m512i *)f);
    }

    force_inline unsigned operator[](const int i) const {
        __m512i temp = _mm512_maskz_compress_epi32(__mmask16(1u << (i & 15)), vec_);
        return _mm512_cvtsi512_si32(temp);
    }

    force_inline unsigned operator[](const long i) const { return operator[](int(i)); }

    template <int i> force_inline unsigned get() const {
        __m128i temp = _mm512_extracti32x4_epi32(vec_, (i & 15) / 4);
        return _mm_extract_epi32(temp, (i & 15) % 4);
    }
    template <int i> force_inline void set(const unsigned v) {
        //  TODO: find more optimal implementation (with compile-time index)
        vec_ = _mm512_mask_set1_epi32(vec_, __mmask16(1u << (i & 15)), v);
    }
    force_inline void set(const int i, const unsigned v) {
        vec_ = _mm512_mask_set1_epi32(vec_, __mmask16(1u << (i & 15)), v);
    }

    force_inline fixed_size_simd<unsigned, 16> &vectorcall operator+=(const fixed_size_simd<unsigned, 16> rhs) {
        vec_ = _mm512_add_epi32(vec_, rhs.vec_);
        return *this;
    }

    force_inline fixed_size_simd<unsigned, 16> &vectorcall operator-=(const fixed_size_simd<unsigned, 16> rhs) {
        vec_ = _mm512_sub_epi32(vec_, rhs.vec_);
        return *this;
    }

    force_inline fixed_size_simd<unsigned, 16> &vectorcall operator*=(const fixed_size_simd<unsigned, 16> rhs) {
        vec_ = _mm512_mullo_epi32(vec_, rhs.vec_);
        return *this;
    }

    fixed_size_simd<unsigned, 16> &vectorcall operator/=(const fixed_size_simd<unsigned, 16> rhs) {
        UNROLLED_FOR(i, 16, { comp_[i] /= rhs.comp_[i]; })
        return *this;
    }

    force_inline fixed_size_simd<unsigned, 16> &vectorcall operator|=(const fixed_size_simd<unsigned, 16> rhs) {
        vec_ = _mm512_or_si512(vec_, rhs.vec_);
        return *this;
    }

    force_inline fixed_size_simd<unsigned, 16> &vectorcall operator^=(const fixed_size_simd<unsigned, 16> rhs) {
        vec_ = _mm512_xor_epi32(vec_, rhs.vec_);
        return *this;
    }

    force_inline fixed_size_simd<unsigned, 16> vectorcall operator==(const fixed_size_simd<unsigned, 16> rhs) const {
        return _mm512_movm_epi32(_mm512_cmpeq_epi32_mask(vec_, rhs.vec_));
    }

    force_inline fixed_size_simd<unsigned, 16> vectorcall operator!=(const fixed_size_simd<unsigned, 16> rhs) const {
        return _mm512_andnot_si512(_mm512_movm_epi32(_mm512_cmpeq_epi32_mask(vec_, rhs.vec_)), _mm512_set1_epi32(~0));
    }

    force_inline fixed_size_simd<unsigned, 16> &vectorcall operator&=(const fixed_size_simd<unsigned, 16> rhs) {
        vec_ = _mm512_and_si512(vec_, rhs.vec_);
        return *this;
    }

    force_inline explicit operator fixed_size_simd<float, 16>() const { return _mm512_cvtepu32_ps(vec_); }

    force_inline explicit operator fixed_size_simd<int, 16>() const {
        fixed_size_simd<int, 16> ret;
        ret.vec_ = vec_;
        return ret;
    }

    force_inline unsigned hsum() const { return _mm512_reduce_add_epi32(vec_); }

    force_inline void store_to(unsigned *f) const { _mm512_storeu_si512((__m512i *)f, vec_); }
    force_inline void store_to(unsigned *f, vector_aligned_tag) const { _mm512_store_si512((__m512i *)f, vec_); }

    force_inline void vectorcall blend_to(const fixed_size_simd<unsigned, 16> mask,
                                          const fixed_size_simd<unsigned, 16> v1) {
        validate_mask(mask);
        vec_ = _mm512_ternarylogic_epi32(vec_, v1.vec_, _mm512_srai_epi32(mask.vec_, 31), 0xd8);
    }

    force_inline void vectorcall blend_inv_to(const fixed_size_simd<unsigned, 16> mask,
                                              const fixed_size_simd<unsigned, 16> v1) {
        validate_mask(mask);
        vec_ = _mm512_ternarylogic_epi32(v1.vec_, vec_, _mm512_srai_epi32(mask.vec_, 31), 0xd8);
    }

    force_inline int movemask() const { return _mm512_movemask_epi32(vec_); }
    force_inline int movemask8() const { return _mm512_movemask_epi8(vec_); }

    force_inline bool vectorcall all_zeros() const {
        return _mm512_cmpeq_epi32_mask(vec_, _mm512_setzero_si512()) == 0xFFFF;
    }

    force_inline bool vectorcall all_zeros(const fixed_size_simd<unsigned, 16> mask) const {
        return _mm512_cmpeq_epi32_mask(_mm512_and_si512(vec_, mask.vec_), _mm512_setzero_si512()) == 0xFFFF;
    }

    force_inline bool not_all_zeros() const { return !all_zeros(); }

    friend force_inline fixed_size_simd<unsigned, 16> vectorcall min(const fixed_size_simd<unsigned, 16> v1,
                                                                     const fixed_size_simd<unsigned, 16> v2) {
        return _mm512_min_epu32(v1.vec_, v2.vec_);
    }

    friend force_inline fixed_size_simd<unsigned, 16> vectorcall max(const fixed_size_simd<unsigned, 16> v1,
                                                                     const fixed_size_simd<unsigned, 16> v2) {
        return _mm512_max_epu32(v1.vec_, v2.vec_);
    }

    friend force_inline fixed_size_simd<unsigned, 16> vectorcall clamp(const fixed_size_simd<unsigned, 16> v1,
                                                                       const fixed_size_simd<unsigned, 16> _min,
                                                                       const fixed_size_simd<unsigned, 16> _max) {
        return max(_min, min(v1, _max));
    }

    friend force_inline fixed_size_simd<unsigned, 16> vectorcall cmpeq8(const fixed_size_simd<unsigned, 16> &v1,
                                                                        const fixed_size_simd<unsigned, 16> &v2) {
        fixed_size_simd<unsigned, 16> ret;
        ret.vec_ = _mm512_movm_epi8(_mm512_cmpeq_epi8_mask(v1.vec_, v2.vec_));
        return ret;
    }

    force_inline static fixed_size_simd<unsigned, 16> vectorcall and_not(const fixed_size_simd<unsigned, 16> v1,
                                                                         const fixed_size_simd<unsigned, 16> v2) {
        return _mm512_andnot_si512(v1.vec_, v2.vec_);
    }

    friend force_inline fixed_size_simd<unsigned, 16> vectorcall operator&(const fixed_size_simd<unsigned, 16> v1,
                                                                           const fixed_size_simd<unsigned, 16> v2) {
        return _mm512_and_si512(v1.vec_, v2.vec_);
    }

    friend force_inline fixed_size_simd<unsigned, 16> vectorcall operator|(const fixed_size_simd<unsigned, 16> v1,
                                                                           const fixed_size_simd<unsigned, 16> v2) {
        return _mm512_or_si512(v1.vec_, v2.vec_);
    }

    friend force_inline fixed_size_simd<unsigned, 16> vectorcall operator^(const fixed_size_simd<unsigned, 16> v1,
                                                                           const fixed_size_simd<unsigned, 16> v2) {
        return _mm512_xor_si512(v1.vec_, v2.vec_);
    }

    friend force_inline fixed_size_simd<unsigned, 16> vectorcall operator+(const fixed_size_simd<unsigned, 16> v1,
                                                                           const fixed_size_simd<unsigned, 16> v2) {
        return _mm512_add_epi32(v1.vec_, v2.vec_);
    }

    friend force_inline fixed_size_simd<unsigned, 16> vectorcall operator-(const fixed_size_simd<unsigned, 16> v1,
                                                                           const fixed_size_simd<unsigned, 16> v2) {
        return _mm512_sub_epi32(v1.vec_, v2.vec_);
    }

    friend fixed_size_simd<unsigned, 16> vectorcall operator*(const fixed_size_simd<unsigned, 16> v1,
                                                              const fixed_size_simd<unsigned, 16> v2) {
        return _mm512_mullo_epi32(v1.vec_, v2.vec_);
    }

    friend fixed_size_simd<unsigned, 16> vectorcall operator/(const fixed_size_simd<unsigned, 16> v1,
                                                              const fixed_size_simd<unsigned, 16> v2) {
        fixed_size_simd<unsigned, 16> ret;
        UNROLLED_FOR(i, 16, { ret.comp_[i] = v1.comp_[i] / v2.comp_[i]; })
        return ret;
    }

    friend force_inline fixed_size_simd<unsigned, 16> vectorcall operator>>(const fixed_size_simd<unsigned, 16> v1,
                                                                            const fixed_size_simd<unsigned, 16> v2) {
        return _mm512_srlv_epi32(v1.vec_, v2.vec_);
    }

    friend force_inline fixed_size_simd<unsigned, 16> vectorcall operator>>(const fixed_size_simd<unsigned, 16> v1,
                                                                            const unsigned v2) {
        return _mm512_srli_epi32(v1.vec_, v2);
    }

    friend force_inline fixed_size_simd<unsigned, 16> vectorcall operator<<(const fixed_size_simd<unsigned, 16> v1,
                                                                            const fixed_size_simd<unsigned, 16> v2) {
        return _mm512_sllv_epi32(v1.vec_, v2.vec_);
    }

    friend force_inline fixed_size_simd<unsigned, 16> vectorcall operator<<(const fixed_size_simd<unsigned, 16> v1,
                                                                            const unsigned v2) {
        return _mm512_slli_epi32(v1.vec_, v2);
    }

    force_inline fixed_size_simd<unsigned, 16> operator~() const {
        return _mm512_andnot_si512(vec_, _mm512_set1_epi32(~0));
    }

    friend force_inline fixed_size_simd<unsigned, 16> vectorcall srli(const fixed_size_simd<unsigned, 16> v1,
                                                                      const unsigned v2) {
        return _mm512_srli_epi32(v1.vec_, v2);
    }

    friend force_inline bool vectorcall is_equal(const fixed_size_simd<unsigned, 16> v1,
                                                 const fixed_size_simd<unsigned, 16> v2) {
        return _mm512_cmpeq_epi32_mask(v1.vec_, v2.vec_) == 0xFFFF;
    }

    friend fixed_size_simd<unsigned, 16> vectorcall inclusive_scan(fixed_size_simd<unsigned, 16> v1);

    friend force_inline fixed_size_simd<unsigned, 16> vectorcall gather(const unsigned *base_addr,
                                                                        fixed_size_simd<int, 16> vindex);

    friend force_inline void vectorcall scatter(unsigned *base_addr, fixed_size_simd<int, 16> vindex,
                                                fixed_size_simd<unsigned, 16> v);
    friend force_inline void vectorcall scatter(unsigned *base_addr, fixed_size_simd<int, 16> vindex,
                                                const unsigned v) {
        scatter(base_addr, vindex, fixed_size_simd<unsigned, 16>{v});
    }
    friend force_inline void vectorcall scatter(unsigned *base_addr, fixed_size_simd<int, 16> mask,
                                                fixed_size_simd<int, 16> vindex, fixed_size_simd<unsigned, 16> v);
    friend force_inline void vectorcall scatter(unsigned *base_addr, fixed_size_simd<int, 16> mask,
                                                fixed_size_simd<int, 16> vindex, const unsigned v) {
        scatter(base_addr, mask, vindex, fixed_size_simd<unsigned, 16>{v});
    }

    template <typename U>
    friend force_inline fixed_size_simd<float, 16> vectorcall select(const fixed_size_simd<U, 16> mask,
                                                                     const fixed_size_simd<float, 16> vec1,
                                                                     const fixed_size_simd<float, 16> vec2);
    template <typename U>
    friend force_inline fixed_size_simd<int, 16> vectorcall select(const fixed_size_simd<U, 16> mask,
                                                                   const fixed_size_simd<int, 16> vec1,
                                                                   const fixed_size_simd<int, 16> vec2);
    template <typename U>
    friend force_inline fixed_size_simd<unsigned, 16> vectorcall select(const fixed_size_simd<U, 16> mask,
                                                                        const fixed_size_simd<unsigned, 16> vec1,
                                                                        const fixed_size_simd<unsigned, 16> vec2);

#ifndef NDEBUG
    friend void vectorcall __assert_valid_mask(const fixed_size_simd<unsigned, 16> mask) {
        UNROLLED_FOR(i, 16, {
            const int val = mask.get<i>();
            assert(val == 0 || val == 0xffffffff);
        })
    }
#endif

    friend force_inline const unsigned *value_ptr(const fixed_size_simd<unsigned, 16> &v1) {
        return reinterpret_cast<const unsigned *>(&v1.vec_);
    }
    friend force_inline unsigned *value_ptr(fixed_size_simd<unsigned, 16> &v1) {
        return reinterpret_cast<unsigned *>(&v1.vec_);
    }

    static int size() { return 16; }
    static bool is_native() { return true; }
};

#ifdef USE_AVX512_FP16
// C++ UNet conv kernels may bind hw f16vec8 storage (consumed by CoreSIMD.h/Convolution.h)
#define RAY_AVX512_FP16_CONV
template <> class fixed_size_simd<float16, 8> {
    union {
        __m128h vec_;
        float16 comp_[8];
    };

    force_inline fixed_size_simd(const __m128h vec) : vec_(vec) {}

  public:
    force_inline fixed_size_simd() = default;
    force_inline fixed_size_simd(const float f) {
        const __m128i h = _mm256_cvtps_ph(_mm256_set1_ps(f), _MM_FROUND_TO_NEAREST_INT);
        vec_ = _mm_castsi128_ph(h);
    }
    force_inline fixed_size_simd(const float16 f0, const float16 f1, const float16 f2, const float16 f3,
                                 const float16 f4, const float16 f5, const float16 f6, const float16 f7) {
        const float16 data[8] = {f0, f1, f2, f3, f4, f5, f6, f7};
        vec_ = _mm_loadu_ph(data);
    }
    force_inline explicit fixed_size_simd(const float16 *f) { vec_ = _mm_loadu_ph(f); }
    force_inline explicit fixed_size_simd(const uint16_t *f) {
        vec_ = _mm_loadu_ph(reinterpret_cast<const __m128h *>(f));
    }
    force_inline fixed_size_simd(const float16 *f, vector_aligned_tag) { vec_ = _mm_load_ph(f); }
    force_inline fixed_size_simd(const uint16_t *f, vector_aligned_tag) {
        vec_ = _mm_load_ph(reinterpret_cast<const __m128h *>(f));
    }

    force_inline float16 operator[](const int i) const { return comp_[i & 7]; }
    force_inline float16 operator[](const long i) const { return operator[](int(i)); }

    template <int i> force_inline float16 get() const { return comp_[i & 7]; }
    template <int i> force_inline void set(const float16 v) { comp_[i & 7] = v; }
    force_inline void set(const int i, const float16 v) { comp_[i & 7] = v; }

    force_inline fixed_size_simd<float16, 8> &vectorcall operator+=(const fixed_size_simd<float16, 8> rhs) {
        vec_ = _mm_add_ph(vec_, rhs.vec_);
        return *this;
    }
    force_inline fixed_size_simd<float16, 8> &vectorcall operator-=(const fixed_size_simd<float16, 8> rhs) {
        vec_ = _mm_sub_ph(vec_, rhs.vec_);
        return *this;
    }
    force_inline fixed_size_simd<float16, 8> &vectorcall operator*=(const fixed_size_simd<float16, 8> rhs) {
        vec_ = _mm_mul_ph(vec_, rhs.vec_);
        return *this;
    }
    force_inline fixed_size_simd<float16, 8> &vectorcall operator/=(const fixed_size_simd<float16, 8> rhs) {
        vec_ = _mm_div_ph(vec_, rhs.vec_);
        return *this;
    }
    force_inline fixed_size_simd<float16, 8> &vectorcall operator|=(const fixed_size_simd<float16, 8> rhs) {
        vec_ = _mm_castsi128_ph(_mm_or_si128(_mm_castph_si128(vec_), _mm_castph_si128(rhs.vec_)));
        return *this;
    }
    force_inline fixed_size_simd<float16, 8> &vectorcall operator&=(const fixed_size_simd<float16, 8> rhs) {
        vec_ = _mm_castsi128_ph(_mm_and_si128(_mm_castph_si128(vec_), _mm_castph_si128(rhs.vec_)));
        return *this;
    }

    force_inline fixed_size_simd<float16, 8> operator~() const {
        return _mm_castsi128_ph(_mm_andnot_si128(_mm_castph_si128(vec_), _mm_set1_epi32(~0)));
    }
    force_inline fixed_size_simd<float16, 8> operator-() const {
        return _mm_castsi128_ph(_mm_xor_si128(_mm_castph_si128(vec_), _mm_set1_epi16(short(0x8000u))));
    }

    force_inline fixed_size_simd<float16, 8> sqrt() const { return _mm_sqrt_ph(vec_); }

    force_inline float hsum() const {
        const __m256 f32 = _mm256_cvtph_ps(_mm_castph_si128(vec_));
        __m256 temp = _mm256_hadd_ps(f32, f32);
        temp = _mm256_hadd_ps(temp, temp);
        return _mm_cvtss_f32(_mm_add_ss(_mm256_castps256_ps128(temp), _mm256_extractf128_ps(temp, 1)));
    }

    force_inline void store_to(float16 *f) const { _mm_storeu_ph(f, vec_); }
    force_inline void store_to(float16 *f, vector_aligned_tag) const { _mm_store_ph(f, vec_); }

    force_inline void vectorcall blend_to(const fixed_size_simd<float16, 8> mask,
                                          const fixed_size_simd<float16, 8> v1) {
        validate_mask(mask);
        const __mmask8 k = _mm_cmpneq_epi16_mask(_mm_castph_si128(mask.vec_), _mm_setzero_si128());
        vec_ = _mm_mask_blend_ph(k, vec_, v1.vec_);
    }
    force_inline void vectorcall blend_inv_to(const fixed_size_simd<float16, 8> mask,
                                              const fixed_size_simd<float16, 8> v1) {
        validate_mask(mask);
        const __mmask8 k = _mm_cmpneq_epi16_mask(_mm_castph_si128(mask.vec_), _mm_setzero_si128());
        vec_ = _mm_mask_blend_ph(k, v1.vec_, vec_);
    }

    friend force_inline fixed_size_simd<float16, 8> vectorcall min(fixed_size_simd<float16, 8> v1,
                                                                   fixed_size_simd<float16, 8> v2) {
        return _mm_min_ph(v1.vec_, v2.vec_);
    }
    friend force_inline fixed_size_simd<float16, 8> vectorcall max(fixed_size_simd<float16, 8> v1,
                                                                   fixed_size_simd<float16, 8> v2) {
        return _mm_max_ph(v1.vec_, v2.vec_);
    }
    friend force_inline fixed_size_simd<float16, 8> vectorcall clamp(fixed_size_simd<float16, 8> v1,
                                                                     fixed_size_simd<float16, 8> _min,
                                                                     fixed_size_simd<float16, 8> _max) {
        return max(_min, min(v1, _max));
    }
    friend force_inline fixed_size_simd<float16, 8> vectorcall saturate(const fixed_size_simd<float16, 8> v1) {
        return clamp(v1, 0.0f, 1.0f);
    }

    friend force_inline fixed_size_simd<float16, 8> vectorcall operator+(fixed_size_simd<float16, 8> v1,
                                                                         fixed_size_simd<float16, 8> v2) {
        return _mm_add_ph(v1.vec_, v2.vec_);
    }
    friend force_inline fixed_size_simd<float16, 8> vectorcall operator-(fixed_size_simd<float16, 8> v1,
                                                                         fixed_size_simd<float16, 8> v2) {
        return _mm_sub_ph(v1.vec_, v2.vec_);
    }
    friend force_inline fixed_size_simd<float16, 8> vectorcall operator*(fixed_size_simd<float16, 8> v1,
                                                                         fixed_size_simd<float16, 8> v2) {
        return _mm_mul_ph(v1.vec_, v2.vec_);
    }
    friend force_inline fixed_size_simd<float16, 8> vectorcall operator/(fixed_size_simd<float16, 8> v1,
                                                                         fixed_size_simd<float16, 8> v2) {
        return _mm_div_ph(v1.vec_, v2.vec_);
    }

    friend force_inline fixed_size_simd<float16, 8> vectorcall operator<(fixed_size_simd<float16, 8> v1,
                                                                         fixed_size_simd<float16, 8> v2) {
        return _mm_castsi128_ph(_mm_movm_epi16(_mm_cmp_ph_mask(v1.vec_, v2.vec_, _CMP_LT_OS)));
    }
    friend force_inline fixed_size_simd<float16, 8> vectorcall operator<=(fixed_size_simd<float16, 8> v1,
                                                                          fixed_size_simd<float16, 8> v2) {
        return _mm_castsi128_ph(_mm_movm_epi16(_mm_cmp_ph_mask(v1.vec_, v2.vec_, _CMP_LE_OS)));
    }
    friend force_inline fixed_size_simd<float16, 8> vectorcall operator>(fixed_size_simd<float16, 8> v1,
                                                                         fixed_size_simd<float16, 8> v2) {
        return _mm_castsi128_ph(_mm_movm_epi16(_mm_cmp_ph_mask(v1.vec_, v2.vec_, _CMP_GT_OS)));
    }
    friend force_inline fixed_size_simd<float16, 8> vectorcall operator>=(fixed_size_simd<float16, 8> v1,
                                                                          fixed_size_simd<float16, 8> v2) {
        return _mm_castsi128_ph(_mm_movm_epi16(_mm_cmp_ph_mask(v1.vec_, v2.vec_, _CMP_GE_OS)));
    }
    friend force_inline fixed_size_simd<float16, 8> vectorcall operator==(fixed_size_simd<float16, 8> v1,
                                                                          fixed_size_simd<float16, 8> v2) {
        return _mm_castsi128_ph(_mm_movm_epi16(_mm_cmp_ph_mask(v1.vec_, v2.vec_, _CMP_EQ_OS)));
    }
    friend force_inline fixed_size_simd<float16, 8> vectorcall operator!=(fixed_size_simd<float16, 8> v1,
                                                                          fixed_size_simd<float16, 8> v2) {
        return _mm_castsi128_ph(_mm_movm_epi16(_mm_cmp_ph_mask(v1.vec_, v2.vec_, _CMP_NEQ_OS)));
    }

    friend force_inline fixed_size_simd<float16, 8>
        vectorcall fmadd(fixed_size_simd<float16, 8> a, fixed_size_simd<float16, 8> b, fixed_size_simd<float16, 8> c) {
        return _mm_fmadd_ph(a.vec_, b.vec_, c.vec_);
    }
    friend force_inline fixed_size_simd<float16, 8>
        vectorcall fmsub(fixed_size_simd<float16, 8> a, fixed_size_simd<float16, 8> b, fixed_size_simd<float16, 8> c) {
        return _mm_fmsub_ph(a.vec_, b.vec_, c.vec_);
    }

    friend force_inline fixed_size_simd<float16, 8> vectorcall copysign(fixed_size_simd<float16, 8> val,
                                                                        fixed_size_simd<float16, 8> sign) {
        const __m128i sign_mask = _mm_set1_epi16(short(0x8000u));
        const __m128i s = _mm_and_si128(_mm_castph_si128(sign.vec_), sign_mask);
        const __m128i v = _mm_andnot_si128(sign_mask, _mm_castph_si128(val.vec_));
        return _mm_castsi128_ph(_mm_or_si128(v, s));
    }

#ifndef NDEBUG
    friend void vectorcall __assert_valid_mask(const fixed_size_simd<float16, 8> mask) {
        const uint16_t *p = reinterpret_cast<const uint16_t *>(&mask.vec_);
        UNROLLED_FOR(i, 8, { assert(p[i] == 0u || p[i] == 0xffffu); })
    }
#endif

    friend force_inline const float16 *value_ptr(const fixed_size_simd<float16, 8> &v1) {
        return reinterpret_cast<const float16 *>(&v1.vec_);
    }
    friend force_inline float16 *value_ptr(fixed_size_simd<float16, 8> &v1) {
        return reinterpret_cast<float16 *>(&v1.vec_);
    }

    static int size() { return 8; }
    static bool is_native() { return true; }
};

template <> class fixed_size_simd<float16, 16> {
    union {
        __m256h vec_;
        float16 comp_[16];
    };

    force_inline fixed_size_simd(const __m256h vec) : vec_(vec) {}

  public:
    force_inline fixed_size_simd() = default;
    force_inline fixed_size_simd(const float f) {
        const __m256i h = _mm512_cvtps_ph(_mm512_set1_ps(f), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        vec_ = _mm256_castsi256_ph(h);
    }
    force_inline fixed_size_simd(const float16 f0, const float16 f1, const float16 f2, const float16 f3,
                                 const float16 f4, const float16 f5, const float16 f6, const float16 f7,
                                 const float16 f8, const float16 f9, const float16 f10, const float16 f11,
                                 const float16 f12, const float16 f13, const float16 f14, const float16 f15) {
        const float16 data[16] = {f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15};
        vec_ = _mm256_loadu_ph(data);
    }
    force_inline explicit fixed_size_simd(const float16 *f) { vec_ = _mm256_loadu_ph(f); }
    force_inline fixed_size_simd(const float16 *f, vector_aligned_tag) { vec_ = _mm256_load_ph(f); }

    force_inline float16 operator[](const int i) const { return comp_[i & 15]; }
    force_inline float16 operator[](const long i) const { return operator[](int(i)); }

    template <int i> force_inline float16 get() const { return comp_[i & 15]; }
    template <int i> force_inline void set(const float16 v) { comp_[i & 15] = v; }
    force_inline void set(const int i, const float16 v) { comp_[i & 15] = v; }

    force_inline fixed_size_simd<float16, 16> &vectorcall operator+=(const fixed_size_simd<float16, 16> rhs) {
        vec_ = _mm256_add_ph(vec_, rhs.vec_);
        return *this;
    }
    force_inline fixed_size_simd<float16, 16> &vectorcall operator-=(const fixed_size_simd<float16, 16> rhs) {
        vec_ = _mm256_sub_ph(vec_, rhs.vec_);
        return *this;
    }
    force_inline fixed_size_simd<float16, 16> &vectorcall operator*=(const fixed_size_simd<float16, 16> rhs) {
        vec_ = _mm256_mul_ph(vec_, rhs.vec_);
        return *this;
    }
    force_inline fixed_size_simd<float16, 16> &vectorcall operator/=(const fixed_size_simd<float16, 16> rhs) {
        vec_ = _mm256_div_ph(vec_, rhs.vec_);
        return *this;
    }
    force_inline fixed_size_simd<float16, 16> &vectorcall operator|=(const fixed_size_simd<float16, 16> rhs) {
        vec_ = _mm256_castsi256_ph(_mm256_or_si256(_mm256_castph_si256(vec_), _mm256_castph_si256(rhs.vec_)));
        return *this;
    }
    force_inline fixed_size_simd<float16, 16> &vectorcall operator&=(const fixed_size_simd<float16, 16> rhs) {
        vec_ = _mm256_castsi256_ph(_mm256_and_si256(_mm256_castph_si256(vec_), _mm256_castph_si256(rhs.vec_)));
        return *this;
    }

    force_inline fixed_size_simd<float16, 16> operator~() const {
        return _mm256_castsi256_ph(_mm256_andnot_si256(_mm256_castph_si256(vec_), _mm256_set1_epi32(~0)));
    }
    force_inline fixed_size_simd<float16, 16> operator-() const {
        return _mm256_castsi256_ph(_mm256_xor_si256(_mm256_castph_si256(vec_), _mm256_set1_epi16(short(0x8000u))));
    }

    force_inline fixed_size_simd<float16, 16> sqrt() const { return _mm256_sqrt_ph(vec_); }

    force_inline float hsum() const {
        const __m512 f32 = _mm512_cvtph_ps(_mm256_castph_si256(vec_));
        return _mm512_reduce_add_ps(f32);
    }

    force_inline void store_to(float16 *f) const { _mm256_storeu_ph(f, vec_); }
    force_inline void store_to(float16 *f, vector_aligned_tag) const { _mm256_store_ph(f, vec_); }

    force_inline void vectorcall blend_to(const fixed_size_simd<float16, 16> mask,
                                          const fixed_size_simd<float16, 16> v1) {
        validate_mask(mask);
        const __mmask16 k = _mm256_cmpneq_epi16_mask(_mm256_castph_si256(mask.vec_), _mm256_setzero_si256());
        vec_ = _mm256_mask_blend_ph(k, vec_, v1.vec_);
    }
    force_inline void vectorcall blend_inv_to(const fixed_size_simd<float16, 16> mask,
                                              const fixed_size_simd<float16, 16> v1) {
        validate_mask(mask);
        const __mmask16 k = _mm256_cmpneq_epi16_mask(_mm256_castph_si256(mask.vec_), _mm256_setzero_si256());
        vec_ = _mm256_mask_blend_ph(k, v1.vec_, vec_);
    }

    friend force_inline fixed_size_simd<float16, 16> vectorcall min(fixed_size_simd<float16, 16> v1,
                                                                    fixed_size_simd<float16, 16> v2) {
        return _mm256_min_ph(v1.vec_, v2.vec_);
    }
    friend force_inline fixed_size_simd<float16, 16> vectorcall max(fixed_size_simd<float16, 16> v1,
                                                                    fixed_size_simd<float16, 16> v2) {
        return _mm256_max_ph(v1.vec_, v2.vec_);
    }
    friend force_inline fixed_size_simd<float16, 16> vectorcall clamp(fixed_size_simd<float16, 16> v1,
                                                                      fixed_size_simd<float16, 16> _min,
                                                                      fixed_size_simd<float16, 16> _max) {
        return max(_min, min(v1, _max));
    }
    friend force_inline fixed_size_simd<float16, 16> vectorcall saturate(const fixed_size_simd<float16, 16> v1) {
        return clamp(v1, 0.0f, 1.0f);
    }

    friend force_inline fixed_size_simd<float16, 16> vectorcall operator+(fixed_size_simd<float16, 16> v1,
                                                                          fixed_size_simd<float16, 16> v2) {
        return _mm256_add_ph(v1.vec_, v2.vec_);
    }
    friend force_inline fixed_size_simd<float16, 16> vectorcall operator-(fixed_size_simd<float16, 16> v1,
                                                                          fixed_size_simd<float16, 16> v2) {
        return _mm256_sub_ph(v1.vec_, v2.vec_);
    }
    friend force_inline fixed_size_simd<float16, 16> vectorcall operator*(fixed_size_simd<float16, 16> v1,
                                                                          fixed_size_simd<float16, 16> v2) {
        return _mm256_mul_ph(v1.vec_, v2.vec_);
    }
    friend force_inline fixed_size_simd<float16, 16> vectorcall operator/(fixed_size_simd<float16, 16> v1,
                                                                          fixed_size_simd<float16, 16> v2) {
        return _mm256_div_ph(v1.vec_, v2.vec_);
    }

    friend force_inline fixed_size_simd<float16, 16> vectorcall operator<(fixed_size_simd<float16, 16> v1,
                                                                          fixed_size_simd<float16, 16> v2) {
        return _mm256_castsi256_ph(_mm256_movm_epi16(_mm256_cmp_ph_mask(v1.vec_, v2.vec_, _CMP_LT_OS)));
    }
    friend force_inline fixed_size_simd<float16, 16> vectorcall operator<=(fixed_size_simd<float16, 16> v1,
                                                                           fixed_size_simd<float16, 16> v2) {
        return _mm256_castsi256_ph(_mm256_movm_epi16(_mm256_cmp_ph_mask(v1.vec_, v2.vec_, _CMP_LE_OS)));
    }
    friend force_inline fixed_size_simd<float16, 16> vectorcall operator>(fixed_size_simd<float16, 16> v1,
                                                                          fixed_size_simd<float16, 16> v2) {
        return _mm256_castsi256_ph(_mm256_movm_epi16(_mm256_cmp_ph_mask(v1.vec_, v2.vec_, _CMP_GT_OS)));
    }
    friend force_inline fixed_size_simd<float16, 16> vectorcall operator>=(fixed_size_simd<float16, 16> v1,
                                                                           fixed_size_simd<float16, 16> v2) {
        return _mm256_castsi256_ph(_mm256_movm_epi16(_mm256_cmp_ph_mask(v1.vec_, v2.vec_, _CMP_GE_OS)));
    }
    friend force_inline fixed_size_simd<float16, 16> vectorcall operator==(fixed_size_simd<float16, 16> v1,
                                                                           fixed_size_simd<float16, 16> v2) {
        return _mm256_castsi256_ph(_mm256_movm_epi16(_mm256_cmp_ph_mask(v1.vec_, v2.vec_, _CMP_EQ_OS)));
    }
    friend force_inline fixed_size_simd<float16, 16> vectorcall operator!=(fixed_size_simd<float16, 16> v1,
                                                                           fixed_size_simd<float16, 16> v2) {
        return _mm256_castsi256_ph(_mm256_movm_epi16(_mm256_cmp_ph_mask(v1.vec_, v2.vec_, _CMP_NEQ_OS)));
    }

    friend force_inline fixed_size_simd<float16, 16> vectorcall fmadd(fixed_size_simd<float16, 16> a,
                                                                      fixed_size_simd<float16, 16> b,
                                                                      fixed_size_simd<float16, 16> c) {
        return _mm256_fmadd_ph(a.vec_, b.vec_, c.vec_);
    }
    friend force_inline fixed_size_simd<float16, 16> vectorcall fmsub(fixed_size_simd<float16, 16> a,
                                                                      fixed_size_simd<float16, 16> b,
                                                                      fixed_size_simd<float16, 16> c) {
        return _mm256_fmsub_ph(a.vec_, b.vec_, c.vec_);
    }

    friend force_inline fixed_size_simd<float16, 16> vectorcall copysign(fixed_size_simd<float16, 16> val,
                                                                         fixed_size_simd<float16, 16> sign) {
        const __m256i sign_mask = _mm256_set1_epi16(short(0x8000u));
        const __m256i s = _mm256_and_si256(_mm256_castph_si256(sign.vec_), sign_mask);
        const __m256i v = _mm256_andnot_si256(sign_mask, _mm256_castph_si256(val.vec_));
        return _mm256_castsi256_ph(_mm256_or_si256(v, s));
    }

#ifndef NDEBUG
    friend void vectorcall __assert_valid_mask(const fixed_size_simd<float16, 16> mask) {
        const uint16_t *p = reinterpret_cast<const uint16_t *>(&mask.vec_);
        UNROLLED_FOR(i, 16, { assert(p[i] == 0u || p[i] == 0xffffu); })
    }
#endif

    friend force_inline const float16 *value_ptr(const fixed_size_simd<float16, 16> &v1) {
        return reinterpret_cast<const float16 *>(&v1.vec_);
    }
    friend force_inline float16 *value_ptr(fixed_size_simd<float16, 16> &v1) {
        return reinterpret_cast<float16 *>(&v1.vec_);
    }

    static int size() { return 16; }
    static bool is_native() { return true; }
};

template <> class fixed_size_simd<float16, 32> {
    union {
        __m512h vec_;
        float16 comp_[32];
    };

    force_inline fixed_size_simd(const __m512h vec) : vec_(vec) {}

  public:
    force_inline fixed_size_simd() = default;
    force_inline fixed_size_simd(const float f) {
        const __m256i h = _mm512_cvtps_ph(_mm512_set1_ps(f), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        vec_ = _mm512_castsi512_ph(_mm512_inserti64x4(_mm512_castsi256_si512(h), h, 1));
    }
    force_inline explicit fixed_size_simd(const float16 *f) { vec_ = _mm512_loadu_ph(f); }
    force_inline fixed_size_simd(const float16 *f, vector_aligned_tag) { vec_ = _mm512_load_ph(f); }
    force_inline fixed_size_simd(const fixed_size_simd<float, 16> &lo, const fixed_size_simd<float, 16> &hi) {
        const __m256i lo16 = _mm512_cvtps_ph(lo.vec_, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        const __m256i hi16 = _mm512_cvtps_ph(hi.vec_, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        vec_ = _mm512_castsi512_ph(_mm512_inserti64x4(_mm512_castsi256_si512(lo16), hi16, 1));
    }

    force_inline float16 operator[](const int i) const { return comp_[i & 31]; }
    force_inline float16 operator[](const long i) const { return operator[](int(i)); }

    template <int i> force_inline float16 get() const { return comp_[i & 31]; }
    template <int i> force_inline void set(const float16 v) { comp_[i & 31] = v; }
    force_inline void set(const int i, const float16 v) { comp_[i & 31] = v; }

    // Accessors that convert halves to float for mixed-precision code
    force_inline fixed_size_simd<float, 16> lo() const {
        return _mm512_cvtph_ps(_mm512_castsi512_si256(_mm512_castph_si512(vec_)));
    }
    force_inline fixed_size_simd<float, 16> hi() const {
        return _mm512_cvtph_ps(_mm512_extracti64x4_epi64(_mm512_castph_si512(vec_), 1));
    }

    force_inline fixed_size_simd<float16, 32> &vectorcall operator+=(const fixed_size_simd<float16, 32> rhs) {
        vec_ = _mm512_add_ph(vec_, rhs.vec_);
        return *this;
    }
    force_inline fixed_size_simd<float16, 32> &vectorcall operator-=(const fixed_size_simd<float16, 32> rhs) {
        vec_ = _mm512_sub_ph(vec_, rhs.vec_);
        return *this;
    }
    force_inline fixed_size_simd<float16, 32> &vectorcall operator*=(const fixed_size_simd<float16, 32> rhs) {
        vec_ = _mm512_mul_ph(vec_, rhs.vec_);
        return *this;
    }
    force_inline fixed_size_simd<float16, 32> &vectorcall operator/=(const fixed_size_simd<float16, 32> rhs) {
        vec_ = _mm512_div_ph(vec_, rhs.vec_);
        return *this;
    }
    force_inline fixed_size_simd<float16, 32> &vectorcall operator|=(const fixed_size_simd<float16, 32> rhs) {
        vec_ = _mm512_castsi512_ph(_mm512_or_si512(_mm512_castph_si512(vec_), _mm512_castph_si512(rhs.vec_)));
        return *this;
    }
    force_inline fixed_size_simd<float16, 32> &vectorcall operator&=(const fixed_size_simd<float16, 32> rhs) {
        vec_ = _mm512_castsi512_ph(_mm512_and_si512(_mm512_castph_si512(vec_), _mm512_castph_si512(rhs.vec_)));
        return *this;
    }

    force_inline fixed_size_simd<float16, 32> operator~() const {
        return _mm512_castsi512_ph(_mm512_andnot_si512(_mm512_castph_si512(vec_), _mm512_set1_epi32(~0)));
    }
    force_inline fixed_size_simd<float16, 32> operator-() const {
        return _mm512_castsi512_ph(_mm512_xor_si512(_mm512_castph_si512(vec_), _mm512_set1_epi16(short(0x8000u))));
    }

    force_inline fixed_size_simd<float16, 32> sqrt() const { return _mm512_sqrt_ph(vec_); }

    force_inline float hsum() const { return _mm512_reduce_add_ps(lo().vec_) + _mm512_reduce_add_ps(hi().vec_); }

    force_inline void store_to(float16 *f) const { _mm512_storeu_ph(f, vec_); }
    force_inline void store_to(float16 *f, vector_aligned_tag) const { _mm512_store_ph(f, vec_); }

    force_inline void vectorcall blend_to(const fixed_size_simd<float16, 32> mask,
                                          const fixed_size_simd<float16, 32> v1) {
        validate_mask(mask);
        const __mmask32 k = _mm512_cmpneq_epi16_mask(_mm512_castph_si512(mask.vec_), _mm512_setzero_si512());
        vec_ = _mm512_mask_blend_ph(k, vec_, v1.vec_);
    }
    force_inline void vectorcall blend_inv_to(const fixed_size_simd<float16, 32> mask,
                                              const fixed_size_simd<float16, 32> v1) {
        validate_mask(mask);
        const __mmask32 k = _mm512_cmpneq_epi16_mask(_mm512_castph_si512(mask.vec_), _mm512_setzero_si512());
        vec_ = _mm512_mask_blend_ph(k, v1.vec_, vec_);
    }

    friend force_inline fixed_size_simd<float16, 32> vectorcall min(fixed_size_simd<float16, 32> v1,
                                                                    fixed_size_simd<float16, 32> v2) {
        return _mm512_min_ph(v1.vec_, v2.vec_);
    }
    friend force_inline fixed_size_simd<float16, 32> vectorcall max(fixed_size_simd<float16, 32> v1,
                                                                    fixed_size_simd<float16, 32> v2) {
        return _mm512_max_ph(v1.vec_, v2.vec_);
    }
    friend force_inline fixed_size_simd<float16, 32> vectorcall clamp(fixed_size_simd<float16, 32> v1,
                                                                      fixed_size_simd<float16, 32> _min,
                                                                      fixed_size_simd<float16, 32> _max) {
        return max(_min, min(v1, _max));
    }
    friend force_inline fixed_size_simd<float16, 32> vectorcall saturate(const fixed_size_simd<float16, 32> v1) {
        return clamp(v1, 0.0f, 1.0f);
    }

    friend force_inline fixed_size_simd<float16, 32> vectorcall operator+(fixed_size_simd<float16, 32> v1,
                                                                          fixed_size_simd<float16, 32> v2) {
        return _mm512_add_ph(v1.vec_, v2.vec_);
    }
    friend force_inline fixed_size_simd<float16, 32> vectorcall operator-(fixed_size_simd<float16, 32> v1,
                                                                          fixed_size_simd<float16, 32> v2) {
        return _mm512_sub_ph(v1.vec_, v2.vec_);
    }
    friend force_inline fixed_size_simd<float16, 32> vectorcall operator*(fixed_size_simd<float16, 32> v1,
                                                                          fixed_size_simd<float16, 32> v2) {
        return _mm512_mul_ph(v1.vec_, v2.vec_);
    }
    friend force_inline fixed_size_simd<float16, 32> vectorcall operator/(fixed_size_simd<float16, 32> v1,
                                                                          fixed_size_simd<float16, 32> v2) {
        return _mm512_div_ph(v1.vec_, v2.vec_);
    }

    friend force_inline fixed_size_simd<float16, 32> vectorcall operator<(fixed_size_simd<float16, 32> v1,
                                                                          fixed_size_simd<float16, 32> v2) {
        return _mm512_castsi512_ph(_mm512_movm_epi16(_mm512_cmp_ph_mask(v1.vec_, v2.vec_, _CMP_LT_OS)));
    }
    friend force_inline fixed_size_simd<float16, 32> vectorcall operator<=(fixed_size_simd<float16, 32> v1,
                                                                           fixed_size_simd<float16, 32> v2) {
        return _mm512_castsi512_ph(_mm512_movm_epi16(_mm512_cmp_ph_mask(v1.vec_, v2.vec_, _CMP_LE_OS)));
    }
    friend force_inline fixed_size_simd<float16, 32> vectorcall operator>(fixed_size_simd<float16, 32> v1,
                                                                          fixed_size_simd<float16, 32> v2) {
        return _mm512_castsi512_ph(_mm512_movm_epi16(_mm512_cmp_ph_mask(v1.vec_, v2.vec_, _CMP_GT_OS)));
    }
    friend force_inline fixed_size_simd<float16, 32> vectorcall operator>=(fixed_size_simd<float16, 32> v1,
                                                                           fixed_size_simd<float16, 32> v2) {
        return _mm512_castsi512_ph(_mm512_movm_epi16(_mm512_cmp_ph_mask(v1.vec_, v2.vec_, _CMP_GE_OS)));
    }
    friend force_inline fixed_size_simd<float16, 32> vectorcall operator==(fixed_size_simd<float16, 32> v1,
                                                                           fixed_size_simd<float16, 32> v2) {
        return _mm512_castsi512_ph(_mm512_movm_epi16(_mm512_cmp_ph_mask(v1.vec_, v2.vec_, _CMP_EQ_OS)));
    }
    friend force_inline fixed_size_simd<float16, 32> vectorcall operator!=(fixed_size_simd<float16, 32> v1,
                                                                           fixed_size_simd<float16, 32> v2) {
        return _mm512_castsi512_ph(_mm512_movm_epi16(_mm512_cmp_ph_mask(v1.vec_, v2.vec_, _CMP_NEQ_OS)));
    }

    friend force_inline fixed_size_simd<float16, 32> vectorcall fmadd(fixed_size_simd<float16, 32> a,
                                                                      fixed_size_simd<float16, 32> b,
                                                                      fixed_size_simd<float16, 32> c) {
        return _mm512_fmadd_ph(a.vec_, b.vec_, c.vec_);
    }
    friend force_inline fixed_size_simd<float16, 32> vectorcall fmsub(fixed_size_simd<float16, 32> a,
                                                                      fixed_size_simd<float16, 32> b,
                                                                      fixed_size_simd<float16, 32> c) {
        return _mm512_fmsub_ph(a.vec_, b.vec_, c.vec_);
    }

    friend force_inline fixed_size_simd<float16, 32> vectorcall copysign(fixed_size_simd<float16, 32> val,
                                                                         fixed_size_simd<float16, 32> sign) {
        const __m512i sign_mask = _mm512_set1_epi16(short(0x8000u));
        const __m512i s = _mm512_and_si512(_mm512_castph_si512(sign.vec_), sign_mask);
        const __m512i v = _mm512_andnot_si512(sign_mask, _mm512_castph_si512(val.vec_));
        return _mm512_castsi512_ph(_mm512_or_si512(v, s));
    }

#ifndef NDEBUG
    friend void vectorcall __assert_valid_mask(const fixed_size_simd<float16, 32> mask) {
        const uint16_t *p = reinterpret_cast<const uint16_t *>(&mask.vec_);
        UNROLLED_FOR(i, 16, {
            assert(p[i] == 0u || p[i] == 0xffffu);
            assert(p[16 + i] == 0u || p[16 + i] == 0xffffu);
        })
    }
#endif

    friend force_inline const float16 *value_ptr(const fixed_size_simd<float16, 32> &v1) {
        return reinterpret_cast<const float16 *>(&v1.vec_);
    }
    friend force_inline float16 *value_ptr(fixed_size_simd<float16, 32> &v1) {
        return reinterpret_cast<float16 *>(&v1.vec_);
    }

    static int size() { return 32; }
    static bool is_native() { return true; }
};
#endif

force_inline fixed_size_simd<float, 16> fixed_size_simd<float, 16>::operator~() const {
    return _mm512_castsi512_ps(_mm512_andnot_si512(_mm512_castps_si512(vec_), _mm512_set1_epi32(~0)));
}

force_inline fixed_size_simd<float, 16> fixed_size_simd<float, 16>::operator-() const {
    const __m512 m = _mm512_set1_ps(-0.0f);
    return _mm512_xor_ps(vec_, m);
}

force_inline fixed_size_simd<float, 16>::operator fixed_size_simd<int, 16>() const { return _mm512_cvttps_epi32(vec_); }

force_inline fixed_size_simd<float, 16>::operator fixed_size_simd<unsigned, 16>() const {
    return _mm512_cvttps_epi32(vec_);
}

force_inline fixed_size_simd<int, 16>::operator fixed_size_simd<unsigned, 16>() const {
    fixed_size_simd<unsigned, 16> ret;
    ret.vec_ = vec_;
    return ret;
}

force_inline fixed_size_simd<float, 16> fixed_size_simd<float, 16>::sqrt() const { return _mm512_sqrt_ps(vec_); }

inline fixed_size_simd<float, 16> fixed_size_simd<float, 16>::log() const {
    fixed_size_simd<float, 16> ret;
    UNROLLED_FOR(i, 16, { ret.comp_[i] = logf(comp_[i]); })
    return ret;
}

force_inline fixed_size_simd<float, 16> vectorcall min(const fixed_size_simd<float, 16> v1,
                                                       const fixed_size_simd<float, 16> v2) {
    return _mm512_min_ps(v1.vec_, v2.vec_);
}

force_inline fixed_size_simd<float, 16> vectorcall max(const fixed_size_simd<float, 16> v1,
                                                       const fixed_size_simd<float, 16> v2) {
    return _mm512_max_ps(v1.vec_, v2.vec_);
}

force_inline fixed_size_simd<float, 16> vectorcall and_not(const fixed_size_simd<float, 16> v1,
                                                           const fixed_size_simd<float, 16> v2) {
    return _mm512_andnot_ps(v1.vec_, v2.vec_);
}

force_inline fixed_size_simd<float, 16> vectorcall floor(const fixed_size_simd<float, 16> v1) {
    return _mm512_floor_ps(v1.vec_);
}

force_inline fixed_size_simd<float, 16> vectorcall ceil(const fixed_size_simd<float, 16> v1) {
    return _mm512_ceil_ps(v1.vec_);
}

force_inline fixed_size_simd<float, 16> vectorcall operator&(const fixed_size_simd<float, 16> v1,
                                                             const fixed_size_simd<float, 16> v2) {
    return _mm512_and_ps(v1.vec_, v2.vec_);
}

force_inline fixed_size_simd<float, 16> vectorcall operator|(const fixed_size_simd<float, 16> v1,
                                                             const fixed_size_simd<float, 16> v2) {
    return _mm512_or_ps(v1.vec_, v2.vec_);
}

force_inline fixed_size_simd<float, 16> vectorcall operator^(const fixed_size_simd<float, 16> v1,
                                                             const fixed_size_simd<float, 16> v2) {
    return _mm512_xor_ps(v1.vec_, v2.vec_);
}

force_inline fixed_size_simd<float, 16> vectorcall operator+(const fixed_size_simd<float, 16> v1,
                                                             const fixed_size_simd<float, 16> v2) {
    return _mm512_add_ps(v1.vec_, v2.vec_);
}

force_inline fixed_size_simd<float, 16> vectorcall operator-(const fixed_size_simd<float, 16> v1,
                                                             const fixed_size_simd<float, 16> v2) {
    return _mm512_sub_ps(v1.vec_, v2.vec_);
}

force_inline fixed_size_simd<float, 16> vectorcall operator*(const fixed_size_simd<float, 16> v1,
                                                             const fixed_size_simd<float, 16> v2) {
    return _mm512_mul_ps(v1.vec_, v2.vec_);
}

force_inline fixed_size_simd<float, 16> vectorcall operator/(const fixed_size_simd<float, 16> v1,
                                                             const fixed_size_simd<float, 16> v2) {
    return _mm512_div_ps(v1.vec_, v2.vec_);
}

force_inline fixed_size_simd<float, 16> vectorcall operator<(const fixed_size_simd<float, 16> v1,
                                                             const fixed_size_simd<float, 16> v2) {
    return _mm512_cmp_ps(v1.vec_, v2.vec_, _CMP_LT_OS);
}

force_inline fixed_size_simd<float, 16> vectorcall operator<=(const fixed_size_simd<float, 16> v1,
                                                              const fixed_size_simd<float, 16> v2) {
    return _mm512_cmp_ps(v1.vec_, v2.vec_, _CMP_LE_OS);
}

force_inline fixed_size_simd<float, 16> vectorcall operator>(const fixed_size_simd<float, 16> v1,
                                                             const fixed_size_simd<float, 16> v2) {
    return _mm512_cmp_ps(v1.vec_, v2.vec_, _CMP_GT_OS);
}

force_inline fixed_size_simd<float, 16> vectorcall operator>=(const fixed_size_simd<float, 16> v1,
                                                              const fixed_size_simd<float, 16> v2) {
    return _mm512_cmp_ps(v1.vec_, v2.vec_, _CMP_GE_OS);
}

force_inline fixed_size_simd<float, 16> vectorcall operator==(const fixed_size_simd<float, 16> v1,
                                                              const fixed_size_simd<float, 16> v2) {
    return _mm512_cmp_ps(v1.vec_, v2.vec_, _CMP_EQ_OS);
}

force_inline fixed_size_simd<float, 16> vectorcall operator!=(const fixed_size_simd<float, 16> v1,
                                                              const fixed_size_simd<float, 16> v2) {
    return _mm512_cmp_ps(v1.vec_, v2.vec_, _CMP_NEQ_OS);
}

force_inline fixed_size_simd<float, 16> vectorcall clamp(const fixed_size_simd<float, 16> v1,
                                                         const fixed_size_simd<float, 16> min,
                                                         const fixed_size_simd<float, 16> max) {
    return _mm512_max_ps(min.vec_, _mm512_min_ps(v1.vec_, max.vec_));
}

inline fixed_size_simd<float, 16> vectorcall pow(const fixed_size_simd<float, 16> v1,
                                                 const fixed_size_simd<float, 16> v2) {
    alignas(64) float comp1[16], comp2[16];
    _mm512_store_ps(comp1, v1.vec_);
    _mm512_store_ps(comp2, v2.vec_);
    UNROLLED_FOR(i, 16, { comp1[i] = powf(comp1[i], comp2[i]); })
    return fixed_size_simd<float, 16>{comp1, vector_aligned};
}

inline fixed_size_simd<float, 16> vectorcall exp(const fixed_size_simd<float, 16> v1) {
    alignas(64) float comp1[16];
    _mm512_store_ps(comp1, v1.vec_);
    UNROLLED_FOR(i, 16, { comp1[i] = expf(comp1[i]); })
    return fixed_size_simd<float, 16>{comp1, vector_aligned};
}

force_inline fixed_size_simd<float, 16> vectorcall normalize(const fixed_size_simd<float, 16> v1) {
    return v1 / v1.length();
}

force_inline fixed_size_simd<float, 16> vectorcall normalize_len(const fixed_size_simd<float, 16> v1, float &out_len) {
    return v1 / (out_len = v1.length());
}

force_inline fixed_size_simd<float, 16> vectorcall inclusive_scan(fixed_size_simd<float, 16> v1) {
    v1.vec_ = _mm512_add_ps(v1.vec_, _mm512_castsi512_ps(_mm512_slli_si512(_mm512_castps_si512(v1.vec_), 1)));
    v1.vec_ = _mm512_add_ps(v1.vec_, _mm512_castsi512_ps(_mm512_slli_si512(_mm512_castps_si512(v1.vec_), 2)));
    v1.vec_ = _mm512_add_ps(v1.vec_, _mm512_castsi512_ps(_mm512_slli_si512(_mm512_castps_si512(v1.vec_), 4)));
    v1.vec_ = _mm512_add_ps(v1.vec_, _mm512_castsi512_ps(_mm512_slli_si512(_mm512_castps_si512(v1.vec_), 8)));
    return v1;
}

force_inline fixed_size_simd<int, 16> vectorcall inclusive_scan(fixed_size_simd<int, 16> v1) {
    v1.vec_ = _mm512_add_epi32(v1.vec_, _mm512_slli_si512(v1.vec_, 1));
    v1.vec_ = _mm512_add_epi32(v1.vec_, _mm512_slli_si512(v1.vec_, 2));
    v1.vec_ = _mm512_add_epi32(v1.vec_, _mm512_slli_si512(v1.vec_, 4));
    v1.vec_ = _mm512_add_epi32(v1.vec_, _mm512_slli_si512(v1.vec_, 8));
    return v1;
}

force_inline fixed_size_simd<unsigned, 16> vectorcall inclusive_scan(fixed_size_simd<unsigned, 16> v1) {
    v1.vec_ = _mm512_add_epi32(v1.vec_, _mm512_slli_si512(v1.vec_, 1));
    v1.vec_ = _mm512_add_epi32(v1.vec_, _mm512_slli_si512(v1.vec_, 2));
    v1.vec_ = _mm512_add_epi32(v1.vec_, _mm512_slli_si512(v1.vec_, 4));
    v1.vec_ = _mm512_add_epi32(v1.vec_, _mm512_slli_si512(v1.vec_, 8));
    return v1;
}

force_inline fixed_size_simd<float, 16> vectorcall copysign(const fixed_size_simd<float, 16> val,
                                                            const fixed_size_simd<float, 16> sign) {
    const __m512 sign_mask = _mm512_and_ps(sign.vec_, _mm512_set1_ps(-0.0f));
    const __m512 abs_val = _mm512_andnot_ps(sign_mask, val.vec_);
    return _mm512_or_ps(abs_val, sign_mask);
}

force_inline fixed_size_simd<float, 16> vectorcall fmadd(const fixed_size_simd<float, 16> a,
                                                         const fixed_size_simd<float, 16> b,
                                                         const fixed_size_simd<float, 16> c) {
    return _mm512_fmadd_ps(a.vec_, b.vec_, c.vec_);
}

force_inline fixed_size_simd<float, 16> vectorcall fmsub(const fixed_size_simd<float, 16> a,
                                                         const fixed_size_simd<float, 16> b,
                                                         const fixed_size_simd<float, 16> c) {
    return _mm512_fmsub_ps(a.vec_, b.vec_, c.vec_);
}

force_inline fixed_size_simd<float, 16> vectorcall gather(const float *base_addr,
                                                          const fixed_size_simd<int, 16> vindex) {
    return _mm512_i32gather_ps(vindex.vec_, base_addr, sizeof(float));
}

force_inline fixed_size_simd<int, 16> vectorcall gather(const int *base_addr, const fixed_size_simd<int, 16> vindex) {
    return _mm512_i32gather_epi32(vindex.vec_, base_addr, sizeof(int));
}

force_inline fixed_size_simd<unsigned, 16> vectorcall gather(const unsigned *base_addr,
                                                             const fixed_size_simd<int, 16> vindex) {
    return _mm512_i32gather_epi32(vindex.vec_, reinterpret_cast<const int *>(base_addr), sizeof(unsigned));
}

force_inline void vectorcall scatter(float *base_addr, fixed_size_simd<int, 16> vindex, fixed_size_simd<float, 16> v) {
    _mm512_i32scatter_ps(base_addr, vindex.vec_, v.vec_, sizeof(float));
}

force_inline void vectorcall scatter(float *base_addr, fixed_size_simd<int, 16> mask, fixed_size_simd<int, 16> vindex,
                                     fixed_size_simd<float, 16> v) {
    _mm512_mask_i32scatter_ps(base_addr, mask.movemask(), vindex.vec_, v.vec_, sizeof(float));
}

force_inline void vectorcall scatter(int *base_addr, fixed_size_simd<int, 16> vindex, fixed_size_simd<int, 16> v) {
    _mm512_i32scatter_epi32(base_addr, vindex.vec_, v.vec_, sizeof(int));
}

force_inline void vectorcall scatter(int *base_addr, fixed_size_simd<int, 16> mask, fixed_size_simd<int, 16> vindex,
                                     fixed_size_simd<int, 16> v) {
    _mm512_mask_i32scatter_epi32(base_addr, mask.movemask(), vindex.vec_, v.vec_, sizeof(int));
}

force_inline void vectorcall scatter(unsigned *base_addr, fixed_size_simd<int, 16> vindex,
                                     fixed_size_simd<unsigned, 16> v) {
    _mm512_i32scatter_epi32(base_addr, vindex.vec_, v.vec_, sizeof(unsigned));
}

force_inline void vectorcall scatter(unsigned *base_addr, fixed_size_simd<int, 16> mask,
                                     fixed_size_simd<int, 16> vindex, fixed_size_simd<unsigned, 16> v) {
    _mm512_mask_i32scatter_epi32(base_addr, mask.movemask(), vindex.vec_, v.vec_, sizeof(int));
}

template <typename U>
force_inline fixed_size_simd<float, 16> vectorcall select(const fixed_size_simd<U, 16> mask,
                                                          const fixed_size_simd<float, 16> vec1,
                                                          const fixed_size_simd<float, 16> vec2) {
    validate_mask(mask);
    return _mm512_blendv_ps(vec2.vec_, vec1.vec_, _mm_cast<__m512>(mask.vec_));
}

template <typename U>
force_inline fixed_size_simd<int, 16> vectorcall select(const fixed_size_simd<U, 16> mask,
                                                        const fixed_size_simd<int, 16> vec1,
                                                        const fixed_size_simd<int, 16> vec2) {
    validate_mask(mask);
    return _mm512_ternarylogic_epi32(vec2.vec_, vec1.vec_, _mm512_srai_epi32(_mm_cast<__m512i>(mask.vec_), 31), 0xd8);
}

template <typename U>
force_inline fixed_size_simd<unsigned, 16> vectorcall select(const fixed_size_simd<U, 16> mask,
                                                             const fixed_size_simd<unsigned, 16> vec1,
                                                             const fixed_size_simd<unsigned, 16> vec2) {
    validate_mask(mask);
    return _mm512_ternarylogic_epi32(vec2.vec_, vec1.vec_, _mm512_srai_epi32(_mm_cast<__m512i>(mask.vec_), 31), 0xd8);
}

} // namespace NS
} // namespace Ray

#undef validate_mask

#pragma warning(pop)
