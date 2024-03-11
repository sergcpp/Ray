// #pragma once

#include "simd_avx.h"

#include <immintrin.h>

#define _mm512_cmp_ps(a, b, c) _mm512_castsi512_ps(_mm512_movm_epi32(_mm512_cmp_ps_mask(a, b, c)))

#define _mm512_blendv_ps(a, b, m)                                                                                      \
    _mm512_castsi512_ps(_mm512_ternarylogic_epi32(_mm512_castps_si512(a), _mm512_castps_si512(b),                      \
                                                  _mm512_srai_epi32(_mm512_castps_si512(m), 31), 0xd8))

#define _mm512_movemask_epi32(a)                                                                                       \
    (int)_mm512_cmpneq_epi32_mask(_mm512_setzero_si512(), _mm512_and_si512(_mm512_set1_epi32(0x80000000U), a))

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

template <> class fixed_size_simd<float, 16> {
    union {
        __m512 vec_;
        float comp_[16];
    };

    friend class fixed_size_simd<int, 16>;
    friend class fixed_size_simd<unsigned, 16>;

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
        float temp = 0;
        UNROLLED_FOR(i, 16, { temp += comp_[i] * comp_[i]; })
        return temp;
    }

    force_inline float hsum() const { return _mm512_reduce_add_ps(vec_); }

    force_inline void store_to(float *f) const { _mm512_storeu_ps(f, vec_); }
    force_inline void store_to(float *f, vector_aligned_tag) const { _mm512_store_ps(f, vec_); }

    force_inline void vectorcall blend_to(const fixed_size_simd<float, 16> mask, const fixed_size_simd<float, 16> v1) {
        validate_mask(mask);
        //__mmask16 msk =
        //    _mm512_fpclass_ps_mask(mask.vec_, 0x54); // 0x54 = Negative_Finite | Negative_Infinity | Negative_Zero
        // vec_ = _mm512_mask_blend_ps(msk, vec_, v1.vec_);
        vec_ = _mm512_blendv_ps(vec_, v1.vec_, mask.vec_);
    }

    force_inline void vectorcall blend_inv_to(const fixed_size_simd<float, 16> mask,
                                              const fixed_size_simd<float, 16> v1) {
        validate_mask(mask);
        //__mmask16 msk =
        //    _mm512_fpclass_ps_mask(mask.vec_, 0x54); // 0x54 = Negative_Finite | Negative_Infinity | Negative_Zero
        // vec_ = _mm512_mask_blend_ps(msk, v1.vec_, vec_);
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
    friend force_inline fixed_size_simd<float, 16> vectorcall normalize(fixed_size_simd<float, 16> v1);
    friend force_inline fixed_size_simd<float, 16> vectorcall normalize_len(fixed_size_simd<float, 16> v1,
                                                                            float &out_len);
    friend force_inline fixed_size_simd<float, 16> vectorcall inclusive_scan(fixed_size_simd<float, 16> v1);

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

    fixed_size_simd<int, 16> &vectorcall operator*=(const fixed_size_simd<int, 16> rhs) {
        UNROLLED_FOR(i, 16, { comp_[i] *= rhs.comp_[i]; })
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

    force_inline fixed_size_simd<int, 16> operator-() const {
        fixed_size_simd<int, 16> temp;
        temp.vec_ = _mm512_sub_epi32(_mm512_setzero_si512(), vec_);
        return temp;
    }

    force_inline fixed_size_simd<int, 16> vectorcall operator==(const fixed_size_simd<int, 16> rhs) const {
        fixed_size_simd<int, 16> ret;
        ret.vec_ = _mm512_movm_epi32(_mm512_cmpeq_epi32_mask(vec_, rhs.vec_));
        return ret;
    }

    force_inline fixed_size_simd<int, 16> vectorcall operator!=(const fixed_size_simd<int, 16> rhs) const {
        fixed_size_simd<int, 16> ret;
        ret.vec_ =
            _mm512_andnot_si512(_mm512_movm_epi32(_mm512_cmpeq_epi32_mask(vec_, rhs.vec_)), _mm512_set1_epi32(~0));
        return ret;
    }

    force_inline fixed_size_simd<int, 16> &vectorcall operator&=(const fixed_size_simd<int, 16> rhs) {
        vec_ = _mm512_and_si512(vec_, rhs.vec_);
        return *this;
    }

    force_inline explicit operator fixed_size_simd<float, 16>() const {
        fixed_size_simd<float, 16> ret;
        ret.vec_ = _mm512_cvtepi32_ps(vec_);
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

    force_inline bool vectorcall all_zeros() const {
        return _mm512_cmpeq_epi32_mask(vec_, _mm512_setzero_si512()) == 0xFFFF;
    }

    force_inline bool vectorcall all_zeros(const fixed_size_simd<int, 16> mask) const {
        return _mm512_cmpeq_epi32_mask(_mm512_and_si512(vec_, mask.vec_), _mm512_setzero_si512()) == 0xFFFF;
    }

    force_inline bool not_all_zeros() const { return !all_zeros(); }

    friend force_inline fixed_size_simd<int, 16> vectorcall min(const fixed_size_simd<int, 16> v1,
                                                                const fixed_size_simd<int, 16> v2) {
        fixed_size_simd<int, 16> temp;
        temp.vec_ = _mm512_min_epi32(v1.vec_, v2.vec_);
        return temp;
    }

    friend force_inline fixed_size_simd<int, 16> vectorcall max(const fixed_size_simd<int, 16> v1,
                                                                const fixed_size_simd<int, 16> v2) {
        fixed_size_simd<int, 16> temp;
        temp.vec_ = _mm512_max_epi32(v1.vec_, v2.vec_);
        return temp;
    }

    friend force_inline fixed_size_simd<int, 16> vectorcall clamp(const fixed_size_simd<int, 16> v1,
                                                                  const fixed_size_simd<int, 16> _min,
                                                                  const fixed_size_simd<int, 16> _max) {
        return max(_min, min(v1, _max));
    }

    force_inline static fixed_size_simd<int, 16> vectorcall and_not(const fixed_size_simd<int, 16> v1,
                                                                    const fixed_size_simd<int, 16> v2) {
        fixed_size_simd<int, 16> temp;
        temp.vec_ = _mm512_castps_si512(_mm512_andnot_ps(_mm512_castsi512_ps(v1.vec_), _mm512_castsi512_ps(v2.vec_)));
        return temp;
    }

    friend force_inline fixed_size_simd<int, 16> vectorcall operator&(const fixed_size_simd<int, 16> v1,
                                                                      const fixed_size_simd<int, 16> v2) {
        fixed_size_simd<int, 16> temp;
        temp.vec_ = _mm512_castps_si512(_mm512_and_ps(_mm512_castsi512_ps(v1.vec_), _mm512_castsi512_ps(v2.vec_)));
        return temp;
    }

    friend force_inline fixed_size_simd<int, 16> vectorcall operator|(const fixed_size_simd<int, 16> v1,
                                                                      const fixed_size_simd<int, 16> v2) {
        fixed_size_simd<int, 16> temp;
        temp.vec_ = _mm512_castps_si512(_mm512_or_ps(_mm512_castsi512_ps(v1.vec_), _mm512_castsi512_ps(v2.vec_)));
        return temp;
    }

    friend force_inline fixed_size_simd<int, 16> vectorcall operator^(const fixed_size_simd<int, 16> v1,
                                                                      const fixed_size_simd<int, 16> v2) {
        fixed_size_simd<int, 16> temp;
        temp.vec_ = _mm512_castps_si512(_mm512_xor_ps(_mm512_castsi512_ps(v1.vec_), _mm512_castsi512_ps(v2.vec_)));
        return temp;
    }

    friend force_inline fixed_size_simd<int, 16> vectorcall operator+(const fixed_size_simd<int, 16> v1,
                                                                      const fixed_size_simd<int, 16> v2) {
        fixed_size_simd<int, 16> temp;
        temp.vec_ = _mm512_add_epi32(v1.vec_, v2.vec_);
        return temp;
    }

    friend force_inline fixed_size_simd<int, 16> vectorcall operator-(const fixed_size_simd<int, 16> v1,
                                                                      const fixed_size_simd<int, 16> v2) {
        fixed_size_simd<int, 16> temp;
        temp.vec_ = _mm512_sub_epi32(v1.vec_, v2.vec_);
        return temp;
    }

    friend fixed_size_simd<int, 16> vectorcall operator*(const fixed_size_simd<int, 16> v1,
                                                         const fixed_size_simd<int, 16> v2) {
        fixed_size_simd<int, 16> ret;
        UNROLLED_FOR(i, 16, { ret.comp_[i] = v1.comp_[i] * v2.comp_[i]; })
        return ret;
    }

    friend fixed_size_simd<int, 16> vectorcall operator/(const fixed_size_simd<int, 16> v1,
                                                         const fixed_size_simd<int, 16> v2) {
        fixed_size_simd<int, 16> ret;
        UNROLLED_FOR(i, 16, { ret.comp_[i] = v1.comp_[i] / v2.comp_[i]; })
        return ret;
    }

    friend force_inline fixed_size_simd<int, 16> vectorcall operator<(const fixed_size_simd<int, 16> v1,
                                                                      const fixed_size_simd<int, 16> v2) {
        fixed_size_simd<int, 16> ret;
        ret.vec_ = _mm512_movm_epi32(_mm512_cmpgt_epi32_mask(v2.vec_, v1.vec_));
        return ret;
    }

    friend force_inline fixed_size_simd<int, 16> vectorcall operator>(const fixed_size_simd<int, 16> v1,
                                                                      const fixed_size_simd<int, 16> v2) {
        fixed_size_simd<int, 16> ret;
        ret.vec_ = _mm512_movm_epi32(_mm512_cmpgt_epi32_mask(v1.vec_, v2.vec_));
        return ret;
    }

    friend force_inline fixed_size_simd<int, 16> vectorcall operator>=(const fixed_size_simd<int, 16> v1,
                                                                       const fixed_size_simd<int, 16> v2) {
        fixed_size_simd<int, 16> ret;
        ret.vec_ = _mm512_movm_epi32(_mm512_cmpge_epi32_mask(v1.vec_, v2.vec_));
        return ret;
    }

    friend force_inline fixed_size_simd<int, 16> vectorcall operator>>(const fixed_size_simd<int, 16> v1,
                                                                       const fixed_size_simd<int, 16> v2) {
        fixed_size_simd<int, 16> ret;
        ret.vec_ = _mm512_srlv_epi32(v1.vec_, v2.vec_);
        return ret;
    }

    friend force_inline fixed_size_simd<int, 16> vectorcall operator>>(const fixed_size_simd<int, 16> v1,
                                                                       const int v2) {
        fixed_size_simd<int, 16> ret;
        ret.vec_ = _mm512_srli_epi32(v1.vec_, v2);
        return ret;
    }

    friend force_inline fixed_size_simd<int, 16> vectorcall operator<<(const fixed_size_simd<int, 16> v1,
                                                                       const fixed_size_simd<int, 16> v2) {
        fixed_size_simd<int, 16> ret;
        ret.vec_ = _mm512_sllv_epi32(v1.vec_, v2.vec_);
        return ret;
    }

    friend force_inline fixed_size_simd<int, 16> vectorcall operator<<(const fixed_size_simd<int, 16> v1,
                                                                       const int v2) {
        fixed_size_simd<int, 16> ret;
        ret.vec_ = _mm512_slli_epi32(v1.vec_, v2);
        return ret;
    }

    force_inline fixed_size_simd<int, 16> operator~() const {
        fixed_size_simd<int, 16> ret;
        ret.vec_ = _mm512_andnot_si512(vec_, _mm512_set1_epi32(~0));
        return ret;
    }

    friend force_inline fixed_size_simd<int, 16> vectorcall srai(const fixed_size_simd<int, 16> v1, const int v2) {
        fixed_size_simd<int, 16> ret;
        ret.vec_ = _mm512_srai_epi32(v1.vec_, v2);
        return ret;
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

    fixed_size_simd<unsigned, 16> &vectorcall operator*=(const fixed_size_simd<unsigned, 16> rhs) {
        UNROLLED_FOR(i, 16, { comp_[i] *= rhs.comp_[i]; })
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
        fixed_size_simd<unsigned, 16> ret;
        ret.vec_ = _mm512_movm_epi32(_mm512_cmpeq_epi32_mask(vec_, rhs.vec_));
        return ret;
    }

    force_inline fixed_size_simd<unsigned, 16> vectorcall operator!=(const fixed_size_simd<unsigned, 16> rhs) const {
        fixed_size_simd<unsigned, 16> ret;
        ret.vec_ =
            _mm512_andnot_si512(_mm512_movm_epi32(_mm512_cmpeq_epi32_mask(vec_, rhs.vec_)), _mm512_set1_epi32(~0));
        return ret;
    }

    force_inline fixed_size_simd<unsigned, 16> &vectorcall operator&=(const fixed_size_simd<unsigned, 16> rhs) {
        vec_ = _mm512_and_si512(vec_, rhs.vec_);
        return *this;
    }

    force_inline explicit operator fixed_size_simd<float, 16>() const {
        fixed_size_simd<float, 16> ret;
        ret.vec_ = _mm512_cvtepu32_ps(vec_);
        return ret;
    }

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

    force_inline bool vectorcall all_zeros() const {
        return _mm512_cmpeq_epi32_mask(vec_, _mm512_setzero_si512()) == 0xFFFF;
    }

    force_inline bool vectorcall all_zeros(const fixed_size_simd<unsigned, 16> mask) const {
        return _mm512_cmpeq_epi32_mask(_mm512_and_si512(vec_, mask.vec_), _mm512_setzero_si512()) == 0xFFFF;
    }

    force_inline bool not_all_zeros() const { return !all_zeros(); }

    friend force_inline fixed_size_simd<unsigned, 16> vectorcall min(const fixed_size_simd<unsigned, 16> v1,
                                                                     const fixed_size_simd<unsigned, 16> v2) {
        fixed_size_simd<unsigned, 16> temp;
        temp.vec_ = _mm512_min_epu32(v1.vec_, v2.vec_);
        return temp;
    }

    friend force_inline fixed_size_simd<unsigned, 16> vectorcall max(const fixed_size_simd<unsigned, 16> v1,
                                                                     const fixed_size_simd<unsigned, 16> v2) {
        fixed_size_simd<unsigned, 16> temp;
        temp.vec_ = _mm512_max_epu32(v1.vec_, v2.vec_);
        return temp;
    }

    friend force_inline fixed_size_simd<unsigned, 16> vectorcall clamp(const fixed_size_simd<unsigned, 16> v1,
                                                                       const fixed_size_simd<unsigned, 16> _min,
                                                                       const fixed_size_simd<unsigned, 16> _max) {
        return max(_min, min(v1, _max));
    }

    force_inline static fixed_size_simd<unsigned, 16> vectorcall and_not(const fixed_size_simd<unsigned, 16> v1,
                                                                         const fixed_size_simd<unsigned, 16> v2) {
        fixed_size_simd<unsigned, 16> temp;
        temp.vec_ = _mm512_castps_si512(_mm512_andnot_ps(_mm512_castsi512_ps(v1.vec_), _mm512_castsi512_ps(v2.vec_)));
        return temp;
    }

    friend force_inline fixed_size_simd<unsigned, 16> vectorcall operator&(const fixed_size_simd<unsigned, 16> v1,
                                                                           const fixed_size_simd<unsigned, 16> v2) {
        fixed_size_simd<unsigned, 16> temp;
        temp.vec_ = _mm512_castps_si512(_mm512_and_ps(_mm512_castsi512_ps(v1.vec_), _mm512_castsi512_ps(v2.vec_)));
        return temp;
    }

    friend force_inline fixed_size_simd<unsigned, 16> vectorcall operator|(const fixed_size_simd<unsigned, 16> v1,
                                                                           const fixed_size_simd<unsigned, 16> v2) {
        fixed_size_simd<unsigned, 16> temp;
        temp.vec_ = _mm512_castps_si512(_mm512_or_ps(_mm512_castsi512_ps(v1.vec_), _mm512_castsi512_ps(v2.vec_)));
        return temp;
    }

    friend force_inline fixed_size_simd<unsigned, 16> vectorcall operator^(const fixed_size_simd<unsigned, 16> v1,
                                                                           const fixed_size_simd<unsigned, 16> v2) {
        fixed_size_simd<unsigned, 16> temp;
        temp.vec_ = _mm512_castps_si512(_mm512_xor_ps(_mm512_castsi512_ps(v1.vec_), _mm512_castsi512_ps(v2.vec_)));
        return temp;
    }

    friend force_inline fixed_size_simd<unsigned, 16> vectorcall operator+(const fixed_size_simd<unsigned, 16> v1,
                                                                           const fixed_size_simd<unsigned, 16> v2) {
        fixed_size_simd<unsigned, 16> temp;
        temp.vec_ = _mm512_add_epi32(v1.vec_, v2.vec_);
        return temp;
    }

    friend force_inline fixed_size_simd<unsigned, 16> vectorcall operator-(const fixed_size_simd<unsigned, 16> v1,
                                                                           const fixed_size_simd<unsigned, 16> v2) {
        fixed_size_simd<unsigned, 16> temp;
        temp.vec_ = _mm512_sub_epi32(v1.vec_, v2.vec_);
        return temp;
    }

    friend fixed_size_simd<unsigned, 16> vectorcall operator*(const fixed_size_simd<unsigned, 16> v1,
                                                              const fixed_size_simd<unsigned, 16> v2) {
        fixed_size_simd<unsigned, 16> ret;
        UNROLLED_FOR(i, 16, { ret.comp_[i] = v1.comp_[i] * v2.comp_[i]; })
        return ret;
    }

    friend fixed_size_simd<unsigned, 16> vectorcall operator/(const fixed_size_simd<unsigned, 16> v1,
                                                              const fixed_size_simd<unsigned, 16> v2) {
        fixed_size_simd<unsigned, 16> ret;
        UNROLLED_FOR(i, 16, { ret.comp_[i] = v1.comp_[i] / v2.comp_[i]; })
        return ret;
    }

    friend force_inline fixed_size_simd<unsigned, 16> vectorcall operator>>(const fixed_size_simd<unsigned, 16> v1,
                                                                            const fixed_size_simd<unsigned, 16> v2) {
        fixed_size_simd<unsigned, 16> ret;
        ret.vec_ = _mm512_srlv_epi32(v1.vec_, v2.vec_);
        return ret;
    }

    friend force_inline fixed_size_simd<unsigned, 16> vectorcall operator>>(const fixed_size_simd<unsigned, 16> v1,
                                                                            const unsigned v2) {
        fixed_size_simd<unsigned, 16> ret;
        ret.vec_ = _mm512_srli_epi32(v1.vec_, v2);
        return ret;
    }

    friend force_inline fixed_size_simd<unsigned, 16> vectorcall operator<<(const fixed_size_simd<unsigned, 16> v1,
                                                                            const fixed_size_simd<unsigned, 16> v2) {
        fixed_size_simd<unsigned, 16> ret;
        ret.vec_ = _mm512_sllv_epi32(v1.vec_, v2.vec_);
        return ret;
    }

    friend force_inline fixed_size_simd<unsigned, 16> vectorcall operator<<(const fixed_size_simd<unsigned, 16> v1,
                                                                            const unsigned v2) {
        fixed_size_simd<unsigned, 16> ret;
        ret.vec_ = _mm512_slli_epi32(v1.vec_, v2);
        return ret;
    }

    force_inline fixed_size_simd<unsigned, 16> operator~() const {
        fixed_size_simd<unsigned, 16> ret;
        ret.vec_ = _mm512_andnot_si512(vec_, _mm512_set1_epi32(~0));
        return ret;
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

force_inline fixed_size_simd<float, 16> fixed_size_simd<float, 16>::operator~() const {
    fixed_size_simd<float, 16> ret;
    ret.vec_ = _mm512_castsi512_ps(_mm512_andnot_si512(_mm512_castps_si512(vec_), _mm512_set1_epi32(~0)));
    return ret;
}

force_inline fixed_size_simd<float, 16> fixed_size_simd<float, 16>::operator-() const {
    fixed_size_simd<float, 16> temp;
    __m512 m = _mm512_set1_ps(-0.0f);
    temp.vec_ = _mm512_xor_ps(vec_, m);
    return temp;
}

force_inline fixed_size_simd<float, 16>::operator fixed_size_simd<int, 16>() const {
    fixed_size_simd<int, 16> ret;
    ret.vec_ = _mm512_cvttps_epi32(vec_);
    return ret;
}

force_inline fixed_size_simd<float, 16>::operator fixed_size_simd<unsigned, 16>() const {
    fixed_size_simd<unsigned, 16> ret;
    ret.vec_ = _mm512_cvttps_epi32(vec_);
    return ret;
}

force_inline fixed_size_simd<int, 16>::operator fixed_size_simd<unsigned, 16>() const {
    fixed_size_simd<unsigned, 16> ret;
    ret.vec_ = vec_;
    return ret;
}

force_inline fixed_size_simd<float, 16> fixed_size_simd<float, 16>::sqrt() const {
    fixed_size_simd<float, 16> temp;
    temp.vec_ = _mm512_sqrt_ps(vec_);
    return temp;
}

inline fixed_size_simd<float, 16> fixed_size_simd<float, 16>::log() const {
    fixed_size_simd<float, 16> ret;
    UNROLLED_FOR(i, 16, { ret.comp_[i] = logf(comp_[i]); })
    return ret;
}

force_inline fixed_size_simd<float, 16> vectorcall min(const fixed_size_simd<float, 16> v1,
                                                       const fixed_size_simd<float, 16> v2) {
    fixed_size_simd<float, 16> temp;
    temp.vec_ = _mm512_min_ps(v1.vec_, v2.vec_);
    return temp;
}

force_inline fixed_size_simd<float, 16> vectorcall max(const fixed_size_simd<float, 16> v1,
                                                       const fixed_size_simd<float, 16> v2) {
    fixed_size_simd<float, 16> temp;
    temp.vec_ = _mm512_max_ps(v1.vec_, v2.vec_);
    return temp;
}

force_inline fixed_size_simd<float, 16> vectorcall and_not(const fixed_size_simd<float, 16> v1,
                                                           const fixed_size_simd<float, 16> v2) {
    fixed_size_simd<float, 16> temp;
    temp.vec_ = _mm512_andnot_ps(v1.vec_, v2.vec_);
    return temp;
}

force_inline fixed_size_simd<float, 16> vectorcall floor(const fixed_size_simd<float, 16> v1) {
    fixed_size_simd<float, 16> temp;
    temp.vec_ = _mm512_floor_ps(v1.vec_);
    return temp;
}

force_inline fixed_size_simd<float, 16> vectorcall ceil(const fixed_size_simd<float, 16> v1) {
    fixed_size_simd<float, 16> temp;
    temp.vec_ = _mm512_ceil_ps(v1.vec_);
    return temp;
}

force_inline fixed_size_simd<float, 16> vectorcall operator&(const fixed_size_simd<float, 16> v1,
                                                             const fixed_size_simd<float, 16> v2) {
    fixed_size_simd<float, 16> temp;
    temp.vec_ = _mm512_and_ps(v1.vec_, v2.vec_);
    return temp;
}

force_inline fixed_size_simd<float, 16> vectorcall operator|(const fixed_size_simd<float, 16> v1,
                                                             const fixed_size_simd<float, 16> v2) {
    fixed_size_simd<float, 16> temp;
    temp.vec_ = _mm512_or_ps(v1.vec_, v2.vec_);
    return temp;
}

force_inline fixed_size_simd<float, 16> vectorcall operator^(const fixed_size_simd<float, 16> v1,
                                                             const fixed_size_simd<float, 16> v2) {
    fixed_size_simd<float, 16> temp;
    temp.vec_ = _mm512_xor_ps(v1.vec_, v2.vec_);
    return temp;
}

force_inline fixed_size_simd<float, 16> vectorcall operator+(const fixed_size_simd<float, 16> v1,
                                                             const fixed_size_simd<float, 16> v2) {
    fixed_size_simd<float, 16> temp;
    temp.vec_ = _mm512_add_ps(v1.vec_, v2.vec_);
    return temp;
}

force_inline fixed_size_simd<float, 16> vectorcall operator-(const fixed_size_simd<float, 16> v1,
                                                             const fixed_size_simd<float, 16> v2) {
    fixed_size_simd<float, 16> temp;
    temp.vec_ = _mm512_sub_ps(v1.vec_, v2.vec_);
    return temp;
}

force_inline fixed_size_simd<float, 16> vectorcall operator*(const fixed_size_simd<float, 16> v1,
                                                             const fixed_size_simd<float, 16> v2) {
    fixed_size_simd<float, 16> temp;
    temp.vec_ = _mm512_mul_ps(v1.vec_, v2.vec_);
    return temp;
}

force_inline fixed_size_simd<float, 16> vectorcall operator/(const fixed_size_simd<float, 16> v1,
                                                             const fixed_size_simd<float, 16> v2) {
    fixed_size_simd<float, 16> temp;
    temp.vec_ = _mm512_div_ps(v1.vec_, v2.vec_);
    return temp;
}

force_inline fixed_size_simd<float, 16> vectorcall operator<(const fixed_size_simd<float, 16> v1,
                                                             const fixed_size_simd<float, 16> v2) {
    fixed_size_simd<float, 16> ret;
    ret.vec_ = _mm512_cmp_ps(v1.vec_, v2.vec_, _CMP_LT_OS);
    return ret;
}

force_inline fixed_size_simd<float, 16> vectorcall operator<=(const fixed_size_simd<float, 16> v1,
                                                              const fixed_size_simd<float, 16> v2) {
    fixed_size_simd<float, 16> ret;
    ret.vec_ = _mm512_cmp_ps(v1.vec_, v2.vec_, _CMP_LE_OS);
    return ret;
}

force_inline fixed_size_simd<float, 16> vectorcall operator>(const fixed_size_simd<float, 16> v1,
                                                             const fixed_size_simd<float, 16> v2) {
    fixed_size_simd<float, 16> ret;
    ret.vec_ = _mm512_cmp_ps(v1.vec_, v2.vec_, _CMP_GT_OS);
    return ret;
}

force_inline fixed_size_simd<float, 16> vectorcall operator>=(const fixed_size_simd<float, 16> v1,
                                                              const fixed_size_simd<float, 16> v2) {
    fixed_size_simd<float, 16> ret;
    ret.vec_ = _mm512_cmp_ps(v1.vec_, v2.vec_, _CMP_GE_OS);
    return ret;
}

force_inline fixed_size_simd<float, 16> vectorcall operator==(const fixed_size_simd<float, 16> v1,
                                                              const fixed_size_simd<float, 16> v2) {
    fixed_size_simd<float, 16> ret;
    ret.vec_ = _mm512_cmp_ps(v1.vec_, v2.vec_, _CMP_EQ_OS);
    return ret;
}

force_inline fixed_size_simd<float, 16> vectorcall operator!=(const fixed_size_simd<float, 16> v1,
                                                              const fixed_size_simd<float, 16> v2) {
    fixed_size_simd<float, 16> ret;
    ret.vec_ = _mm512_cmp_ps(v1.vec_, v2.vec_, _CMP_NEQ_OS);
    return ret;
}

force_inline fixed_size_simd<float, 16> vectorcall clamp(const fixed_size_simd<float, 16> v1,
                                                         const fixed_size_simd<float, 16> min,
                                                         const fixed_size_simd<float, 16> max) {
    fixed_size_simd<float, 16> ret;
    ret.vec_ = _mm512_max_ps(min.vec_, _mm512_min_ps(v1.vec_, max.vec_));
    return ret;
}

inline fixed_size_simd<float, 16> vectorcall pow(const fixed_size_simd<float, 16> v1,
                                                 const fixed_size_simd<float, 16> v2) {
    alignas(64) float comp1[16], comp2[16];
    _mm512_store_ps(comp1, v1.vec_);
    _mm512_store_ps(comp2, v2.vec_);
    UNROLLED_FOR(i, 16, { comp1[i] = powf(comp1[i], comp2[i]); })
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

force_inline fixed_size_simd<float, 16> vectorcall fmadd(const fixed_size_simd<float, 16> a,
                                                         const fixed_size_simd<float, 16> b,
                                                         const fixed_size_simd<float, 16> c) {
    fixed_size_simd<float, 16> ret;
    ret.vec_ = _mm512_fmadd_ps(a.vec_, b.vec_, c.vec_);
    return ret;
}

force_inline fixed_size_simd<float, 16> vectorcall fmsub(const fixed_size_simd<float, 16> a,
                                                         const fixed_size_simd<float, 16> b,
                                                         const fixed_size_simd<float, 16> c) {
    fixed_size_simd<float, 16> ret;
    ret.vec_ = _mm512_fmsub_ps(a.vec_, b.vec_, c.vec_);
    return ret;
}

force_inline fixed_size_simd<float, 16> vectorcall gather(const float *base_addr,
                                                          const fixed_size_simd<int, 16> vindex) {
    fixed_size_simd<float, 16> ret;
    ret.vec_ = _mm512_i32gather_ps(vindex.vec_, base_addr, sizeof(float));
    return ret;
}

force_inline fixed_size_simd<int, 16> vectorcall gather(const int *base_addr, const fixed_size_simd<int, 16> vindex) {
    fixed_size_simd<int, 16> ret;
    ret.vec_ = _mm512_i32gather_epi32(vindex.vec_, base_addr, sizeof(int));
    return ret;
}

force_inline fixed_size_simd<unsigned, 16> vectorcall gather(const unsigned *base_addr,
                                                             const fixed_size_simd<int, 16> vindex) {
    fixed_size_simd<unsigned, 16> ret;
    ret.vec_ = _mm512_i32gather_epi32(vindex.vec_, reinterpret_cast<const int *>(base_addr), sizeof(unsigned));
    return ret;
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
    fixed_size_simd<float, 16> ret;
    ret.vec_ = _mm512_blendv_ps(vec2.vec_, vec1.vec_, _mm_cast<__m512>(mask.vec_));
    return ret;
}

template <typename U>
force_inline fixed_size_simd<int, 16> vectorcall select(const fixed_size_simd<U, 16> mask,
                                                        const fixed_size_simd<int, 16> vec1,
                                                        const fixed_size_simd<int, 16> vec2) {
    validate_mask(mask);
    fixed_size_simd<int, 16> ret;
    ret.vec_ =
        _mm512_ternarylogic_epi32(vec2.vec_, vec1.vec_, _mm512_srai_epi32(_mm_cast<__m512i>(mask.vec_), 31), 0xd8);
    return ret;
}

template <typename U>
force_inline fixed_size_simd<unsigned, 16> vectorcall select(const fixed_size_simd<U, 16> mask,
                                                             const fixed_size_simd<unsigned, 16> vec1,
                                                             const fixed_size_simd<unsigned, 16> vec2) {
    validate_mask(mask);
    fixed_size_simd<unsigned, 16> ret;
    ret.vec_ =
        _mm512_ternarylogic_epi32(vec2.vec_, vec1.vec_, _mm512_srai_epi32(_mm_cast<__m512i>(mask.vec_), 31), 0xd8);
    return ret;
}

} // namespace NS
} // namespace Ray

#undef validate_mask

#pragma warning(pop)
