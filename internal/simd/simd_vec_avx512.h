// #pragma once

#include "simd_vec_avx.h"

#include <immintrin.h>

#ifdef __GNUC__
#pragma GCC push_options
#pragma GCC target("avx512f", "avx512bw", "avx512dq")
#pragma clang attribute push(__attribute__((target("avx512f,avx512bw,avx512dq"))), apply_to = function)
#endif

#define _mm512_cmp_ps(a, b, c) _mm512_castsi512_ps(_mm512_movm_epi32(_mm512_cmp_ps_mask(a, b, c)))

#define _mm512_blendv_ps(a, b, m)                                                                                      \
    _mm512_castsi512_ps(_mm512_ternarylogic_epi32(_mm512_castps_si512(a), _mm512_castps_si512(b),                      \
                                                  _mm512_srai_epi32(_mm512_castps_si512(m), 31), 0xd8))

#define _mm512_movemask_epi32(a)                                                                                       \
    (int)_mm512_cmpneq_epi32_mask(_mm512_setzero_si512(), _mm512_and_si512(_mm512_set1_epi32(0x80000000U), a))

#ifndef NDEBUG
#define validate_mask(m) __assert_valid_mask(m)
#else
#define validate_mask(m) ((void)m)
#endif

#pragma warning(push)
#pragma warning(disable : 4752)

namespace Ray {
namespace NS {

template <> class simd_vec<float, 16> {
  public:
    __m512 vec_;

    friend class simd_vec<int, 16>;

  public:
    force_inline simd_vec() = default;
    force_inline simd_vec(const float f) { vec_ = _mm512_set1_ps(f); }
    force_inline simd_vec(const float f0, const float f1, const float f2, const float f3, const float f4,
                          const float f5, const float f6, const float f7, const float f8, const float f9,
                          const float f10, const float f11, const float f12, const float f13, const float f14,
                          const float f15) {
        vec_ = _mm512_setr_ps(f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15);
    }
    force_inline explicit simd_vec(const float *f) { vec_ = _mm512_loadu_ps(f); }
    force_inline simd_vec(const float *f, simd_mem_aligned_tag) { vec_ = _mm512_load_ps(f); }

    force_inline float operator[](const int i) const {
        __m512 temp = _mm512_maskz_compress_ps(__mmask16(1u << i), vec_);
        return _mm512_cvtss_f32(temp);
    }

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

    force_inline simd_vec<float, 16> &vectorcall operator+=(const simd_vec<float, 16> rhs) {
        vec_ = _mm512_add_ps(vec_, rhs.vec_);
        return *this;
    }

    force_inline simd_vec<float, 16> &vectorcall operator+=(const float rhs) {
        __m512 _rhs = _mm512_set1_ps(rhs);
        vec_ = _mm512_add_ps(vec_, _rhs);
        return *this;
    }

    force_inline simd_vec<float, 16> &vectorcall operator-=(const simd_vec<float, 16> rhs) {
        vec_ = _mm512_sub_ps(vec_, rhs.vec_);
        return *this;
    }

    force_inline simd_vec<float, 16> &vectorcall operator-=(const float rhs) {
        vec_ = _mm512_sub_ps(vec_, _mm512_set1_ps(rhs));
        return *this;
    }

    force_inline simd_vec<float, 16> &vectorcall operator*=(const simd_vec<float, 16> rhs) {
        vec_ = _mm512_mul_ps(vec_, rhs.vec_);
        return *this;
    }

    force_inline simd_vec<float, 16> &vectorcall operator*=(const float rhs) {
        vec_ = _mm512_mul_ps(vec_, _mm512_set1_ps(rhs));
        return *this;
    }

    force_inline simd_vec<float, 16> &vectorcall operator/=(const simd_vec<float, 16> rhs) {
        vec_ = _mm512_div_ps(vec_, rhs.vec_);
        return *this;
    }

    force_inline simd_vec<float, 16> &vectorcall operator/=(const float rhs) {
        __m512 _rhs = _mm512_set1_ps(rhs);
        vec_ = _mm512_div_ps(vec_, _rhs);
        return *this;
    }

    force_inline simd_vec<float, 16> &vectorcall operator|=(const simd_vec<float, 16> rhs) {
        vec_ = _mm512_or_ps(vec_, rhs.vec_);
        return *this;
    }

    force_inline simd_vec<float, 16> &vectorcall operator|=(const float rhs) {
        vec_ = _mm512_or_ps(vec_, _mm512_set1_ps(rhs));
        return *this;
    }

    force_inline simd_vec<float, 16> &vectorcall operator&=(const simd_vec<float, 16> rhs) {
        vec_ = _mm512_and_ps(vec_, rhs.vec_);
        return *this;
    }

    force_inline simd_vec<float, 16> operator~() const;
    force_inline simd_vec<float, 16> operator-() const;
    force_inline explicit operator simd_vec<int, 16>() const;

    force_inline simd_vec<float, 16> sqrt() const;
    force_inline simd_vec<float, 16> log() const;

    force_inline float length() const { return std::sqrt(length2()); }

    force_inline float length2() const {
        alignas(64) float comp[16];
        _mm512_store_ps(comp, vec_);

        float temp = 0;
        ITERATE_16({ temp += comp[i] * comp[i]; })
        return temp;
    }

    force_inline void copy_to(float *f) const { _mm512_storeu_ps(f, vec_); }
    force_inline void copy_to(float *f, simd_mem_aligned_tag) const { _mm512_store_ps(f, vec_); }

    force_inline void vectorcall blend_to(const simd_vec<float, 16> mask, const simd_vec<float, 16> v1) {
        validate_mask(mask);
        //__mmask16 msk =
        //    _mm512_fpclass_ps_mask(mask.vec_, 0x54); // 0x54 = Negative_Finite | Negative_Infinity | Negative_Zero
        // vec_ = _mm512_mask_blend_ps(msk, vec_, v1.vec_);
        vec_ = _mm512_blendv_ps(vec_, v1.vec_, mask.vec_);
    }

    force_inline void vectorcall blend_inv_to(const simd_vec<float, 16> mask, const simd_vec<float, 16> v1) {
        validate_mask(mask);
        //__mmask16 msk =
        //    _mm512_fpclass_ps_mask(mask.vec_, 0x54); // 0x54 = Negative_Finite | Negative_Infinity | Negative_Zero
        // vec_ = _mm512_mask_blend_ps(msk, v1.vec_, vec_);
        vec_ = _mm512_blendv_ps(v1.vec_, vec_, mask.vec_);
    }

    force_inline static simd_vec<float, 16> vectorcall min(simd_vec<float, 16> v1, simd_vec<float, 16> v2);
    force_inline static simd_vec<float, 16> vectorcall max(simd_vec<float, 16> v1, simd_vec<float, 16> v2);

    force_inline static simd_vec<float, 16> vectorcall and_not(simd_vec<float, 16> v1, simd_vec<float, 16> v2);

    force_inline static simd_vec<float, 16> vectorcall floor(simd_vec<float, 16> v1);
    force_inline static simd_vec<float, 16> vectorcall ceil(simd_vec<float, 16> v1);

    friend force_inline simd_vec<float, 16> vectorcall operator&(simd_vec<float, 16> v1, simd_vec<float, 16> v2);
    friend force_inline simd_vec<float, 16> vectorcall operator|(simd_vec<float, 16> v1, simd_vec<float, 16> v2);
    friend force_inline simd_vec<float, 16> vectorcall operator^(simd_vec<float, 16> v1, simd_vec<float, 16> v2);
    friend force_inline simd_vec<float, 16> vectorcall operator+(simd_vec<float, 16> v1, simd_vec<float, 16> v2);
    friend force_inline simd_vec<float, 16> vectorcall operator-(simd_vec<float, 16> v1, simd_vec<float, 16> v2);
    friend force_inline simd_vec<float, 16> vectorcall operator*(simd_vec<float, 16> v1, simd_vec<float, 16> v2);
    friend force_inline simd_vec<float, 16> vectorcall operator/(simd_vec<float, 16> v1, simd_vec<float, 16> v2);

    friend force_inline simd_vec<float, 16> vectorcall operator+(simd_vec<float, 16> v1, float v2);
    friend force_inline simd_vec<float, 16> vectorcall operator-(simd_vec<float, 16> v1, float v2);
    friend force_inline simd_vec<float, 16> vectorcall operator*(simd_vec<float, 16> v1, float v2);
    friend force_inline simd_vec<float, 16> vectorcall operator/(simd_vec<float, 16> v1, float v2);

    friend force_inline simd_vec<float, 16> vectorcall operator+(float v1, simd_vec<float, 16> v2);
    friend force_inline simd_vec<float, 16> vectorcall operator-(float v1, simd_vec<float, 16> v2);
    friend force_inline simd_vec<float, 16> vectorcall operator*(float v1, simd_vec<float, 16> v2);
    friend force_inline simd_vec<float, 16> vectorcall operator/(float v1, simd_vec<float, 16> v2);

    friend force_inline simd_vec<float, 16> vectorcall operator<(simd_vec<float, 16> v1, simd_vec<float, 16> v2);
    friend force_inline simd_vec<float, 16> vectorcall operator<=(simd_vec<float, 16> v1, simd_vec<float, 16> v2);
    friend force_inline simd_vec<float, 16> vectorcall operator>(simd_vec<float, 16> v1, simd_vec<float, 16> v2);
    friend force_inline simd_vec<float, 16> vectorcall operator>=(simd_vec<float, 16> v1, simd_vec<float, 16> v2);
    friend force_inline simd_vec<float, 16> vectorcall operator==(simd_vec<float, 16> v1, simd_vec<float, 16> v2);
    friend force_inline simd_vec<float, 16> vectorcall operator!=(simd_vec<float, 16> v1, simd_vec<float, 16> v2);

    friend force_inline simd_vec<float, 16> vectorcall operator<(simd_vec<float, 16> v1, float v2);
    friend force_inline simd_vec<float, 16> vectorcall operator<=(simd_vec<float, 16> v1, float v2);
    friend force_inline simd_vec<float, 16> vectorcall operator>(simd_vec<float, 16> v1, float v2);
    friend force_inline simd_vec<float, 16> vectorcall operator>=(simd_vec<float, 16> v1, float v2);
    friend force_inline simd_vec<float, 16> vectorcall operator==(simd_vec<float, 16> v1, float v2);
    friend force_inline simd_vec<float, 16> vectorcall operator!=(simd_vec<float, 16> v1, float v2);

    friend force_inline simd_vec<float, 16> vectorcall clamp(simd_vec<float, 16> v1, float min, float max);
    friend force_inline simd_vec<float, 16> vectorcall pow(simd_vec<float, 16> v1, simd_vec<float, 16> v2);

    friend force_inline simd_vec<float, 16> vectorcall normalize(simd_vec<float, 16> v1);

    friend force_inline simd_vec<float, 16> vectorcall fmadd(simd_vec<float, 16> a, simd_vec<float, 16> b,
                                                             simd_vec<float, 16> c);
    friend force_inline simd_vec<float, 16> vectorcall fmadd(simd_vec<float, 16> a, float b, simd_vec<float, 16> c);
    friend force_inline simd_vec<float, 16> vectorcall fmadd(float a, simd_vec<float, 16> b, float c);

    friend force_inline simd_vec<float, 16> vectorcall fmsub(simd_vec<float, 16> a, simd_vec<float, 16> b,
                                                             simd_vec<float, 16> c);
    friend force_inline simd_vec<float, 16> vectorcall fmsub(simd_vec<float, 16> a, float b, simd_vec<float, 16> c);
    friend force_inline simd_vec<float, 16> vectorcall fmsub(float a, simd_vec<float, 16> b, float c);

#ifndef NDEBUG
    friend void vectorcall __assert_valid_mask(const simd_vec<float, 16> mask) {
        ITERATE_16({
            const float val = mask.get<i>();
            assert(reinterpret_cast<const uint32_t &>(val) == 0 ||
                   reinterpret_cast<const uint32_t &>(val) == 0xffffffff);
        })
    }
#endif

    friend force_inline const float *value_ptr(const simd_vec<float, 16> &v1) {
        return reinterpret_cast<const float *>(&v1.vec_);
    }
    friend force_inline float *value_ptr(simd_vec<float, 16> &v1) { return reinterpret_cast<float *>(&v1.vec_); }

    static int size() { return 16; }
    static bool is_native() { return true; }
};

template <> class simd_vec<int, 16> {
    __m512i vec_;

    friend class simd_vec<float, 16>;

  public:
    force_inline simd_vec() = default;
    force_inline simd_vec(const int f) { vec_ = _mm512_set1_epi32(f); }
    force_inline simd_vec(const int i0, const int i1, const int i2, const int i3, const int i4, const int i5,
                          const int i6, const int i7, const int i8, const int i9, const int i10, const int i11,
                          const int i12, const int i13, const int i14, const int i15) {
        vec_ = _mm512_setr_epi32(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15);
    }
    force_inline explicit simd_vec(const int *f) { vec_ = _mm512_loadu_si512((const __m512i *)f); }
    force_inline simd_vec(const int *f, simd_mem_aligned_tag) { vec_ = _mm512_load_si512((const __m512i *)f); }

    force_inline int operator[](const int i) const {
        __m512i temp = _mm512_maskz_compress_epi32(__mmask16(1u << (i & 15)), vec_);
        return _mm512_cvtsi512_si32(temp);
    }

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

    force_inline simd_vec<int, 16> &vectorcall operator+=(const simd_vec<int, 16> rhs) {
        vec_ = _mm512_add_epi32(vec_, rhs.vec_);
        return *this;
    }

    force_inline simd_vec<int, 16> &vectorcall operator+=(const int rhs) {
        vec_ = _mm512_add_epi32(vec_, _mm512_set1_epi32(rhs));
        return *this;
    }

    force_inline simd_vec<int, 16> &vectorcall operator-=(const simd_vec<int, 16> rhs) {
        vec_ = _mm512_sub_epi32(vec_, rhs.vec_);
        return *this;
    }

    force_inline simd_vec<int, 16> &vectorcall operator-=(const int rhs) {
        vec_ = _mm512_sub_epi32(vec_, _mm512_set1_epi32(rhs));
        return *this;
    }

    force_inline simd_vec<int, 16> &vectorcall operator*=(const simd_vec<int, 16> rhs) {
#if defined(_MSC_VER) && !defined(__clang__)
        ITERATE_16({ vec_.m512i_i32[i] *= rhs.vec_.m512i_i32[i]; })
#else
        alignas(64) int comp[16], rhs_comp[16];
        _mm512_store_epi32(comp, vec_);
        _mm512_store_epi32(rhs_comp, rhs.vec_);
        ITERATE_16({ comp[i] *= rhs_comp[i]; })
        vec_ = _mm512_load_epi32(comp);
#endif
        return *this;
    }

    force_inline simd_vec<int, 16> &vectorcall operator*=(const int rhs) {
#if defined(_MSC_VER) && !defined(__clang__)
        ITERATE_16({ vec_.m512i_i32[i] *= rhs; })
#else
        alignas(64) int comp[16];
        _mm512_store_epi32(comp, vec_);
        ITERATE_16({ comp[i] *= rhs; })
        vec_ = _mm512_load_epi32(comp);
#endif
        return *this;
    }

    force_inline simd_vec<int, 16> &vectorcall operator/=(const simd_vec<int, 16> rhs) {
#if defined(_MSC_VER) && !defined(__clang__)
        ITERATE_16({ vec_.m512i_i32[i] /= rhs.vec_.m512i_i32[i]; })
#else
        alignas(64) int comp[16], rhs_comp[16];
        _mm512_store_epi32(comp, vec_);
        _mm512_store_epi32(rhs_comp, rhs.vec_);
        ITERATE_16({ comp[i] /= rhs_comp[i]; })
        vec_ = _mm512_load_epi32(comp);
#endif
        return *this;
    }

    force_inline simd_vec<int, 16> &vectorcall operator/=(const int rhs) {
#if defined(_MSC_VER) && !defined(__clang__)
        ITERATE_16({ vec_.m512i_i32[i] /= rhs; })
#else
        alignas(64) int comp[16];
        _mm512_store_epi32(comp, vec_);
        ITERATE_16({ comp[i] /= rhs; })
        vec_ = _mm512_load_epi32(comp);
#endif
        return *this;
    }

    force_inline simd_vec<int, 16> &vectorcall operator|=(const simd_vec<int, 16> rhs) {
        vec_ = _mm512_or_si512(vec_, rhs.vec_);
        return *this;
    }

    force_inline simd_vec<int, 16> &vectorcall operator|=(const int rhs) {
        vec_ = _mm512_or_si512(vec_, _mm512_set1_epi32(rhs));
        return *this;
    }

    force_inline simd_vec<int, 16> operator-() const {
        simd_vec<int, 16> temp;
        temp.vec_ = _mm512_sub_epi32(_mm512_setzero_si512(), vec_);
        return temp;
    }

    force_inline simd_vec<int, 16> operator==(const int rhs) const {
        simd_vec<int, 16> ret;
        ret.vec_ = _mm512_movm_epi32(_mm512_cmpeq_epi32_mask(vec_, _mm512_set1_epi32(rhs)));
        return ret;
    }

    force_inline simd_vec<int, 16> vectorcall operator==(const simd_vec<int, 16> rhs) const {
        simd_vec<int, 16> ret;
        ret.vec_ = _mm512_movm_epi32(_mm512_cmpeq_epi32_mask(vec_, rhs.vec_));
        return ret;
    }

    force_inline simd_vec<int, 16> operator!=(const int rhs) const {
        simd_vec<int, 16> ret;
        ret.vec_ = _mm512_andnot_si512(_mm512_movm_epi32(_mm512_cmpeq_epi32_mask(vec_, _mm512_set1_epi32(rhs))),
                                       _mm512_set1_epi32(~0));
        return ret;
    }

    force_inline simd_vec<int, 16> vectorcall operator!=(const simd_vec<int, 16> rhs) const {
        simd_vec<int, 16> ret;
        ret.vec_ =
            _mm512_andnot_si512(_mm512_movm_epi32(_mm512_cmpeq_epi32_mask(vec_, rhs.vec_)), _mm512_set1_epi32(~0));
        return ret;
    }

    force_inline simd_vec<int, 16> &vectorcall operator&=(const simd_vec<int, 16> rhs) {
        vec_ = _mm512_and_si512(vec_, rhs.vec_);
        return *this;
    }

    force_inline simd_vec<int, 16> &vectorcall operator&=(const int rhs) {
        vec_ = _mm512_and_si512(vec_, _mm512_set1_epi32(rhs));
        return *this;
    }

    force_inline explicit operator simd_vec<float, 16>() const {
        simd_vec<float, 16> ret;
        ret.vec_ = _mm512_cvtepi32_ps(vec_);
        return ret;
    }

    force_inline void copy_to(int *f) const { _mm512_storeu_si512((__m512i *)f, vec_); }
    force_inline void copy_to(int *f, simd_mem_aligned_tag) const { _mm512_store_si512((__m512i *)f, vec_); }

    force_inline void vectorcall blend_to(const simd_vec<int, 16> mask, const simd_vec<int, 16> v1) {
        validate_mask(mask);
        vec_ = _mm512_ternarylogic_epi32(vec_, v1.vec_, _mm512_srai_epi32(mask.vec_, 31), 0xd8);
    }

    force_inline void vectorcall blend_inv_to(const simd_vec<int, 16> mask, const simd_vec<int, 16> v1) {
        validate_mask(mask);
        vec_ = _mm512_ternarylogic_epi32(v1.vec_, vec_, _mm512_srai_epi32(mask.vec_, 31), 0xd8);
    }

    force_inline int movemask() const { return _mm512_movemask_epi32(vec_); }

    force_inline bool vectorcall all_zeros() const {
        return _mm512_cmpeq_epi32_mask(vec_, _mm512_setzero_si512()) == 0xFFFF;
    }

    force_inline bool vectorcall all_zeros(const simd_vec<int, 16> mask) const {
        return _mm512_cmpeq_epi32_mask(_mm512_and_si512(vec_, mask.vec_), _mm512_setzero_si512()) == 0xFFFF;
    }

    force_inline bool not_all_zeros() const { return !all_zeros(); }

    force_inline static simd_vec<int, 16> vectorcall min(const simd_vec<int, 16> v1, const simd_vec<int, 16> v2) {
        simd_vec<int, 16> temp;
        temp.vec_ = _mm512_min_epi32(v1.vec_, v2.vec_);
        return temp;
    }

    force_inline static simd_vec<int, 16> vectorcall max(const simd_vec<int, 16> v1, const simd_vec<int, 16> v2) {
        simd_vec<int, 16> temp;
        temp.vec_ = _mm512_max_epi32(v1.vec_, v2.vec_);
        return temp;
    }

    force_inline static simd_vec<int, 16> vectorcall and_not(const simd_vec<int, 16> v1, const simd_vec<int, 16> v2) {
        simd_vec<int, 16> temp;
        temp.vec_ = _mm512_castps_si512(_mm512_andnot_ps(_mm512_castsi512_ps(v1.vec_), _mm512_castsi512_ps(v2.vec_)));
        return temp;
    }

    friend force_inline simd_vec<int, 16> vectorcall operator&(const simd_vec<int, 16> v1, const simd_vec<int, 16> v2) {
        simd_vec<int, 16> temp;
        temp.vec_ = _mm512_castps_si512(_mm512_and_ps(_mm512_castsi512_ps(v1.vec_), _mm512_castsi512_ps(v2.vec_)));
        return temp;
    }

    friend force_inline simd_vec<int, 16> vectorcall operator|(const simd_vec<int, 16> v1, const simd_vec<int, 16> v2) {
        simd_vec<int, 16> temp;
        temp.vec_ = _mm512_castps_si512(_mm512_or_ps(_mm512_castsi512_ps(v1.vec_), _mm512_castsi512_ps(v2.vec_)));
        return temp;
    }

    friend force_inline simd_vec<int, 16> vectorcall operator^(const simd_vec<int, 16> v1, const simd_vec<int, 16> v2) {
        simd_vec<int, 16> temp;
        temp.vec_ = _mm512_castps_si512(_mm512_xor_ps(_mm512_castsi512_ps(v1.vec_), _mm512_castsi512_ps(v2.vec_)));
        return temp;
    }

    friend force_inline simd_vec<int, 16> vectorcall operator+(const simd_vec<int, 16> v1, const simd_vec<int, 16> v2) {
        simd_vec<int, 16> temp;
        temp.vec_ = _mm512_add_epi32(v1.vec_, v2.vec_);
        return temp;
    }

    friend force_inline simd_vec<int, 16> vectorcall operator-(const simd_vec<int, 16> v1, const simd_vec<int, 16> v2) {
        simd_vec<int, 16> temp;
        temp.vec_ = _mm512_sub_epi32(v1.vec_, v2.vec_);
        return temp;
    }

    friend force_inline simd_vec<int, 16> vectorcall operator*(const simd_vec<int, 16> v1, const simd_vec<int, 16> v2) {
        simd_vec<int, 16> ret;
#if defined(_MSC_VER) && !defined(__clang__)
        ITERATE_16({ ret.vec_.m512i_i32[i] = v1.vec_.m512i_i32[i] * v2.vec_.m512i_i32[i]; })
#else
        alignas(64) int comp1[16], comp2[16];
        _mm512_store_epi32(comp1, v1.vec_);
        _mm512_store_epi32(comp2, v2.vec_);
        ITERATE_16({ comp1[i] *= comp2[i]; })
        ret.vec_ = _mm512_load_epi32(comp1);
#endif
        return ret;
    }

    friend force_inline simd_vec<int, 16> vectorcall operator/(const simd_vec<int, 16> v1, const simd_vec<int, 16> v2) {
        simd_vec<int, 16> ret;
#if defined(_MSC_VER) && !defined(__clang__)
        ITERATE_16({ ret.vec_.m512i_i32[i] = v1.vec_.m512i_i32[i] / v2.vec_.m512i_i32[i]; })
#else
        alignas(64) int comp1[16], comp2[16];
        _mm512_store_epi32(comp1, v1.vec_);
        _mm512_store_epi32(comp2, v2.vec_);
        ITERATE_16({ comp1[i] /= comp2[i]; })
        ret.vec_ = _mm512_load_epi32(comp1);
#endif
        return ret;
    }

    friend force_inline simd_vec<int, 16> vectorcall operator+(const simd_vec<int, 16> v1, const int v2) {
        simd_vec<int, 16> temp;
        temp.vec_ = _mm512_add_epi32(v1.vec_, _mm512_set1_epi32(v2));
        return temp;
    }

    friend force_inline simd_vec<int, 16> vectorcall operator-(const simd_vec<int, 16> v1, const int v2) {
        simd_vec<int, 16> temp;
        temp.vec_ = _mm512_sub_epi32(v1.vec_, _mm512_set1_epi32(v2));
        return temp;
    }

    friend force_inline simd_vec<int, 16> vectorcall operator*(const simd_vec<int, 16> v1, const int v2) {
        simd_vec<int, 16> ret;
#if defined(_MSC_VER) && !defined(__clang__)
        ITERATE_16({ ret.vec_.m512i_i32[i] = v1.vec_.m512i_i32[i] * v2; })
#else
        alignas(64) int comp[16];
        _mm512_store_epi32(comp, v1.vec_);
        ITERATE_16({ comp[i] *= v2; })
        ret.vec_ = _mm512_load_epi32(comp);
#endif
        return ret;
    }

    friend force_inline simd_vec<int, 16> vectorcall operator/(const simd_vec<int, 16> v1, const int v2) {
        simd_vec<int, 16> ret;
#if defined(_MSC_VER) && !defined(__clang__)
        ITERATE_16({ ret.vec_.m512i_i32[i] = v1.vec_.m512i_i32[i] / v2; })
#else
        alignas(64) int comp[16];
        _mm512_store_epi32(comp, v1.vec_);
        ITERATE_16({ comp[i] /= v2; })
        ret.vec_ = _mm512_load_epi32(comp);
#endif
        return ret;
    }

    friend force_inline simd_vec<int, 16> vectorcall operator+(const int v1, const simd_vec<int, 16> v2) {
        simd_vec<int, 16> temp;
        temp.vec_ = _mm512_add_epi32(_mm512_set1_epi32(v1), v2.vec_);
        return temp;
    }

    friend force_inline simd_vec<int, 16> vectorcall operator-(const int v1, const simd_vec<int, 16> v2) {
        simd_vec<int, 16> temp;
        temp.vec_ = _mm512_sub_epi32(_mm512_set1_epi32(v1), v2.vec_);
        return temp;
    }

    friend force_inline simd_vec<int, 16> vectorcall operator*(const int v1, const simd_vec<int, 16> v2) {
        simd_vec<int, 16> ret;
#if defined(_MSC_VER) && !defined(__clang__)
        ITERATE_16({ ret.vec_.m512i_i32[i] = v1 * v2.vec_.m512i_i32[i]; })
#else
        alignas(64) int comp[16];
        _mm512_store_epi32(comp, v2.vec_);
        ITERATE_16({ comp[i] *= v1; })
        ret.vec_ = _mm512_load_epi32(comp);
#endif
        return ret;
    }

    friend force_inline simd_vec<int, 16> vectorcall operator/(const int v1, const simd_vec<int, 16> v2) {
        simd_vec<int, 16> ret;
#if defined(_MSC_VER) && !defined(__clang__)
        ITERATE_16({ ret.vec_.m512i_i32[i] = v1 / v2.vec_.m512i_i32[i]; })
#else
        alignas(64) int comp[16];
        _mm512_store_epi32(comp, v2.vec_);
        ITERATE_16({ comp[i] = v1 / comp[i]; })
        ret.vec_ = _mm512_load_epi32(comp);
#endif
        return ret;
    }

    friend force_inline simd_vec<int, 16> vectorcall operator<(const simd_vec<int, 16> v1, const simd_vec<int, 16> v2) {
        simd_vec<int, 16> ret;
        ret.vec_ = _mm512_movm_epi32(_mm512_cmpgt_epi32_mask(v2.vec_, v1.vec_));
        return ret;
    }

    friend force_inline simd_vec<int, 16> vectorcall operator>(const simd_vec<int, 16> v1, const simd_vec<int, 16> v2) {
        simd_vec<int, 16> ret;
        ret.vec_ = _mm512_movm_epi32(_mm512_cmpgt_epi32_mask(v1.vec_, v2.vec_));
        return ret;
    }

    friend force_inline simd_vec<int, 16> vectorcall operator>=(const simd_vec<int, 16> v1,
                                                                const simd_vec<int, 16> v2) {
        simd_vec<int, 16> ret;
        ret.vec_ = _mm512_movm_epi32(_mm512_cmpge_epi32_mask(v1.vec_, v2.vec_));
        return ret;
    }

    friend force_inline simd_vec<int, 16> vectorcall operator<(const simd_vec<int, 16> v1, const int v2) {
        simd_vec<int, 16> ret;
        ret.vec_ = _mm512_movm_epi32(_mm512_cmpgt_epi32_mask(_mm512_set1_epi32(v2), v1.vec_));
        return ret;
    }

    friend force_inline simd_vec<int, 16> vectorcall operator<=(const simd_vec<int, 16> v1, const int v2) {
        simd_vec<int, 16> ret;
        ret.vec_ = _mm512_movm_epi32(_mm512_cmpge_epi32_mask(_mm512_set1_epi32(v2), v1.vec_));
        return ret;
    }

    friend force_inline simd_vec<int, 16> vectorcall operator>(const simd_vec<int, 16> v1, const int v2) {
        simd_vec<int, 16> ret;
        ret.vec_ = _mm512_movm_epi32(_mm512_cmpgt_epi32_mask(v1.vec_, _mm512_set1_epi32(v2)));
        return ret;
    }

    friend force_inline simd_vec<int, 16> vectorcall operator>=(const simd_vec<int, 16> v1, const int v2) {
        simd_vec<int, 16> ret;
        ret.vec_ = _mm512_movm_epi32(_mm512_cmpge_epi32_mask(v1.vec_, _mm512_set1_epi32(v2)));
        return ret;
    }

    friend force_inline simd_vec<int, 16> vectorcall operator>>(const simd_vec<int, 16> v1,
                                                                const simd_vec<int, 16> v2) {
        simd_vec<int, 16> ret;
        ret.vec_ = _mm512_srlv_epi32(v1.vec_, v2.vec_);
        return ret;
    }

    friend force_inline simd_vec<int, 16> vectorcall operator>>(const simd_vec<int, 16> v1, const int v2) {
        simd_vec<int, 16> ret;
        ret.vec_ = _mm512_srli_epi32(v1.vec_, v2);
        return ret;
    }

    friend force_inline simd_vec<int, 16> vectorcall operator<<(const simd_vec<int, 16> v1,
                                                                const simd_vec<int, 16> v2) {
        simd_vec<int, 16> ret;
        ret.vec_ = _mm512_sllv_epi32(v1.vec_, v2.vec_);
        return ret;
    }

    friend force_inline simd_vec<int, 16> vectorcall operator<<(const simd_vec<int, 16> v1, const int v2) {
        simd_vec<int, 16> ret;
        ret.vec_ = _mm512_slli_epi32(v1.vec_, v2);
        return ret;
    }

    force_inline simd_vec<int, 16> operator~() const {
        simd_vec<int, 16> ret;
        ret.vec_ = _mm512_andnot_si512(vec_, _mm512_set1_epi32(~0));
        return ret;
    }

    friend force_inline simd_vec<int, 16> vectorcall srai(const simd_vec<int, 16> v1, const int v2) {
        simd_vec<int, 16> ret;
        ret.vec_ = _mm512_srai_epi32(v1.vec_, v2);
        return ret;
    }

    friend force_inline bool vectorcall is_equal(const simd_vec<int, 16> v1, const simd_vec<int, 16> v2) {
        return _mm512_cmpeq_epi32_mask(v1.vec_, v2.vec_) == 0xFFFF;
    }

    friend force_inline simd_vec<float, 16> vectorcall gather(const float *base_addr, simd_vec<int, 16> vindex);
    friend force_inline simd_vec<int, 16> vectorcall gather(const int *base_addr, simd_vec<int, 16> vindex);

    friend force_inline void vectorcall scatter(float *base_addr, simd_vec<int, 16> vindex, simd_vec<float, 16> v);
    friend force_inline void vectorcall scatter(int *base_addr, simd_vec<int, 16> vindex, simd_vec<int, 16> v);

#ifndef NDEBUG
    friend void vectorcall __assert_valid_mask(const simd_vec<int, 16> mask) {
        ITERATE_16({
            const int val = mask.get<i>();
            assert(val == 0 || val == -1);
        })
    }
#endif

    friend force_inline const int *value_ptr(const simd_vec<int, 16> &v1) {
        return reinterpret_cast<const int *>(&v1.vec_);
    }
    friend force_inline int *value_ptr(simd_vec<int, 16> &v1) { return reinterpret_cast<int *>(&v1.vec_); }

    static int size() { return 16; }
    static bool is_native() { return true; }
};

force_inline simd_vec<float, 16> simd_vec<float, 16>::operator~() const {
    simd_vec<float, 16> ret;
    ret.vec_ = _mm512_castsi512_ps(_mm512_andnot_si512(_mm512_castps_si512(vec_), _mm512_set1_epi32(~0)));
    return ret;
}

force_inline simd_vec<float, 16> simd_vec<float, 16>::operator-() const {
    simd_vec<float, 16> temp;
    __m512 m = _mm512_set1_ps(-0.0f);
    temp.vec_ = _mm512_xor_ps(vec_, m);
    return temp;
}

force_inline simd_vec<float, 16>::operator simd_vec<int, 16>() const {
    simd_vec<int, 16> ret;
    ret.vec_ = _mm512_cvttps_epi32(vec_);
    return ret;
}

force_inline simd_vec<float, 16> simd_vec<float, 16>::sqrt() const {
    simd_vec<float, 16> temp;
    temp.vec_ = _mm512_sqrt_ps(vec_);
    return temp;
}

force_inline simd_vec<float, 16> simd_vec<float, 16>::log() const {
    alignas(64) float comp[16];
    _mm512_store_ps(comp, vec_);
    ITERATE_16({ comp[i] = std::log(comp[i]); })
    return simd_vec<float, 16>{comp, simd_mem_aligned};
}

force_inline simd_vec<float, 16> vectorcall simd_vec<float, 16>::min(const simd_vec<float, 16> v1,
                                                                     const simd_vec<float, 16> v2) {
    simd_vec<float, 16> temp;
    temp.vec_ = _mm512_min_ps(v1.vec_, v2.vec_);
    return temp;
}

force_inline simd_vec<float, 16> vectorcall simd_vec<float, 16>::max(const simd_vec<float, 16> v1,
                                                                     const simd_vec<float, 16> v2) {
    simd_vec<float, 16> temp;
    temp.vec_ = _mm512_max_ps(v1.vec_, v2.vec_);
    return temp;
}

force_inline static simd_vec<float, 16> vectorcall and_not(const simd_vec<float, 16> v1, const simd_vec<float, 16> v2) {
    simd_vec<float, 16> temp;
    temp.vec_ = _mm512_andnot_ps(v1.vec_, v2.vec_);
    return temp;
}

force_inline static simd_vec<float, 16> vectorcall floor(const simd_vec<float, 16> v1) {
    simd_vec<float, 16> temp;
    temp.vec_ = _mm512_floor_ps(v1.vec_);
    return temp;
}

force_inline static simd_vec<float, 16> vectorcall ceil(const simd_vec<float, 16> v1) {
    simd_vec<float, 16> temp;
    temp.vec_ = _mm512_ceil_ps(v1.vec_);
    return temp;
}

force_inline simd_vec<float, 16> vectorcall operator&(const simd_vec<float, 16> v1, const simd_vec<float, 16> v2) {
    simd_vec<float, 16> temp;
    temp.vec_ = _mm512_and_ps(v1.vec_, v2.vec_);
    return temp;
}

force_inline simd_vec<float, 16> vectorcall operator|(const simd_vec<float, 16> v1, const simd_vec<float, 16> v2) {
    simd_vec<float, 16> temp;
    temp.vec_ = _mm512_or_ps(v1.vec_, v2.vec_);
    return temp;
}

force_inline simd_vec<float, 16> vectorcall operator^(const simd_vec<float, 16> v1, const simd_vec<float, 16> v2) {
    simd_vec<float, 16> temp;
    temp.vec_ = _mm512_xor_ps(v1.vec_, v2.vec_);
    return temp;
}

force_inline simd_vec<float, 16> vectorcall operator+(const simd_vec<float, 16> v1, const simd_vec<float, 16> v2) {
    simd_vec<float, 16> temp;
    temp.vec_ = _mm512_add_ps(v1.vec_, v2.vec_);
    return temp;
}

force_inline simd_vec<float, 16> vectorcall operator-(const simd_vec<float, 16> v1, const simd_vec<float, 16> v2) {
    simd_vec<float, 16> temp;
    temp.vec_ = _mm512_sub_ps(v1.vec_, v2.vec_);
    return temp;
}

force_inline simd_vec<float, 16> vectorcall operator*(const simd_vec<float, 16> v1, const simd_vec<float, 16> v2) {
    simd_vec<float, 16> temp;
    temp.vec_ = _mm512_mul_ps(v1.vec_, v2.vec_);
    return temp;
}

force_inline simd_vec<float, 16> vectorcall operator/(const simd_vec<float, 16> v1, const simd_vec<float, 16> v2) {
    simd_vec<float, 16> temp;
    temp.vec_ = _mm512_div_ps(v1.vec_, v2.vec_);
    return temp;
}

force_inline simd_vec<float, 16> vectorcall operator+(const simd_vec<float, 16> v1, const float v2) {
    simd_vec<float, 16> temp;
    temp.vec_ = _mm512_add_ps(v1.vec_, _mm512_set1_ps(v2));
    return temp;
}

force_inline simd_vec<float, 16> vectorcall operator-(const simd_vec<float, 16> v1, const float v2) {
    simd_vec<float, 16> temp;
    temp.vec_ = _mm512_sub_ps(v1.vec_, _mm512_set1_ps(v2));
    return temp;
}

force_inline simd_vec<float, 16> vectorcall operator*(const simd_vec<float, 16> v1, const float v2) {
    simd_vec<float, 16> temp;
    temp.vec_ = _mm512_mul_ps(v1.vec_, _mm512_set1_ps(v2));
    return temp;
}

force_inline simd_vec<float, 16> vectorcall operator/(const simd_vec<float, 16> v1, const float v2) {
    simd_vec<float, 16> temp;
    temp.vec_ = _mm512_div_ps(v1.vec_, _mm512_set1_ps(v2));
    return temp;
}

force_inline simd_vec<float, 16> vectorcall operator+(const float v1, const simd_vec<float, 16> v2) {
    simd_vec<float, 16> temp;
    temp.vec_ = _mm512_add_ps(_mm512_set1_ps(v1), v2.vec_);
    return temp;
}

force_inline simd_vec<float, 16> vectorcall operator-(const float v1, const simd_vec<float, 16> v2) {
    simd_vec<float, 16> temp;
    temp.vec_ = _mm512_sub_ps(_mm512_set1_ps(v1), v2.vec_);
    return temp;
}

force_inline simd_vec<float, 16> vectorcall operator*(const float v1, const simd_vec<float, 16> v2) {
    simd_vec<float, 16> temp;
    temp.vec_ = _mm512_mul_ps(_mm512_set1_ps(v1), v2.vec_);
    return temp;
}

force_inline simd_vec<float, 16> vectorcall operator/(const float v1, const simd_vec<float, 16> v2) {
    simd_vec<float, 16> temp;
    temp.vec_ = _mm512_div_ps(_mm512_set1_ps(v1), v2.vec_);
    return temp;
}

force_inline simd_vec<float, 16> vectorcall operator<(const simd_vec<float, 16> v1, const simd_vec<float, 16> v2) {
    simd_vec<float, 16> ret;
    ret.vec_ = _mm512_cmp_ps(v1.vec_, v2.vec_, _CMP_LT_OS);
    return ret;
}

force_inline simd_vec<float, 16> vectorcall operator<=(const simd_vec<float, 16> v1, const simd_vec<float, 16> v2) {
    simd_vec<float, 16> ret;
    ret.vec_ = _mm512_cmp_ps(v1.vec_, v2.vec_, _CMP_LE_OS);
    return ret;
}

force_inline simd_vec<float, 16> vectorcall operator>(const simd_vec<float, 16> v1, const simd_vec<float, 16> v2) {
    simd_vec<float, 16> ret;
    ret.vec_ = _mm512_cmp_ps(v1.vec_, v2.vec_, _CMP_GT_OS);
    return ret;
}

force_inline simd_vec<float, 16> vectorcall operator>=(const simd_vec<float, 16> v1, const simd_vec<float, 16> v2) {
    simd_vec<float, 16> ret;
    ret.vec_ = _mm512_cmp_ps(v1.vec_, v2.vec_, _CMP_GE_OS);
    return ret;
}

force_inline simd_vec<float, 16> vectorcall operator==(const simd_vec<float, 16> v1, const simd_vec<float, 16> v2) {
    simd_vec<float, 16> ret;
    ret.vec_ = _mm512_cmp_ps(v1.vec_, v2.vec_, _CMP_EQ_OS);
    return ret;
}

force_inline simd_vec<float, 16> vectorcall operator!=(const simd_vec<float, 16> v1, const simd_vec<float, 16> v2) {
    simd_vec<float, 16> ret;
    ret.vec_ = _mm512_cmp_ps(v1.vec_, v2.vec_, _CMP_NEQ_OS);
    return ret;
}

force_inline simd_vec<float, 16> vectorcall operator<(const simd_vec<float, 16> v1, const float v2) {
    simd_vec<float, 16> ret;
    ret.vec_ = _mm512_cmp_ps(v1.vec_, _mm512_set1_ps(v2), _CMP_LT_OS);
    return ret;
}

force_inline simd_vec<float, 16> vectorcall operator<=(const simd_vec<float, 16> v1, const float v2) {
    simd_vec<float, 16> ret;
    ret.vec_ = _mm512_cmp_ps(v1.vec_, _mm512_set1_ps(v2), _CMP_LE_OS);
    return ret;
}

force_inline simd_vec<float, 16> vectorcall operator>(const simd_vec<float, 16> v1, const float v2) {
    simd_vec<float, 16> ret;
    ret.vec_ = _mm512_cmp_ps(v1.vec_, _mm512_set1_ps(v2), _CMP_GT_OS);
    return ret;
}

force_inline simd_vec<float, 16> vectorcall operator>=(const simd_vec<float, 16> v1, const float v2) {
    simd_vec<float, 16> ret;
    ret.vec_ = _mm512_cmp_ps(v1.vec_, _mm512_set1_ps(v2), _CMP_GE_OS);
    return ret;
}

force_inline simd_vec<float, 16> vectorcall operator==(const simd_vec<float, 16> v1, const float v2) {
    simd_vec<float, 16> ret;
    ret.vec_ = _mm512_cmp_ps(v1.vec_, _mm512_set1_ps(v2), _CMP_EQ_OS);
    return ret;
}

force_inline simd_vec<float, 16> vectorcall operator!=(const simd_vec<float, 16> v1, const float v2) {
    simd_vec<float, 16> ret;
    ret.vec_ = _mm512_cmp_ps(v1.vec_, _mm512_set1_ps(v2), _CMP_NEQ_OS);
    return ret;
}

force_inline simd_vec<float, 16> vectorcall clamp(const simd_vec<float, 16> v1, const float min, const float max) {
    simd_vec<float, 16> ret;
    ret.vec_ = _mm512_max_ps(_mm512_set1_ps(min), _mm512_min_ps(v1.vec_, _mm512_set1_ps(max)));
    return ret;
}

force_inline simd_vec<float, 16> vectorcall pow(const simd_vec<float, 16> v1, const simd_vec<float, 16> v2) {
    alignas(64) float comp1[16], comp2[16];
    _mm512_store_ps(comp1, v1.vec_);
    _mm512_store_ps(comp2, v2.vec_);
    ITERATE_16({ comp1[i] = std::pow(comp1[i], comp2[i]); })
    return simd_vec<float, 16>{comp1, simd_mem_aligned};
}

force_inline simd_vec<float, 16> vectorcall normalize(const simd_vec<float, 16> v1) { return v1 / v1.length(); }

force_inline simd_vec<float, 16> vectorcall fmadd(const simd_vec<float, 16> a, const simd_vec<float, 16> b,
                                                  const simd_vec<float, 16> c) {
    simd_vec<float, 16> ret;
    ret.vec_ = _mm512_fmadd_ps(a.vec_, b.vec_, c.vec_);
    return ret;
}

force_inline simd_vec<float, 16> vectorcall fmadd(const simd_vec<float, 16> a, const float b,
                                                  const simd_vec<float, 16> c) {
    simd_vec<float, 16> ret;
    ret.vec_ = _mm512_fmadd_ps(a.vec_, _mm512_set1_ps(b), c.vec_);
    return ret;
}

force_inline simd_vec<float, 16> vectorcall fmadd(const float a, const simd_vec<float, 16> b, const float c) {
    simd_vec<float, 16> ret;
    ret.vec_ = _mm512_fmadd_ps(_mm512_set1_ps(a), b.vec_, _mm512_set1_ps(c));
    return ret;
}

force_inline simd_vec<float, 16> vectorcall fmsub(const simd_vec<float, 16> a, const simd_vec<float, 16> b,
                                                  const simd_vec<float, 16> c) {
    simd_vec<float, 16> ret;
    ret.vec_ = _mm512_fmsub_ps(a.vec_, b.vec_, c.vec_);
    return ret;
}

force_inline simd_vec<float, 16> vectorcall fmsub(const simd_vec<float, 16> a, const float b,
                                                  const simd_vec<float, 16> c) {
    simd_vec<float, 16> ret;
    ret.vec_ = _mm512_fmsub_ps(a.vec_, _mm512_set1_ps(b), c.vec_);
    return ret;
}

force_inline simd_vec<float, 16> vectorcall fmsub(const float a, const simd_vec<float, 16> b, const float c) {
    simd_vec<float, 16> ret;
    ret.vec_ = _mm512_fmsub_ps(_mm512_set1_ps(a), b.vec_, _mm512_set1_ps(c));
    return ret;
}

force_inline simd_vec<float, 16> vectorcall gather(const float *base_addr, const simd_vec<int, 16> vindex) {
    simd_vec<float, 16> ret;
    ret.vec_ = _mm512_i32gather_ps(vindex.vec_, base_addr, sizeof(float));
    return ret;
}

force_inline simd_vec<int, 16> vectorcall gather(const int *base_addr, const simd_vec<int, 16> vindex) {
    simd_vec<int, 16> ret;
    ret.vec_ = _mm512_i32gather_epi32(vindex.vec_, base_addr, sizeof(int));
    return ret;
}

force_inline void vectorcall scatter(float* base_addr, simd_vec<int, 16> vindex, simd_vec<float, 16> v) {
    _mm512_i32scatter_ps(base_addr, vindex.vec_, v.vec_, sizeof(float));
}

force_inline void vectorcall scatter(int* base_addr, simd_vec<int, 16> vindex, simd_vec<int, 16> v) {
    _mm512_i32scatter_epi32(base_addr, vindex.vec_, v.vec_, sizeof(int));
}

} // namespace NS
} // namespace Ray

#undef validate_mask

#pragma warning(pop)

#ifdef __GNUC__
#pragma GCC pop_options
#pragma clang attribute pop
#endif
