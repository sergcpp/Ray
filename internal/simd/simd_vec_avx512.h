//#pragma once

#include "simd_vec_avx.h"

#include <immintrin.h>

#ifdef __GNUC__
#pragma GCC push_options
#pragma GCC target("avx512f", "avx512bw", "avx512dq")
#pragma clang attribute push (__attribute__((target("avx512f,avx512bw,avx512dq"))), apply_to=function)
#endif

#define _mm512_cmp_ps(a, b, c) _mm512_castsi512_ps(_mm512_movm_epi32(_mm512_cmp_ps_mask(a, b, c)))

#define _mm512_blendv_ps(a, b, m)                                                                                      \
    _mm512_castsi512_ps(_mm512_ternarylogic_epi32(_mm512_castps_si512(a), _mm512_castps_si512(b),                                 \
                                       _mm512_srai_epi32(_mm512_castps_si512(m), 31), 0xd8))

#define _mm512_movemask_epi32(a)                                                                                       \
    (int)_mm512_cmpneq_epi32_mask(_mm512_setzero_si512(), _mm512_and_si512(_mm512_set1_epi32(0x80000000U), a))

#ifndef NDEBUG
#define VALIDATE_MASKS 1
#endif

#pragma warning(push)
#pragma warning(disable : 4752)

namespace Ray {
namespace NS {

template <> class simd_vec<float, 16> {
  public:
    union {
        __m512 vec_;
        float comp_[16];
    };

    friend class simd_vec<int, 16>;

  public:
    force_inline simd_vec() = default;
    force_inline simd_vec(float f) { vec_ = _mm512_set1_ps(f); }
    force_inline simd_vec(float f0, float f1, float f2, float f3, float f4, float f5, float f6, float f7, float f8,
                          float f9, float f10, float f11, float f12, float f13, float f14, float f15) {
        vec_ = _mm512_setr_ps(f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15);
    }
    force_inline explicit simd_vec(const float *f) { vec_ = _mm512_loadu_ps(f); }
    force_inline simd_vec(const float *f, simd_mem_aligned_tag) { vec_ = _mm512_load_ps(f); }

    force_inline float &operator[](int i) { return comp_[i]; }
    force_inline const float &operator[](int i) const { return comp_[i]; }

    force_inline simd_vec<float, 16> &operator+=(const simd_vec<float, 16> &rhs) {
        vec_ = _mm512_add_ps(vec_, rhs.vec_);
        return *this;
    }

    force_inline simd_vec<float, 16> &operator+=(float rhs) {
        __m512 _rhs = _mm512_set1_ps(rhs);
        vec_ = _mm512_add_ps(vec_, _rhs);
        return *this;
    }

    force_inline simd_vec<float, 16> &operator-=(const simd_vec<float, 16> &rhs) {
        vec_ = _mm512_sub_ps(vec_, rhs.vec_);
        return *this;
    }

    force_inline simd_vec<float, 16> &operator-=(float rhs) {
        vec_ = _mm512_sub_ps(vec_, _mm512_set1_ps(rhs));
        return *this;
    }

    force_inline simd_vec<float, 16> &operator*=(const simd_vec<float, 16> &rhs) {
        vec_ = _mm512_mul_ps(vec_, rhs.vec_);
        return *this;
    }

    force_inline simd_vec<float, 16> &operator*=(float rhs) {
        vec_ = _mm512_mul_ps(vec_, _mm512_set1_ps(rhs));
        return *this;
    }

    force_inline simd_vec<float, 16> &operator/=(const simd_vec<float, 16> &rhs) {
        vec_ = _mm512_div_ps(vec_, rhs.vec_);
        return *this;
    }

    force_inline simd_vec<float, 16> &operator/=(float rhs) {
        __m512 _rhs = _mm512_set1_ps(rhs);
        vec_ = _mm512_div_ps(vec_, _rhs);
        return *this;
    }

    force_inline simd_vec<float, 16> &operator|=(const simd_vec<float, 16> &rhs) {
        vec_ = _mm512_or_ps(vec_, rhs.vec_);
        return *this;
    }

    force_inline simd_vec<float, 16> &operator|=(const float rhs) {
        vec_ = _mm512_or_ps(vec_, _mm512_set1_ps(rhs));
        return *this;
    }

    force_inline simd_vec<float, 16> &operator&=(const simd_vec<float, 16> &rhs) {
        vec_ = _mm512_and_ps(vec_, rhs.vec_);
        return *this;
    }

    force_inline simd_vec<float, 16> operator~() const;
    force_inline simd_vec<float, 16> operator-() const;
    force_inline explicit operator simd_vec<int, 16>() const;

    force_inline simd_vec<float, 16> sqrt() const;
    force_inline simd_vec<float, 16> log() const;

    force_inline float length() const {
        float temp = 0;
        ITERATE_16({ temp += comp_[i] * comp_[i]; })
        return std::sqrt(temp);
    }

    force_inline float length2() const {
        float temp = 0;
        ITERATE_16({ temp += comp_[i] * comp_[i]; })
        return temp;
    }

    force_inline void copy_to(float *f) const { _mm512_storeu_ps(f, vec_); }
    force_inline void copy_to(float *f, simd_mem_aligned_tag) const { _mm512_store_ps(f, vec_); }

    force_inline void blend_to(const simd_vec<float, 16> &mask, const simd_vec<float, 16> &v1) {
#if VALIDATE_MASKS
        ITERATE_16({
            assert(reinterpret_cast<const uint32_t &>(mask.comp_[i]) == 0 ||
                   reinterpret_cast<const uint32_t &>(mask.comp_[i]) == 0xffffffff);
        })
#endif
        //__mmask16 msk =
        //    _mm512_fpclass_ps_mask(mask.vec_, 0x54); // 0x54 = Negative_Finite | Negative_Infinity | Negative_Zero
        //vec_ = _mm512_mask_blend_ps(msk, vec_, v1.vec_);
        vec_ = _mm512_blendv_ps(vec_, v1.vec_, mask.vec_);
    }

    force_inline void blend_inv_to(const simd_vec<float, 16> &mask, const simd_vec<float, 16> &v1) {
#if VALIDATE_MASKS
        ITERATE_16({
            assert(reinterpret_cast<const uint32_t &>(mask.comp_[i]) == 0 ||
                   reinterpret_cast<const uint32_t &>(mask.comp_[i]) == 0xffffffff);
        })
#endif
        //__mmask16 msk =
        //    _mm512_fpclass_ps_mask(mask.vec_, 0x54); // 0x54 = Negative_Finite | Negative_Infinity | Negative_Zero
        //vec_ = _mm512_mask_blend_ps(msk, v1.vec_, vec_);
        vec_ = _mm512_blendv_ps(v1.vec_, vec_, mask.vec_);
    }

    force_inline static simd_vec<float, 16> min(const simd_vec<float, 16> &v1, const simd_vec<float, 16> &v2);
    force_inline static simd_vec<float, 16> max(const simd_vec<float, 16> &v1, const simd_vec<float, 16> &v2);

    force_inline static simd_vec<float, 16> and_not(const simd_vec<float, 16> &v1, const simd_vec<float, 16> &v2);

    force_inline static simd_vec<float, 16> floor(const simd_vec<float, 16> &v1);

    force_inline static simd_vec<float, 16> ceil(const simd_vec<float, 16> &v1);

    friend force_inline simd_vec<float, 16> operator&(const simd_vec<float, 16> &v1, const simd_vec<float, 16> &v2);
    friend force_inline simd_vec<float, 16> operator|(const simd_vec<float, 16> &v1, const simd_vec<float, 16> &v2);
    friend force_inline simd_vec<float, 16> operator^(const simd_vec<float, 16> &v1, const simd_vec<float, 16> &v2);
    friend force_inline simd_vec<float, 16> operator+(const simd_vec<float, 16> &v1, const simd_vec<float, 16> &v2);
    friend force_inline simd_vec<float, 16> operator-(const simd_vec<float, 16> &v1, const simd_vec<float, 16> &v2);
    friend force_inline simd_vec<float, 16> operator*(const simd_vec<float, 16> &v1, const simd_vec<float, 16> &v2);
    friend force_inline simd_vec<float, 16> operator/(const simd_vec<float, 16> &v1, const simd_vec<float, 16> &v2);

    friend force_inline simd_vec<float, 16> operator+(const simd_vec<float, 16> &v1, float v2);
    friend force_inline simd_vec<float, 16> operator-(const simd_vec<float, 16> &v1, float v2);
    friend force_inline simd_vec<float, 16> operator*(const simd_vec<float, 16> &v1, float v2);
    friend force_inline simd_vec<float, 16> operator/(const simd_vec<float, 16> &v1, float v2);

    friend force_inline simd_vec<float, 16> operator+(float v1, const simd_vec<float, 16> &v2);
    friend force_inline simd_vec<float, 16> operator-(float v1, const simd_vec<float, 16> &v2);
    friend force_inline simd_vec<float, 16> operator*(float v1, const simd_vec<float, 16> &v2);
    friend force_inline simd_vec<float, 16> operator/(float v1, const simd_vec<float, 16> &v2);

    friend force_inline simd_vec<float, 16> operator<(const simd_vec<float, 16> &v1, const simd_vec<float, 16> &v2);
    friend force_inline simd_vec<float, 16> operator<=(const simd_vec<float, 16> &v1, const simd_vec<float, 16> &v2);
    friend force_inline simd_vec<float, 16> operator>(const simd_vec<float, 16> &v1, const simd_vec<float, 16> &v2);
    friend force_inline simd_vec<float, 16> operator>=(const simd_vec<float, 16> &v1, const simd_vec<float, 16> &v2);
    friend force_inline simd_vec<float, 16> operator==(const simd_vec<float, 16> &v1, const simd_vec<float, 16> &v2);
    friend force_inline simd_vec<float, 16> operator!=(const simd_vec<float, 16> &v1, const simd_vec<float, 16> &v2);

    friend force_inline simd_vec<float, 16> operator<(const simd_vec<float, 16> &v1, float v2);
    friend force_inline simd_vec<float, 16> operator<=(const simd_vec<float, 16> &v1, float v2);
    friend force_inline simd_vec<float, 16> operator>(const simd_vec<float, 16> &v1, float v2);
    friend force_inline simd_vec<float, 16> operator>=(const simd_vec<float, 16> &v1, float v2);
    friend force_inline simd_vec<float, 16> operator==(const simd_vec<float, 16> &v1, float v2);
    friend force_inline simd_vec<float, 16> operator!=(const simd_vec<float, 16> &v1, float v2);

    friend force_inline simd_vec<float, 16> clamp(const simd_vec<float, 16> &v1, float min, float max);
    friend force_inline simd_vec<float, 16> pow(const simd_vec<float, 16> &v1, const simd_vec<float, 16> &v2);

    friend force_inline simd_vec<float, 16> normalize(const simd_vec<float, 16> &v1);

    friend force_inline simd_vec<float, 16> fmadd(const simd_vec<float, 16> &a, const simd_vec<float, 16> &b,
                                                  const simd_vec<float, 16> &c);
    friend force_inline simd_vec<float, 16> fmadd(const simd_vec<float, 16> &a, const float b, const simd_vec<float, 16> &c);
    friend force_inline simd_vec<float, 16> fmadd(const float a, const simd_vec<float, 16> &b, const float c);

    friend force_inline simd_vec<float, 16> fmsub(const simd_vec<float, 16> &a, const simd_vec<float, 16> &b,
                                                  const simd_vec<float, 16> &c);
    friend force_inline simd_vec<float, 16> fmsub(const simd_vec<float, 16> &a, const float b, const simd_vec<float, 16> &c);
    friend force_inline simd_vec<float, 16> fmsub(const float a, const simd_vec<float, 16> &b, const float c);

    template <int Scale>
    friend force_inline simd_vec<float, 16> gather(const float *base_addr, const simd_vec<int, 16> &vindex);

    friend force_inline const float *value_ptr(const simd_vec<float, 16> &v1) { return &v1.comp_[0]; }

    static int size() { return 16; }
    static bool is_native() { return true; }
};

template <> class simd_vec<int, 16> {
    union {
        __m512i vec_;
        __m512 vec_ps_;
        int comp_[16];
    };

    friend class simd_vec<float, 16>;

  public:
    force_inline simd_vec() = default;
    force_inline simd_vec(int f) { vec_ = _mm512_set1_epi32(f); }
    force_inline simd_vec(int i0, int i1, int i2, int i3, int i4, int i5, int i6, int i7, int i8, int i9, int i10,
                          int i11, int i12, int i13, int i14, int i15) {
        vec_ = _mm512_setr_epi32(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15);
    }
    force_inline explicit simd_vec(const int *f) { vec_ = _mm512_loadu_si512((const __m512i *)f); }
    force_inline simd_vec(const int *f, simd_mem_aligned_tag) { vec_ = _mm512_load_si512((const __m512i *)f); }

    force_inline int &operator[](int i) { return comp_[i]; }
    force_inline const int &operator[](int i) const { return comp_[i]; }

    force_inline simd_vec<int, 16> &operator+=(const simd_vec<int, 16> &rhs) {
        vec_ = _mm512_add_epi32(vec_, rhs.vec_);
        return *this;
    }

    force_inline simd_vec<int, 16> &operator+=(int rhs) {
        vec_ = _mm512_add_epi32(vec_, _mm512_set1_epi32(rhs));
        return *this;
    }

    force_inline simd_vec<int, 16> &operator-=(const simd_vec<int, 16> &rhs) {
        vec_ = _mm512_sub_epi32(vec_, rhs.vec_);
        return *this;
    }

    force_inline simd_vec<int, 16> &operator-=(int rhs) {
        vec_ = _mm512_sub_epi32(vec_, _mm512_set1_epi32(rhs));
        return *this;
    }

    force_inline simd_vec<int, 16> &operator*=(const simd_vec<int, 16> &rhs) {
        ITERATE_16({ comp_[i] = comp_[i] * rhs.comp_[i]; })
        return *this;
    }

    force_inline simd_vec<int, 16> &operator*=(int rhs) {
        ITERATE_16({ comp_[i] = comp_[i] * rhs; })
        return *this;
    }

    force_inline simd_vec<int, 16> &operator/=(const simd_vec<int, 16> &rhs) {
        ITERATE_16({ comp_[i] = comp_[i] / rhs.comp_[i]; })
        return *this;
    }

    force_inline simd_vec<int, 16> &operator/=(int rhs) {
        ITERATE_16({ comp_[i] = comp_[i] / rhs; })
        return *this;
    }

    force_inline simd_vec<int, 16> &operator|=(const simd_vec<int, 16> &rhs) {
        vec_ = _mm512_or_si512(vec_, rhs.vec_);
        return *this;
    }

    force_inline simd_vec<int, 16> &operator|=(const int rhs) {
        vec_ = _mm512_or_si512(vec_, _mm512_set1_epi32(rhs));
        return *this;
    }

    force_inline simd_vec<int, 16> operator-() const {
        simd_vec<int, 16> temp;
        temp.vec_ = _mm512_sub_epi32(_mm512_setzero_si512(), vec_);
        return temp;
    }

    force_inline simd_vec<int, 16> operator==(int rhs) const {
        simd_vec<int, 16> ret;
        ret.vec_ = _mm512_movm_epi32(_mm512_cmpeq_epi32_mask(vec_, _mm512_set1_epi32(rhs)));
        return ret;
    }

    force_inline simd_vec<int, 16> operator==(const simd_vec<int, 16> &rhs) const {
        simd_vec<int, 16> ret;
        ret.vec_ = _mm512_movm_epi32(_mm512_cmpeq_epi32_mask(vec_, rhs.vec_));
        return ret;
    }

    force_inline simd_vec<int, 16> operator!=(int rhs) const {
        simd_vec<int, 16> ret;
        ret.vec_ = 
            _mm512_andnot_si512(_mm512_movm_epi32(_mm512_cmpeq_epi32_mask(vec_, _mm512_set1_epi32(rhs))),
                                                        _mm512_set1_epi32(~0));
        return ret;
    }

    force_inline simd_vec<int, 16> operator!=(const simd_vec<int, 16> &rhs) const {
        simd_vec<int, 16> ret;
        ret.vec_ =
            _mm512_andnot_si512(_mm512_movm_epi32(_mm512_cmpeq_epi32_mask(vec_, rhs.vec_)), _mm512_set1_epi32(~0));
        return ret;
    }

    force_inline simd_vec<int, 16> &operator&=(const simd_vec<int, 16> &rhs) {
        vec_ = _mm512_and_si512(vec_, rhs.vec_);
        return *this;
    }

    force_inline simd_vec<int, 16> &operator&=(const int rhs) {
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

    force_inline void blend_to(const simd_vec<int, 16> &mask, const simd_vec<int, 16> &v1) {
#if VALIDATE_MASKS
        ITERATE_16({ assert(mask.comp_[i] == 0 || mask.comp_[i] == -1); })
#endif
        vec_ = _mm512_ternarylogic_epi32(vec_, v1.vec_, _mm512_srai_epi32(mask.vec_, 31), 0xd8);
    }

    force_inline void blend_inv_to(const simd_vec<int, 16> &mask, const simd_vec<int, 16> &v1) {
#if VALIDATE_MASKS
        ITERATE_16({ assert(mask.comp_[i] == 0 || mask.comp_[i] == -1); })
#endif
        vec_ = _mm512_ternarylogic_epi32(v1.vec_, vec_, _mm512_srai_epi32(mask.vec_, 31), 0xd8);
    }

    force_inline int movemask() const { return _mm512_movemask_epi32(vec_);
    }

    force_inline bool all_zeros() const {
        return _mm512_cmpeq_epi32_mask(vec_, _mm512_setzero_si512()) == 0xFFFF;
    }

    force_inline bool all_zeros(const simd_vec<int, 16> &mask) const {
        return _mm512_cmpeq_epi32_mask(_mm512_and_si512(vec_, mask.vec_), _mm512_setzero_si512()) == 0xFFFF;
    }

    force_inline bool not_all_zeros() const {
        return !all_zeros();
    }

    force_inline static simd_vec<int, 16> min(const simd_vec<int, 16> &v1, const simd_vec<int, 16> &v2) {
        simd_vec<int, 16> temp;
        temp.vec_ = _mm512_min_epi32(v1.vec_, v2.vec_);
        return temp;
    }

    force_inline static simd_vec<int, 16> max(const simd_vec<int, 16> &v1, const simd_vec<int, 16> &v2) {
        simd_vec<int, 16> temp;
        temp.vec_ = _mm512_max_epi32(v1.vec_, v2.vec_);
        return temp;
    }

    force_inline static simd_vec<int, 16> and_not(const simd_vec<int, 16> &v1, const simd_vec<int, 16> &v2) {
        simd_vec<int, 16> temp;
        temp.vec_ps_ = _mm512_andnot_ps(v1.vec_ps_, v2.vec_ps_);
        return temp;
    }

    friend force_inline simd_vec<int, 16> operator&(const simd_vec<int, 16> &v1, const simd_vec<int, 16> &v2) {
        simd_vec<int, 16> temp;
        temp.vec_ps_ = _mm512_and_ps(v1.vec_ps_, v2.vec_ps_);
        return temp;
    }

    friend force_inline simd_vec<int, 16> operator|(const simd_vec<int, 16> &v1, const simd_vec<int, 16> &v2) {
        simd_vec<int, 16> temp;
        temp.vec_ps_ = _mm512_or_ps(v1.vec_ps_, v2.vec_ps_);
        return temp;
    }

    friend force_inline simd_vec<int, 16> operator^(const simd_vec<int, 16> &v1, const simd_vec<int, 16> &v2) {
        simd_vec<int, 16> temp;
        temp.vec_ps_ = _mm512_xor_ps(v1.vec_ps_, v2.vec_ps_);
        return temp;
    }

    friend force_inline simd_vec<int, 16> operator+(const simd_vec<int, 16> &v1, const simd_vec<int, 16> &v2) {
        simd_vec<int, 16> temp;
        temp.vec_ = _mm512_add_epi32(v1.vec_, v2.vec_);
        return temp;
    }

    friend force_inline simd_vec<int, 16> operator-(const simd_vec<int, 16> &v1, const simd_vec<int, 16> &v2) {
        simd_vec<int, 16> temp;
        temp.vec_ = _mm512_sub_epi32(v1.vec_, v2.vec_);
        return temp;
    }

    friend force_inline simd_vec<int, 16> operator*(const simd_vec<int, 16> &v1, const simd_vec<int, 16> &v2) {
        simd_vec<int, 16> temp;
        ITERATE_16({ temp.comp_[i] = v1.comp_[i] * v2.comp_[i]; })
        return temp;
    }

    friend force_inline simd_vec<int, 16> operator/(const simd_vec<int, 16> &v1, const simd_vec<int, 16> &v2) {
        simd_vec<int, 16> temp;
        ITERATE_16({ temp.comp_[i] = v1.comp_[i] / v2.comp_[i]; })
        return temp;
    }

    friend force_inline simd_vec<int, 16> operator+(const simd_vec<int, 16> &v1, int v2) {
        simd_vec<int, 16> temp;
        temp.vec_ = _mm512_add_epi32(v1.vec_, _mm512_set1_epi32(v2));
        return temp;
    }

    friend force_inline simd_vec<int, 16> operator-(const simd_vec<int, 16> &v1, int v2) {
        simd_vec<int, 16> temp;
        temp.vec_ = _mm512_sub_epi32(v1.vec_, _mm512_set1_epi32(v2));
        return temp;
    }

    friend force_inline simd_vec<int, 16> operator*(const simd_vec<int, 16> &v1, int v2) {
        simd_vec<int, 16> temp;
        ITERATE_16({ temp.comp_[i] = v1.comp_[i] * v2; })
        return temp;
    }

    friend force_inline simd_vec<int, 16> operator/(const simd_vec<int, 16> &v1, int v2) {
        simd_vec<int, 16> temp;
        ITERATE_16({ temp.comp_[i] = v1.comp_[i] / v2; })
        return temp;
    }

    friend force_inline simd_vec<int, 16> operator+(int v1, const simd_vec<int, 16> &v2) {
        simd_vec<int, 16> temp;
        temp.vec_ = _mm512_add_epi32(_mm512_set1_epi32(v1), v2.vec_);
        return temp;
    }

    friend force_inline simd_vec<int, 16> operator-(int v1, const simd_vec<int, 16> &v2) {
        simd_vec<int, 16> temp;
        temp.vec_ = _mm512_sub_epi32(_mm512_set1_epi32(v1), v2.vec_);
        return temp;
    }

    friend force_inline simd_vec<int, 16> operator*(int v1, const simd_vec<int, 16> &v2) {
        simd_vec<int, 16> temp;
        ITERATE_16({ temp.comp_[i] = v1 * v2.comp_[i]; })
        return temp;
    }

    friend force_inline simd_vec<int, 16> operator/(int v1, const simd_vec<int, 16> &v2) {
        simd_vec<int, 16> temp;
        ITERATE_16({ temp.comp_[i] = v1 / v2.comp_[i]; })
        return temp;
    }

    friend force_inline simd_vec<int, 16> operator<(const simd_vec<int, 16> &v1, const simd_vec<int, 16> &v2) {
        simd_vec<int, 16> ret;
        ret.vec_ = _mm512_movm_epi32(_mm512_cmpgt_epi32_mask(v2.vec_, v1.vec_));
        return ret;
    }

    friend force_inline simd_vec<int, 16> operator>(const simd_vec<int, 16> &v1, const simd_vec<int, 16> &v2) {
        simd_vec<int, 16> ret;
        ret.vec_ = _mm512_movm_epi32(_mm512_cmpgt_epi32_mask(v1.vec_, v2.vec_));
        return ret;
    }

    friend force_inline simd_vec<int, 16> operator>=(const simd_vec<int, 16> &v1, const simd_vec<int, 16> &v2) {
        simd_vec<int, 16> ret;
        ret.vec_ = _mm512_movm_epi32(_mm512_cmpge_epi32_mask(v1.vec_, v2.vec_));
        return ret;
    }

    friend force_inline simd_vec<int, 16> operator<(const simd_vec<int, 16> &v1, int v2) {
        simd_vec<int, 16> ret;
        ret.vec_ = _mm512_movm_epi32(_mm512_cmpgt_epi32_mask(_mm512_set1_epi32(v2), v1.vec_));
        return ret;
    }

    friend force_inline simd_vec<int, 16> operator<=(const simd_vec<int, 16> &v1, int v2) {
        simd_vec<int, 16> ret;
        ret.vec_ = _mm512_movm_epi32(_mm512_cmpge_epi32_mask(_mm512_set1_epi32(v2), v1.vec_));
        return ret;
    }

    friend force_inline simd_vec<int, 16> operator>(const simd_vec<int, 16> &v1, int v2) {
        simd_vec<int, 16> ret;
        ret.vec_ = _mm512_movm_epi32(_mm512_cmpgt_epi32_mask(v1.vec_, _mm512_set1_epi32(v2)));
        return ret;
    }

    friend force_inline simd_vec<int, 16> operator>=(const simd_vec<int, 16> &v1, int v2) {
        simd_vec<int, 16> ret;
        ret.vec_ = _mm512_movm_epi32(_mm512_cmpge_epi32_mask(v1.vec_, _mm512_set1_epi32(v2)));
        return ret;
    }

    friend force_inline simd_vec<int, 16> operator>>(const simd_vec<int, 16> &v1, const simd_vec<int, 16> &v2) {
        simd_vec<int, 16> ret;
        ret.vec_ = _mm512_srlv_epi32(v1.vec_, v2.vec_);
        return ret;
    }

    friend force_inline simd_vec<int, 16> operator>>(const simd_vec<int, 16> &v1, int v2) {
        simd_vec<int, 16> ret;
        ret.vec_ = _mm512_srli_epi32(v1.vec_, v2);
        return ret;
    }

    friend force_inline simd_vec<int, 16> operator<<(const simd_vec<int, 16> &v1, const simd_vec<int, 16> &v2) {
        simd_vec<int, 16> ret;
        ret.vec_ = _mm512_sllv_epi32(v1.vec_, v2.vec_);
        return ret;
    }

    friend force_inline simd_vec<int, 16> operator<<(const simd_vec<int, 16> &v1, int v2) {
        simd_vec<int, 16> ret;
        ret.vec_ = _mm512_slli_epi32(v1.vec_, v2);
        return ret;
    }

    force_inline simd_vec<int, 16> operator~() const {
        simd_vec<int, 16> ret;
        ret.vec_ = _mm512_andnot_si512(vec_, _mm512_set1_epi32(~0));
        return ret;
    }

    friend force_inline simd_vec<int, 16> srai(const simd_vec<int, 16> &v1, int v2) {
        simd_vec<int, 16> ret;
        ret.vec_ = _mm512_srai_epi32(v1.vec_, v2);
        return ret;
    }

    friend force_inline bool is_equal(const simd_vec<int, 16> &v1, const simd_vec<int, 16> &v2) {
        return _mm512_cmpeq_epi32_mask(v1.vec_, v2.vec_) == 0xFFFF;
    }

    template <int Scale>
    friend force_inline simd_vec<float, 16> gather(const float *base_addr, const simd_vec<int, 16> &vindex);
    template <int Scale>
    friend force_inline simd_vec<int, 16> gather(const int *base_addr, const simd_vec<int, 16> &vindex);

    static int size() { return 16; }
    static bool is_native() {
        return true;
    }
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
    simd_vec<float, 16> temp;
    ITERATE_16({ temp.comp_[i] = std::log(comp_[i]); })
    return temp;
}

force_inline simd_vec<float, 16> simd_vec<float, 16>::min(const simd_vec<float, 16> &v1,
                                                          const simd_vec<float, 16> &v2) {
    simd_vec<float, 16> temp;
    temp.vec_ = _mm512_min_ps(v1.vec_, v2.vec_);
    return temp;
}

force_inline simd_vec<float, 16> simd_vec<float, 16>::max(const simd_vec<float, 16> &v1,
                                                          const simd_vec<float, 16> &v2) {
    simd_vec<float, 16> temp;
    temp.vec_ = _mm512_max_ps(v1.vec_, v2.vec_);
    return temp;
}

force_inline static simd_vec<float, 16> and_not(const simd_vec<float, 16> &v1, const simd_vec<float, 16> &v2) {
    simd_vec<float, 16> temp;
    temp.vec_ = _mm512_andnot_ps(v1.vec_, v2.vec_);
    return temp;
}

force_inline static simd_vec<float, 16> floor(const simd_vec<float, 16> &v1) {
    simd_vec<float, 16> temp;
    temp.vec_ = _mm512_floor_ps(v1.vec_);
    return temp;
}

force_inline static simd_vec<float, 16> ceil(const simd_vec<float, 16> &v1) {
    simd_vec<float, 16> temp;
    temp.vec_ = _mm512_ceil_ps(v1.vec_);
    return temp;
}

force_inline simd_vec<float, 16> operator&(const simd_vec<float, 16> &v1, const simd_vec<float, 16> &v2) {
    simd_vec<float, 16> temp;
    temp.vec_ = _mm512_and_ps(v1.vec_, v2.vec_);
    return temp;
}

force_inline simd_vec<float, 16> operator|(const simd_vec<float, 16> &v1, const simd_vec<float, 16> &v2) {
    simd_vec<float, 16> temp;
    temp.vec_ = _mm512_or_ps(v1.vec_, v2.vec_);
    return temp;
}

force_inline simd_vec<float, 16> operator^(const simd_vec<float, 16> &v1, const simd_vec<float, 16> &v2) {
    simd_vec<float, 16> temp;
    temp.vec_ = _mm512_xor_ps(v1.vec_, v2.vec_);
    return temp;
}

force_inline simd_vec<float, 16> operator+(const simd_vec<float, 16> &v1, const simd_vec<float, 16> &v2) {
    simd_vec<float, 16> temp;
    temp.vec_ = _mm512_add_ps(v1.vec_, v2.vec_);
    return temp;
}

force_inline simd_vec<float, 16> operator-(const simd_vec<float, 16> &v1, const simd_vec<float, 16> &v2) {
    simd_vec<float, 16> temp;
    temp.vec_ = _mm512_sub_ps(v1.vec_, v2.vec_);
    return temp;
}

force_inline simd_vec<float, 16> operator*(const simd_vec<float, 16> &v1, const simd_vec<float, 16> &v2) {
    simd_vec<float, 16> temp;
    temp.vec_ = _mm512_mul_ps(v1.vec_, v2.vec_);
    return temp;
}

force_inline simd_vec<float, 16> operator/(const simd_vec<float, 16> &v1, const simd_vec<float, 16> &v2) {
    simd_vec<float, 16> temp;
    temp.vec_ = _mm512_div_ps(v1.vec_, v2.vec_);
    return temp;
}

force_inline simd_vec<float, 16> operator+(const simd_vec<float, 16> &v1, float v2) {
    simd_vec<float, 16> temp;
    temp.vec_ = _mm512_add_ps(v1.vec_, _mm512_set1_ps(v2));
    return temp;
}

force_inline simd_vec<float, 16> operator-(const simd_vec<float, 16> &v1, float v2) {
    simd_vec<float, 16> temp;
    temp.vec_ = _mm512_sub_ps(v1.vec_, _mm512_set1_ps(v2));
    return temp;
}

force_inline simd_vec<float, 16> operator*(const simd_vec<float, 16> &v1, float v2) {
    simd_vec<float, 16> temp;
    temp.vec_ = _mm512_mul_ps(v1.vec_, _mm512_set1_ps(v2));
    return temp;
}

force_inline simd_vec<float, 16> operator/(const simd_vec<float, 16> &v1, float v2) {
    simd_vec<float, 16> temp;
    temp.vec_ = _mm512_div_ps(v1.vec_, _mm512_set1_ps(v2));
    return temp;
}

force_inline simd_vec<float, 16> operator+(float v1, const simd_vec<float, 16> &v2) {
    simd_vec<float, 16> temp;
    temp.vec_ = _mm512_add_ps(_mm512_set1_ps(v1), v2.vec_);
    return temp;
}

force_inline simd_vec<float, 16> operator-(float v1, const simd_vec<float, 16> &v2) {
    simd_vec<float, 16> temp;
    temp.vec_ = _mm512_sub_ps(_mm512_set1_ps(v1), v2.vec_);
    return temp;
}

force_inline simd_vec<float, 16> operator*(float v1, const simd_vec<float, 16> &v2) {
    simd_vec<float, 16> temp;
    temp.vec_ = _mm512_mul_ps(_mm512_set1_ps(v1), v2.vec_);
    return temp;
}

force_inline simd_vec<float, 16> operator/(float v1, const simd_vec<float, 16> &v2) {
    simd_vec<float, 16> temp;
    temp.vec_ = _mm512_div_ps(_mm512_set1_ps(v1), v2.vec_);
    return temp;
}

force_inline simd_vec<float, 16> operator<(const simd_vec<float, 16> &v1, const simd_vec<float, 16> &v2) {
    simd_vec<float, 16> ret;
    ret.vec_ = _mm512_cmp_ps(v1.vec_, v2.vec_, _CMP_LT_OS);
    return ret;
}

force_inline simd_vec<float, 16> operator<=(const simd_vec<float, 16> &v1, const simd_vec<float, 16> &v2) {
    simd_vec<float, 16> ret;
    ret.vec_ = _mm512_cmp_ps(v1.vec_, v2.vec_, _CMP_LE_OS);
    return ret;
}

force_inline simd_vec<float, 16> operator>(const simd_vec<float, 16> &v1, const simd_vec<float, 16> &v2) {
    simd_vec<float, 16> ret;
    ret.vec_ = _mm512_cmp_ps(v1.vec_, v2.vec_, _CMP_GT_OS);
    return ret;
}

force_inline simd_vec<float, 16> operator>=(const simd_vec<float, 16> &v1, const simd_vec<float, 16> &v2) {
    simd_vec<float, 16> ret;
    ret.vec_ = _mm512_cmp_ps(v1.vec_, v2.vec_, _CMP_GE_OS);
    return ret;
}

force_inline simd_vec<float, 16> operator==(const simd_vec<float, 16> &v1, const simd_vec<float, 16> &v2) {
    simd_vec<float, 16> ret;
    ret.vec_ = _mm512_cmp_ps(v1.vec_, v2.vec_, _CMP_EQ_OS);
    return ret;
}

force_inline simd_vec<float, 16> operator!=(const simd_vec<float, 16> &v1, const simd_vec<float, 16> &v2) {
    simd_vec<float, 16> ret;
    ret.vec_ = _mm512_cmp_ps(v1.vec_, v2.vec_, _CMP_NEQ_OS);
    return ret;
}

force_inline simd_vec<float, 16> operator<(const simd_vec<float, 16> &v1, float v2) {
    simd_vec<float, 16> ret;
    ret.vec_ = _mm512_cmp_ps(v1.vec_, _mm512_set1_ps(v2), _CMP_LT_OS);
    return ret;
}

force_inline simd_vec<float, 16> operator<=(const simd_vec<float, 16> &v1, float v2) {
    simd_vec<float, 16> ret;
    ret.vec_ = _mm512_cmp_ps(v1.vec_, _mm512_set1_ps(v2), _CMP_LE_OS);
    return ret;
}

force_inline simd_vec<float, 16> operator>(const simd_vec<float, 16> &v1, float v2) {
    simd_vec<float, 16> ret;
    ret.vec_ = _mm512_cmp_ps(v1.vec_, _mm512_set1_ps(v2), _CMP_GT_OS);
    return ret;
}

force_inline simd_vec<float, 16> operator>=(const simd_vec<float, 16> &v1, float v2) {
    simd_vec<float, 16> ret;
    ret.vec_ = _mm512_cmp_ps(v1.vec_, _mm512_set1_ps(v2), _CMP_GE_OS);
    return ret;
}

force_inline simd_vec<float, 16> operator==(const simd_vec<float, 16> &v1, float v2) {
    simd_vec<float, 16> ret;
    ret.vec_ = _mm512_cmp_ps(v1.vec_, _mm512_set1_ps(v2), _CMP_EQ_OS);
    return ret;
}

force_inline simd_vec<float, 16> operator!=(const simd_vec<float, 16> &v1, float v2) {
    simd_vec<float, 16> ret;
    ret.vec_ = _mm512_cmp_ps(v1.vec_, _mm512_set1_ps(v2), _CMP_NEQ_OS);
    return ret;
}

force_inline simd_vec<float, 16> clamp(const simd_vec<float, 16> &v1, float min, float max) {
    simd_vec<float, 16> ret;
    ret.vec_ = _mm512_max_ps(_mm512_set1_ps(min), _mm512_min_ps(v1.vec_, _mm512_set1_ps(max)));
    return ret;
}

force_inline simd_vec<float, 16> pow(const simd_vec<float, 16> &v1, const simd_vec<float, 16> &v2) {
    simd_vec<float, 16> ret;
    ITERATE_16({ ret.comp_[i] = std::pow(v1.comp_[i], v2.comp_[i]); })
    return ret;
}

force_inline simd_vec<float, 16> normalize(const simd_vec<float, 16> &v1) { return v1 / v1.length(); }

force_inline simd_vec<float, 16> fmadd(const simd_vec<float, 16> &a, const simd_vec<float, 16> &b,
                                       const simd_vec<float, 16> &c) {
    simd_vec<float, 16> ret;
    ret.vec_ = _mm512_fmadd_ps(a.vec_, b.vec_, c.vec_);
    return ret;
}

force_inline simd_vec<float, 16> fmadd(const simd_vec<float, 16> &a, const float b, const simd_vec<float, 16> &c) {
    simd_vec<float, 16> ret;
    ret.vec_ = _mm512_fmadd_ps(a.vec_, _mm512_set1_ps(b), c.vec_);
    return ret;
}

force_inline simd_vec<float, 16> fmadd(const float a, const simd_vec<float, 16> &b, const float c) {
    simd_vec<float, 16> ret;
    ret.vec_ = _mm512_fmadd_ps(_mm512_set1_ps(a), b.vec_, _mm512_set1_ps(c));
    return ret;
}

force_inline simd_vec<float, 16> fmsub(const simd_vec<float, 16> &a, const simd_vec<float, 16> &b,
                                       const simd_vec<float, 16> &c) {
    simd_vec<float, 16> ret;
    ret.vec_ = _mm512_fmsub_ps(a.vec_, b.vec_, c.vec_);
    return ret;
}

force_inline simd_vec<float, 16> fmsub(const simd_vec<float, 16> &a, const float b, const simd_vec<float, 16> &c) {
    simd_vec<float, 16> ret;
    ret.vec_ = _mm512_fmsub_ps(a.vec_, _mm512_set1_ps(b), c.vec_);
    return ret;
}

force_inline simd_vec<float, 16> fmsub(const float a, const simd_vec<float, 16> &b, const float c) {
    simd_vec<float, 16> ret;
    ret.vec_ = _mm512_fmsub_ps(_mm512_set1_ps(a), b.vec_, _mm512_set1_ps(c));
    return ret;
}

template <int Scale>
force_inline simd_vec<float, 16> gather(const float *base_addr, const simd_vec<int, 16> &vindex) {
    simd_vec<float, 16> ret;
    ret.vec_ = _mm512_i32gather_ps(vindex.vec_, base_addr, Scale * sizeof(float));
    return ret;
}

template <int Scale> force_inline simd_vec<int, 16> gather(const int *base_addr, const simd_vec<int, 16> &vindex) {
    simd_vec<int, 16> ret;
    ret.vec_ = _mm512_i32gather_epi32(vindex.vec_, base_addr, Scale * sizeof(int));
    return ret;
}

} // namespace NS
} // namespace Ray

#pragma warning(pop)

#ifdef __GNUC__
#pragma GCC pop_options
#pragma clang attribute pop
#endif
