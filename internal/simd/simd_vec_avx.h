//#pragma once

#include "simd_vec_sse.h"

#include <immintrin.h>

#ifdef __GNUC__
#pragma GCC push_options
#if defined(USE_AVX2) || defined(USE_AVX512)
#pragma GCC target("avx2")
#pragma clang attribute push(__attribute__((target("avx2"))), apply_to = function)
#else
#pragma GCC target("avx")
#pragma clang attribute push(__attribute__((target("avx"))), apply_to = function)
#endif
#endif

#if defined(__GNUC__)
#define _mm256_test_all_zeros(mask, val) _mm256_testz_si256((mask), (val))
#endif

#pragma warning(push)
#pragma warning(disable : 4752)

namespace Ray {
namespace NS {

template <> class simd_vec<float, 8> {
  public:
    union {
        __m256 vec_;
        float comp_[8];
    };

    friend class simd_vec<int, 8>;

  public:
    force_inline simd_vec() = default;
    force_inline simd_vec(float f) { vec_ = _mm256_set1_ps(f); }
    force_inline simd_vec(float f1, float f2, float f3, float f4, float f5, float f6, float f7, float f8) {
        vec_ = _mm256_setr_ps(f1, f2, f3, f4, f5, f6, f7, f8);
    }
    force_inline explicit simd_vec(const float *f) { vec_ = _mm256_loadu_ps(f); }
    force_inline simd_vec(const float *f, simd_mem_aligned_tag) { vec_ = _mm256_load_ps(f); }

    force_inline float &operator[](int i) { return comp_[i]; }
    force_inline const float &operator[](int i) const { return comp_[i]; }

    force_inline simd_vec<float, 8> &operator+=(const simd_vec<float, 8> &rhs) {
        vec_ = _mm256_add_ps(vec_, rhs.vec_);
        return *this;
    }

    force_inline simd_vec<float, 8> &operator+=(float rhs) {
        __m256 _rhs = _mm256_set1_ps(rhs);
        vec_ = _mm256_add_ps(vec_, _rhs);
        return *this;
    }

    force_inline simd_vec<float, 8> &operator-=(const simd_vec<float, 8> &rhs) {
        vec_ = _mm256_sub_ps(vec_, rhs.vec_);
        return *this;
    }

    force_inline simd_vec<float, 8> &operator-=(float rhs) {
        vec_ = _mm256_sub_ps(vec_, _mm256_set1_ps(rhs));
        return *this;
    }

    force_inline simd_vec<float, 8> &operator*=(const simd_vec<float, 8> &rhs) {
        vec_ = _mm256_mul_ps(vec_, rhs.vec_);
        return *this;
    }

    force_inline simd_vec<float, 8> &operator*=(float rhs) {
        vec_ = _mm256_mul_ps(vec_, _mm256_set1_ps(rhs));
        return *this;
    }

    force_inline simd_vec<float, 8> &operator/=(const simd_vec<float, 8> &rhs) {
        vec_ = _mm256_div_ps(vec_, rhs.vec_);
        return *this;
    }

    force_inline simd_vec<float, 8> &operator/=(float rhs) {
        __m256 _rhs = _mm256_set1_ps(rhs);
        vec_ = _mm256_div_ps(vec_, _rhs);
        return *this;
    }

    force_inline simd_vec<float, 8> &operator|=(const simd_vec<float, 8> &rhs) {
        vec_ = _mm256_or_ps(vec_, rhs.vec_);
        return *this;
    }

    force_inline simd_vec<float, 8> &operator|=(float rhs) {
        __m256 _rhs = _mm256_set1_ps(rhs);
        vec_ = _mm256_or_ps(vec_, _rhs);
        return *this;
    }

    force_inline simd_vec<float, 8> operator~() const;
    force_inline simd_vec<float, 8> operator-() const;
    force_inline operator simd_vec<int, 8>() const;

    force_inline simd_vec<float, 8> sqrt() const;

    force_inline float length() const {
        float temp = 0;
        ITERATE_8({ temp += comp_[i] * comp_[i]; })
        return std::sqrt(temp);
    }

    force_inline float length2() const {
        float temp = 0;
        ITERATE_8({ temp += comp_[i] * comp_[i]; })
        return temp;
    }

    force_inline void copy_to(float *f) const { _mm256_storeu_ps(f, vec_); }

    force_inline void copy_to(float *f, simd_mem_aligned_tag) const { _mm256_store_ps(f, vec_); }

    force_inline void blend_to(const simd_vec<float, 8> &mask, const simd_vec<float, 8> &v1) {
        vec_ = _mm256_blendv_ps(vec_, v1.vec_, mask.vec_);
    }

    force_inline void blend_inv_to(const simd_vec<float, 8> &mask, const simd_vec<float, 8> &v1) {
        vec_ = _mm256_blendv_ps(v1.vec_, vec_, mask.vec_);
    }

    force_inline static simd_vec<float, 8> min(const simd_vec<float, 8> &v1, const simd_vec<float, 8> &v2);
    force_inline static simd_vec<float, 8> min(const simd_vec<float, 8> &v1, float v2);
    force_inline static simd_vec<float, 8> max(const simd_vec<float, 8> &v1, const simd_vec<float, 8> &v2);
    force_inline static simd_vec<float, 8> max(const simd_vec<float, 8> &v1, float v2);

    force_inline static simd_vec<float, 8> and_not(const simd_vec<float, 8> &v1, const simd_vec<float, 8> &v2);

    force_inline static simd_vec<float, 8> floor(const simd_vec<float, 8> &v1);

    force_inline static simd_vec<float, 8> ceil(const simd_vec<float, 8> &v1);

    friend force_inline simd_vec<float, 8> operator&(const simd_vec<float, 8> &v1, const simd_vec<float, 8> &v2);
    friend force_inline simd_vec<float, 8> operator|(const simd_vec<float, 8> &v1, const simd_vec<float, 8> &v2);
    friend force_inline simd_vec<float, 8> operator^(const simd_vec<float, 8> &v1, const simd_vec<float, 8> &v2);
    friend force_inline simd_vec<float, 8> operator+(const simd_vec<float, 8> &v1, const simd_vec<float, 8> &v2);
    friend force_inline simd_vec<float, 8> operator-(const simd_vec<float, 8> &v1, const simd_vec<float, 8> &v2);
    friend force_inline simd_vec<float, 8> operator*(const simd_vec<float, 8> &v1, const simd_vec<float, 8> &v2);
    friend force_inline simd_vec<float, 8> operator/(const simd_vec<float, 8> &v1, const simd_vec<float, 8> &v2);

    friend force_inline simd_vec<float, 8> operator+(const simd_vec<float, 8> &v1, float v2);
    friend force_inline simd_vec<float, 8> operator-(const simd_vec<float, 8> &v1, float v2);
    friend force_inline simd_vec<float, 8> operator*(const simd_vec<float, 8> &v1, float v2);
    friend force_inline simd_vec<float, 8> operator/(const simd_vec<float, 8> &v1, float v2);

    friend force_inline simd_vec<float, 8> operator+(float v1, const simd_vec<float, 8> &v2);
    friend force_inline simd_vec<float, 8> operator-(float v1, const simd_vec<float, 8> &v2);
    friend force_inline simd_vec<float, 8> operator*(float v1, const simd_vec<float, 8> &v2);
    friend force_inline simd_vec<float, 8> operator/(float v1, const simd_vec<float, 8> &v2);

    friend force_inline simd_vec<float, 8> operator<(const simd_vec<float, 8> &v1, const simd_vec<float, 8> &v2);
    friend force_inline simd_vec<float, 8> operator<=(const simd_vec<float, 8> &v1, const simd_vec<float, 8> &v2);
    friend force_inline simd_vec<float, 8> operator>(const simd_vec<float, 8> &v1, const simd_vec<float, 8> &v2);
    friend force_inline simd_vec<float, 8> operator>=(const simd_vec<float, 8> &v1, const simd_vec<float, 8> &v2);
    friend force_inline simd_vec<float, 8> operator==(const simd_vec<float, 8> &v1, const simd_vec<float, 8> &v2);
    friend force_inline simd_vec<float, 8> operator!=(const simd_vec<float, 8> &v1, const simd_vec<float, 8> &v2);

    friend force_inline simd_vec<float, 8> operator<(const simd_vec<float, 8> &v1, float v2);
    friend force_inline simd_vec<float, 8> operator<=(const simd_vec<float, 8> &v1, float v2);
    friend force_inline simd_vec<float, 8> operator>(const simd_vec<float, 8> &v1, float v2);
    friend force_inline simd_vec<float, 8> operator>=(const simd_vec<float, 8> &v1, float v2);
    friend force_inline simd_vec<float, 8> operator==(const simd_vec<float, 8> &v1, float v2);
    friend force_inline simd_vec<float, 8> operator!=(const simd_vec<float, 8> &v1, float v2);

    friend force_inline simd_vec<float, 8> clamp(const simd_vec<float, 8> &v1, float min, float max);
    friend force_inline simd_vec<float, 8> pow(const simd_vec<float, 8> &v1, const simd_vec<float, 8> &v2);

    friend force_inline simd_vec<float, 8> normalize(const simd_vec<float, 8> &v1);

#if defined(USE_AVX2) || defined(USE_AVX512)
    friend force_inline simd_vec<float, 8> fmadd(const simd_vec<float, 8> &a, const simd_vec<float, 8> &b,
                                                 const simd_vec<float, 8> &c);
    friend force_inline simd_vec<float, 8> fmadd(const simd_vec<float, 8> &a, const float b,
                                                 const simd_vec<float, 8> &c);
    friend force_inline simd_vec<float, 8> fmadd(const float a, const simd_vec<float, 8> &b, const float c);

    friend force_inline simd_vec<float, 8> fmsub(const simd_vec<float, 8> &a, const simd_vec<float, 8> &b,
                                                 const simd_vec<float, 8> &c);
    friend force_inline simd_vec<float, 8> fmsub(const simd_vec<float, 8> &a, const float b,
                                                 const simd_vec<float, 8> &c);
    friend force_inline simd_vec<float, 8> fmsub(const float a, const simd_vec<float, 8> &b, const float c);

    template <int Scale>
    friend force_inline simd_vec<float, 8> gather(const float *base_addr, const simd_vec<int, 8> &vindex);
#endif

    friend force_inline const float *value_ptr(const simd_vec<float, 8> &v1) { return &v1.comp_[0]; }

    static int size() { return 8; }
    static bool is_native() { return true; }
};

template <> class simd_vec<int, 8> {
    union {
        __m256i vec_;
        __m256 vec_ps_;
        int comp_[8];
    };

    friend class simd_vec<float, 8>;

  public:
    force_inline simd_vec() = default;
    force_inline simd_vec(int f) { vec_ = _mm256_set1_epi32(f); }
    force_inline simd_vec(int i1, int i2, int i3, int i4, int i5, int i6, int i7, int i8) {
        vec_ = _mm256_setr_epi32(i1, i2, i3, i4, i5, i6, i7, i8);
    }
    force_inline explicit simd_vec(const int *f) { vec_ = _mm256_loadu_si256((const __m256i *)f); }
    force_inline simd_vec(const int *f, simd_mem_aligned_tag) { vec_ = _mm256_load_si256((const __m256i *)f); }

    force_inline int &operator[](int i) { return comp_[i]; }
    force_inline const int &operator[](int i) const { return comp_[i]; }

    force_inline simd_vec<int, 8> &operator+=(const simd_vec<int, 8> &rhs) {
#if defined(USE_AVX2) || defined(USE_AVX512)
        vec_ = _mm256_add_epi32(vec_, rhs.vec_);
#else
        ITERATE_8({ comp_[i] = comp_[i] + rhs.comp_[i]; })
#endif
        return *this;
    }

    force_inline simd_vec<int, 8> &operator+=(int rhs) {
#if defined(USE_AVX2) || defined(USE_AVX512)
        vec_ = _mm256_add_epi32(vec_, _mm256_set1_epi32(rhs));
#else
        ITERATE_8({ comp_[i] = comp_[i] + rhs; })
#endif
        return *this;
    }

    force_inline simd_vec<int, 8> &operator-=(const simd_vec<int, 8> &rhs) {
#if defined(USE_AVX2) || defined(USE_AVX512)
        vec_ = _mm256_sub_epi32(vec_, rhs.vec_);
#else
        ITERATE_8({ comp_[i] = comp_[i] - rhs.comp_[i]; })
#endif
        return *this;
    }

    force_inline simd_vec<int, 8> &operator-=(int rhs) {
#if defined(USE_AVX2) || defined(USE_AVX512)
        vec_ = _mm256_sub_epi32(vec_, _mm256_set1_epi32(rhs));
#else
        ITERATE_8({ comp_[i] = comp_[i] - rhs; })
#endif
        return *this;
    }

    force_inline simd_vec<int, 8> &operator*=(const simd_vec<int, 8> &rhs) {
#if defined(USE_AVX2) || defined(USE_AVX512)
        vec_ = _mm256_mullo_epi32(vec_, rhs.vec_);
#else
        ITERATE_8({ comp_[i] = comp_[i] * rhs.comp_[i]; })
#endif
        return *this;
    }

    force_inline simd_vec<int, 8> &operator*=(int rhs) {
#if defined(USE_AVX2) || defined(USE_AVX512)
        vec_ = _mm256_mullo_epi32(vec_, _mm256_set1_epi32(rhs));
#else
        ITERATE_8({ comp_[i] = comp_[i] * rhs; })
#endif
        return *this;
    }

    force_inline simd_vec<int, 8> &operator/=(const simd_vec<int, 8> &rhs) {
        ITERATE_8({ comp_[i] = comp_[i] / rhs.comp_[i]; })
        return *this;
    }

    force_inline simd_vec<int, 8> &operator/=(int rhs) {
        ITERATE_8({ comp_[i] = comp_[i] / rhs; })
        return *this;
    }

    force_inline simd_vec<int, 8> &operator|=(const simd_vec<int, 8> &rhs) {
#if defined(USE_AVX2) || defined(USE_AVX512)
        vec_ = _mm256_or_si256(vec_, rhs.vec_);
#else
        ITERATE_8({ comp_[i] = comp_[i] | rhs.comp_[i]; })
#endif
        return *this;
    }

    force_inline simd_vec<int, 8> &operator|=(int rhs) {
#if defined(USE_AVX2) || defined(USE_AVX512)
        vec_ = _mm256_or_si256(vec_, _mm256_set1_epi32(rhs));
#else
        ITERATE_8({ comp_[i] = comp_[i] | rhs; })
#endif
        return *this;
    }

    force_inline simd_vec<int, 8> operator-() const {
        simd_vec<int, 8> temp;
        temp.vec_ = _mm256_sub_epi32(_mm256_setzero_si256(), vec_);
        return temp;
    }

    force_inline simd_vec<int, 8> operator==(int rhs) const {
        simd_vec<int, 8> ret;
#if defined(USE_AVX2) || defined(USE_AVX512)
        ret.vec_ = _mm256_cmpeq_epi32(vec_, _mm256_set1_epi32(rhs));
#else
        ITERATE_8({ ret.comp_[i] = comp_[i] == rhs ? 0xFFFFFFFF : 0; })
#endif
        return ret;
    }

    force_inline simd_vec<int, 8> operator==(const simd_vec<int, 8> &rhs) const {
        simd_vec<int, 8> ret;
#if defined(USE_AVX2) || defined(USE_AVX512)
        ret.vec_ = _mm256_cmpeq_epi32(vec_, rhs.vec_);
#else
        ITERATE_8({ ret.comp_[i] = comp_[i] == rhs.comp_[i] ? 0xFFFFFFFF : 0; })
#endif
        return ret;
    }

    force_inline simd_vec<int, 8> operator!=(int rhs) const {
        simd_vec<int, 8> ret;
#if defined(USE_AVX2) || defined(USE_AVX512)
        ret.vec_ = _mm256_andnot_si256(_mm256_cmpeq_epi32(vec_, _mm256_set1_epi32(rhs)), _mm256_set1_epi32(~0));
#else
        ITERATE_8({ ret.comp_[i] = comp_[i] != rhs ? 0xFFFFFFFF : 0; })
#endif
        return ret;
    }

    force_inline simd_vec<int, 8> operator!=(const simd_vec<int, 8> &rhs) const {
        simd_vec<int, 8> ret;
#if defined(USE_AVX2) || defined(USE_AVX512)
        ret.vec_ = _mm256_andnot_si256(_mm256_cmpeq_epi32(vec_, rhs.vec_), _mm256_set1_epi32(~0));
#else
        ITERATE_8({ ret.comp_[i] = comp_[i] != rhs.comp_[i] ? 0xFFFFFFFF : 0; })
#endif
        return ret;
    }

    force_inline simd_vec<int, 8> &operator&=(const simd_vec<int, 8> &rhs) {
        vec_ = _mm256_and_si256(vec_, rhs.vec_);
        return *this;
    }

    force_inline simd_vec<int, 8> &operator&=(const int rhs) {
        vec_ = _mm256_and_si256(vec_, _mm256_set1_epi32(rhs));
        return *this;
    }

    force_inline operator simd_vec<float, 8>() const {
        simd_vec<float, 8> ret;
        ret.vec_ = _mm256_cvtepi32_ps(vec_);
        return ret;
    }

    force_inline void copy_to(int *f) const { _mm256_storeu_si256((__m256i *)f, vec_); }

    force_inline void copy_to(int *f, simd_mem_aligned_tag) const { _mm256_store_si256((__m256i *)f, vec_); }

    force_inline void blend_to(const simd_vec<int, 8> &mask, const simd_vec<int, 8> &v1) {
        vec_ = _mm256_castps_si256(
            _mm256_blendv_ps(_mm256_castsi256_ps(vec_), _mm256_castsi256_ps(v1.vec_), _mm256_castsi256_ps(mask.vec_)));
    }

    force_inline void blend_inv_to(const simd_vec<int, 8> &mask, const simd_vec<int, 8> &v1) {
        vec_ = _mm256_castps_si256(
            _mm256_blendv_ps(_mm256_castsi256_ps(v1.vec_), _mm256_castsi256_ps(vec_), _mm256_castsi256_ps(mask.vec_)));
    }

    force_inline int movemask() const { return _mm256_movemask_ps(_mm256_castsi256_ps(vec_)); }

    force_inline bool all_zeros() const { return _mm256_test_all_zeros(vec_, vec_) != 0; }

    force_inline bool all_zeros(const simd_vec<int, 8> &mask) const {
        return _mm256_test_all_zeros(vec_, mask.vec_) != 0;
    }

    force_inline bool not_all_zeros() const {
        int res = _mm256_test_all_zeros(vec_, vec_);
        return res == 0;
    }

    force_inline static simd_vec<int, 8> min(const simd_vec<int, 8> &v1, const simd_vec<int, 8> &v2) {
        simd_vec<int, 8> temp;
#if defined(USE_AVX2) || defined(USE_AVX512)
        temp.vec_ = _mm256_min_epi32(v1.vec_, v2.vec_);
#else
        ITERATE_8({ temp.comp_[i] = std::min(v1.comp_[i], v2.comp_[i]); })
#endif
        return temp;
    }

    force_inline static simd_vec<int, 8> min(const simd_vec<int, 8> &v1, const int v2) {
        simd_vec<int, 8> temp;
#if defined(USE_AVX2) || defined(USE_AVX512)
        temp.vec_ = _mm256_min_epi32(v1.vec_, _mm256_set1_epi32(v2));
#else
        ITERATE_8({ temp.comp_[i] = std::min(v1.comp_[i], v2); })
#endif
        return temp;
    }

    force_inline static simd_vec<int, 8> max(const simd_vec<int, 8> &v1, const simd_vec<int, 8> &v2) {
        simd_vec<int, 8> temp;
#if defined(USE_AVX2) || defined(USE_AVX512)
        temp.vec_ = _mm256_max_epi32(v1.vec_, v2.vec_);
#else
        ITERATE_8({ temp.comp_[i] = std::max(v1.comp_[i], v2.comp_[i]); })
#endif
        return temp;
    }

    force_inline static simd_vec<int, 8> max(const simd_vec<int, 8> &v1, const int v2) {
        simd_vec<int, 8> temp;
#if defined(USE_AVX2) || defined(USE_AVX512)
        temp.vec_ = _mm256_max_epi32(v1.vec_, _mm256_set1_epi32(v2));
#else
        ITERATE_8({ temp.comp_[i] = std::max(v1.comp_[i], v2); })
#endif
        return temp;
    }

    force_inline static simd_vec<int, 8> and_not(const simd_vec<int, 8> &v1, const simd_vec<int, 8> &v2) {
        simd_vec<int, 8> temp;
        temp.vec_ps_ = _mm256_andnot_ps(v1.vec_ps_, v2.vec_ps_);
        return temp;
    }

    friend force_inline simd_vec<int, 8> operator&(const simd_vec<int, 8> &v1, const simd_vec<int, 8> &v2) {
        simd_vec<int, 8> temp;
        temp.vec_ps_ = _mm256_and_ps(v1.vec_ps_, v2.vec_ps_);
        return temp;
    }

    friend force_inline simd_vec<int, 8> operator|(const simd_vec<int, 8> &v1, const simd_vec<int, 8> &v2) {
        simd_vec<int, 8> temp;
        temp.vec_ps_ = _mm256_or_ps(v1.vec_ps_, v2.vec_ps_);
        return temp;
    }

    friend force_inline simd_vec<int, 8> operator^(const simd_vec<int, 8> &v1, const simd_vec<int, 8> &v2) {
        simd_vec<int, 8> temp;
        temp.vec_ps_ = _mm256_xor_ps(v1.vec_ps_, v2.vec_ps_);
        return temp;
    }

    friend force_inline simd_vec<int, 8> operator+(const simd_vec<int, 8> &v1, const simd_vec<int, 8> &v2) {
        simd_vec<int, 8> temp;
#if defined(USE_AVX2) || defined(USE_AVX512)
        temp.vec_ = _mm256_add_epi32(v1.vec_, v2.vec_);
#else
        ITERATE_8({ temp.comp_[i] = v1.comp_[i] + v2.comp_[i]; })
#endif
        return temp;
    }

    friend force_inline simd_vec<int, 8> operator-(const simd_vec<int, 8> &v1, const simd_vec<int, 8> &v2) {
        simd_vec<int, 8> temp;
#if defined(USE_AVX2) || defined(USE_AVX512)
        temp.vec_ = _mm256_sub_epi32(v1.vec_, v2.vec_);
#else
        ITERATE_8({ temp.comp_[i] = v1.comp_[i] - v2.comp_[i]; })
#endif
        return temp;
    }

    friend force_inline simd_vec<int, 8> operator*(const simd_vec<int, 8> &v1, const simd_vec<int, 8> &v2) {
        simd_vec<int, 8> temp;
#if defined(USE_AVX2) || defined(USE_AVX512)
        temp.vec_ = _mm256_mullo_epi32(v1.vec_, v2.vec_);
#else
        ITERATE_8({ temp.comp_[i] = v1.comp_[i] * v2.comp_[i]; })
#endif
        return temp;
    }

    friend force_inline simd_vec<int, 8> operator/(const simd_vec<int, 8> &v1, const simd_vec<int, 8> &v2) {
        simd_vec<int, 8> temp;
        ITERATE_8({ temp.comp_[i] = v1.comp_[i] / v2.comp_[i]; })
        return temp;
    }

    friend force_inline simd_vec<int, 8> operator+(const simd_vec<int, 8> &v1, int v2) {
        simd_vec<int, 8> temp;
#if defined(USE_AVX2) || defined(USE_AVX512)
        temp.vec_ = _mm256_add_epi32(v1.vec_, _mm256_set1_epi32(v2));
#else
        ITERATE_8({ temp.comp_[i] = v1.comp_[i] + v2; })
#endif
        return temp;
    }

    friend force_inline simd_vec<int, 8> operator-(const simd_vec<int, 8> &v1, int v2) {
        simd_vec<int, 8> temp;
#if defined(USE_AVX2) || defined(USE_AVX512)
        temp.vec_ = _mm256_sub_epi32(v1.vec_, _mm256_set1_epi32(v2));
#else
        ITERATE_8({ temp.comp_[i] = v1.comp_[i] - v2; })
#endif
        return temp;
    }

    friend force_inline simd_vec<int, 8> operator*(const simd_vec<int, 8> &v1, int v2) {
        simd_vec<int, 8> temp;
#if defined(USE_AVX2) || defined(USE_AVX512)
        temp.vec_ = _mm256_mullo_epi32(v1.vec_, _mm256_set1_epi32(v2));
#else
        ITERATE_8({ temp.comp_[i] = v1.comp_[i] * v2; })
#endif
        return temp;
    }

    friend force_inline simd_vec<int, 8> operator/(const simd_vec<int, 8> &v1, int v2) {
        simd_vec<int, 8> temp;
        ITERATE_8({ temp.comp_[i] = v1.comp_[i] / v2; })
        return temp;
    }

    friend force_inline simd_vec<int, 8> operator+(int v1, const simd_vec<int, 8> &v2) {
        simd_vec<int, 8> temp;
#if defined(USE_AVX2) || defined(USE_AVX512)
        temp.vec_ = _mm256_add_epi32(_mm256_set1_epi32(v1), v2.vec_);
#else
        ITERATE_8({ temp.comp_[i] = v1 + v2.comp_[i]; })
#endif
        return temp;
    }

    friend force_inline simd_vec<int, 8> operator-(int v1, const simd_vec<int, 8> &v2) {
        simd_vec<int, 8> temp;
#if defined(USE_AVX2) || defined(USE_AVX512)
        temp.vec_ = _mm256_sub_epi32(_mm256_set1_epi32(v1), v2.vec_);
#else
        ITERATE_8({ temp.comp_[i] = v1 - v2.comp_[i]; })
#endif
        return temp;
    }

    friend force_inline simd_vec<int, 8> operator*(int v1, const simd_vec<int, 8> &v2) {
        simd_vec<int, 8> temp;
#if defined(USE_AVX2) || defined(USE_AVX512)
        temp.vec_ = _mm256_mullo_epi32(_mm256_set1_epi32(v1), v2.vec_);
#else
        ITERATE_8({ temp.comp_[i] = v1 * v2.comp_[i]; })
#endif
        return temp;
    }

    friend force_inline simd_vec<int, 8> operator/(int v1, const simd_vec<int, 8> &v2) {
        simd_vec<int, 8> temp;
        ITERATE_8({ temp.comp_[i] = v1 / v2.comp_[i]; })
        return temp;
    }

    friend force_inline simd_vec<int, 8> operator<(const simd_vec<int, 8> &v1, const simd_vec<int, 8> &v2) {
        simd_vec<int, 8> ret;
#if defined(USE_AVX2) || defined(USE_AVX512)
        ret.vec_ = _mm256_cmpgt_epi32(v2.vec_, v1.vec_);
#else
        ITERATE_8({ ret.comp_[i] = v1.comp_[i] < v2.comp_[i] ? 0xFFFFFFFF : 0; })
#endif
        return ret;
    }

    friend force_inline simd_vec<int, 8> operator>(const simd_vec<int, 8> &v1, const simd_vec<int, 8> &v2) {
        simd_vec<int, 8> ret;
#if defined(USE_AVX2) || defined(USE_AVX512)
        ret.vec_ = _mm256_cmpgt_epi32(v1.vec_, v2.vec_);
#else
        ITERATE_8({ ret.comp_[i] = v1.comp_[i] > v2.comp_[i] ? 0xFFFFFFFF : 0; })
#endif
        return ret;
    }

    friend force_inline simd_vec<int, 8> operator<(const simd_vec<int, 8> &v1, int v2) {
        simd_vec<int, 8> ret;
#if defined(USE_AVX2) || defined(USE_AVX512)
        ret.vec_ = _mm256_cmpgt_epi32(_mm256_set1_epi32(v2), v1.vec_);
#else
        ITERATE_8({ ret.comp_[i] = v1.comp_[i] < v2 ? 0xFFFFFFFF : 0; })
#endif
        return ret;
    }

    friend force_inline simd_vec<int, 8> operator<=(const simd_vec<int, 8> &v1, int v2) {
        simd_vec<int, 8> ret;
#if defined(USE_AVX2) || defined(USE_AVX512)
        ret.vec_ = _mm256_or_si256(_mm256_cmpeq_epi32(_mm256_set1_epi32(v2), v1.vec_),
                                   _mm256_cmpgt_epi32(_mm256_set1_epi32(v2), v1.vec_));
#else
        ITERATE_8({ ret.comp_[i] = v1.comp_[i] <= v2 ? 0xFFFFFFFF : 0; })
#endif
        return ret;
    }

    friend force_inline simd_vec<int, 8> operator>(const simd_vec<int, 8> &v1, int v2) {
        simd_vec<int, 8> ret;
#if defined(USE_AVX2) || defined(USE_AVX512)
        ret.vec_ = _mm256_cmpgt_epi32(v1.vec_, _mm256_set1_epi32(v2));
#else
        ITERATE_8({ ret.comp_[i] = v1.comp_[i] > v2 ? 0xFFFFFFFF : 0; })
#endif
        return ret;
    }

    friend force_inline simd_vec<int, 8> operator>>(const simd_vec<int, 8> &v1, const simd_vec<int, 8> &v2) {
        simd_vec<int, 8> ret;
#if defined(USE_AVX2) || defined(USE_AVX512)
        ret.vec_ = _mm256_srlv_epi32(v1.vec_, v2.vec_);
#else
        ITERATE_8({ ret.comp_[i] = reinterpret_cast<const unsigned &>(v1.comp_[i]) >> v2.comp_[i]; })
#endif
        return ret;
    }

    friend force_inline simd_vec<int, 8> operator>>(const simd_vec<int, 8> &v1, int v2) {
        simd_vec<int, 8> ret;
#if defined(USE_AVX2) || defined(USE_AVX512)
        ret.vec_ = _mm256_srli_epi32(v1.vec_, v2);
#else
        ITERATE_8({ ret.comp_[i] = reinterpret_cast<const unsigned &>(v1.comp_[i]) >> v2; })
#endif
        return ret;
    }

    friend force_inline simd_vec<int, 8> operator<<(const simd_vec<int, 8> &v1, const simd_vec<int, 8> &v2) {
        simd_vec<int, 8> ret;
#if defined(USE_AVX2) || defined(USE_AVX512)
        ret.vec_ = _mm256_sllv_epi32(v1.vec_, v2.vec_);
#else
        ITERATE_8({ ret.comp_[i] = v1.comp_[i] << v2.comp_[i]; })
#endif
        return ret;
    }

    friend force_inline simd_vec<int, 8> operator<<(const simd_vec<int, 8> &v1, int v2) {
        simd_vec<int, 8> ret;
#if defined(USE_AVX2) || defined(USE_AVX512)
        ret.vec_ = _mm256_slli_epi32(v1.vec_, v2);
#else
        ITERATE_8({ ret.comp_[i] = v1.comp_[i] << v2; })
#endif
        return ret;
    }

    force_inline simd_vec<int, 8> operator~() const {
        simd_vec<int, 8> ret;
        ret.vec_ = _mm256_andnot_si256(vec_, _mm256_set1_epi32(~0));
        return ret;
    }

    friend force_inline bool is_equal(const simd_vec<int, 8> &v1, const simd_vec<int, 8> &v2) {
#if defined(USE_AVX2) || defined(USE_AVX512)
        __m256i vcmp = _mm256_cmpeq_epi32(v1.vec_, v2.vec_);
        return (_mm256_movemask_epi8(vcmp) == 0xffffffff);
#else
        __m256i vcmp = _mm256_cmpeq_epi16(v1.vec_, v2.vec_);
        return (_mm256_movemask_ps(reinterpret_cast<const __m256 &>(vcmp)) == 0xffffffff);
#endif
    }

#if defined(USE_AVX2) || defined(USE_AVX512)
    template <int Scale>
    friend force_inline simd_vec<float, 8> gather(const float *base_addr, const simd_vec<int, 8> &vindex);
    template <int Scale>
    friend force_inline simd_vec<int, 8> gather(const int *base_addr, const simd_vec<int, 8> &vindex);
#endif

    static int size() { return 8; }
    static bool is_native() {
#if defined(USE_AVX2) || defined(USE_AVX512)
        return true;
#else
        // mostly not native, so return false here
        return false;
#endif
    }
};

force_inline simd_vec<float, 8> simd_vec<float, 8>::operator~() const {
    simd_vec<float, 8> ret;
    ret.vec_ = _mm256_castsi256_ps(_mm256_andnot_si256(_mm256_castps_si256(vec_), _mm256_set1_epi32(~0)));
    return ret;
}

force_inline simd_vec<float, 8> simd_vec<float, 8>::operator-() const {
    simd_vec<float, 8> temp;
    __m256 m = _mm256_set1_ps(-0.0f);
    temp.vec_ = _mm256_xor_ps(vec_, m);
    return temp;
}

force_inline simd_vec<float, 8>::operator simd_vec<int, 8>() const {
    simd_vec<int, 8> ret;
    ret.vec_ = _mm256_cvtps_epi32(vec_);
    return ret;
}

force_inline simd_vec<float, 8> simd_vec<float, 8>::sqrt() const {
    simd_vec<float, 8> temp;
    temp.vec_ = _mm256_sqrt_ps(vec_);
    return temp;
}

force_inline simd_vec<float, 8> simd_vec<float, 8>::min(const simd_vec<float, 8> &v1, const simd_vec<float, 8> &v2) {
    simd_vec<float, 8> temp;
    temp.vec_ = _mm256_min_ps(v1.vec_, v2.vec_);
    return temp;
}

force_inline simd_vec<float, 8> simd_vec<float, 8>::min(const simd_vec<float, 8> &v1, const float v2) {
    simd_vec<float, 8> temp;
    temp.vec_ = _mm256_min_ps(v1.vec_, _mm256_set1_ps(v2));
    return temp;
}

force_inline simd_vec<float, 8> simd_vec<float, 8>::max(const simd_vec<float, 8> &v1, const simd_vec<float, 8> &v2) {
    simd_vec<float, 8> temp;
    temp.vec_ = _mm256_max_ps(v1.vec_, v2.vec_);
    return temp;
}

force_inline simd_vec<float, 8> simd_vec<float, 8>::max(const simd_vec<float, 8> &v1, const float v2) {
    simd_vec<float, 8> temp;
    temp.vec_ = _mm256_max_ps(v1.vec_, _mm256_set1_ps(v2));
    return temp;
}

force_inline static simd_vec<float, 8> and_not(const simd_vec<float, 8> &v1, const simd_vec<float, 8> &v2) {
    simd_vec<float, 8> temp;
    temp.vec_ = _mm256_andnot_ps(v1.vec_, v2.vec_);
    return temp;
}

force_inline static simd_vec<float, 8> floor(const simd_vec<float, 8> &v1) {
    simd_vec<float, 8> temp;
    temp.vec_ = _mm256_floor_ps(v1.vec_);
    return temp;
}

force_inline static simd_vec<float, 8> ceil(const simd_vec<float, 8> &v1) {
    simd_vec<float, 8> temp;
    temp.vec_ = _mm256_ceil_ps(v1.vec_);
    return temp;
}

force_inline simd_vec<float, 8> operator&(const simd_vec<float, 8> &v1, const simd_vec<float, 8> &v2) {
    simd_vec<float, 8> temp;
    temp.vec_ = _mm256_and_ps(v1.vec_, v2.vec_);
    return temp;
}

force_inline simd_vec<float, 8> operator|(const simd_vec<float, 8> &v1, const simd_vec<float, 8> &v2) {
    simd_vec<float, 8> temp;
    temp.vec_ = _mm256_or_ps(v1.vec_, v2.vec_);
    return temp;
}

force_inline simd_vec<float, 8> operator^(const simd_vec<float, 8> &v1, const simd_vec<float, 8> &v2) {
    simd_vec<float, 8> temp;
    temp.vec_ = _mm256_xor_ps(v1.vec_, v2.vec_);
    return temp;
}

force_inline simd_vec<float, 8> operator+(const simd_vec<float, 8> &v1, const simd_vec<float, 8> &v2) {
    simd_vec<float, 8> temp;
    temp.vec_ = _mm256_add_ps(v1.vec_, v2.vec_);
    return temp;
}

force_inline simd_vec<float, 8> operator-(const simd_vec<float, 8> &v1, const simd_vec<float, 8> &v2) {
    simd_vec<float, 8> temp;
    temp.vec_ = _mm256_sub_ps(v1.vec_, v2.vec_);
    return temp;
}

force_inline simd_vec<float, 8> operator*(const simd_vec<float, 8> &v1, const simd_vec<float, 8> &v2) {
    simd_vec<float, 8> temp;
    temp.vec_ = _mm256_mul_ps(v1.vec_, v2.vec_);
    return temp;
}

force_inline simd_vec<float, 8> operator/(const simd_vec<float, 8> &v1, const simd_vec<float, 8> &v2) {
    simd_vec<float, 8> temp;
    temp.vec_ = _mm256_div_ps(v1.vec_, v2.vec_);
    return temp;
}

force_inline simd_vec<float, 8> operator+(const simd_vec<float, 8> &v1, float v2) {
    simd_vec<float, 8> temp;
    temp.vec_ = _mm256_add_ps(v1.vec_, _mm256_set1_ps(v2));
    return temp;
}

force_inline simd_vec<float, 8> operator-(const simd_vec<float, 8> &v1, float v2) {
    simd_vec<float, 8> temp;
    temp.vec_ = _mm256_sub_ps(v1.vec_, _mm256_set1_ps(v2));
    return temp;
}

force_inline simd_vec<float, 8> operator*(const simd_vec<float, 8> &v1, float v2) {
    simd_vec<float, 8> temp;
    temp.vec_ = _mm256_mul_ps(v1.vec_, _mm256_set1_ps(v2));
    return temp;
}

force_inline simd_vec<float, 8> operator/(const simd_vec<float, 8> &v1, float v2) {
    simd_vec<float, 8> temp;
    temp.vec_ = _mm256_div_ps(v1.vec_, _mm256_set1_ps(v2));
    return temp;
}

force_inline simd_vec<float, 8> operator+(float v1, const simd_vec<float, 8> &v2) {
    simd_vec<float, 8> temp;
    temp.vec_ = _mm256_add_ps(_mm256_set1_ps(v1), v2.vec_);
    return temp;
}

force_inline simd_vec<float, 8> operator-(float v1, const simd_vec<float, 8> &v2) {
    simd_vec<float, 8> temp;
    temp.vec_ = _mm256_sub_ps(_mm256_set1_ps(v1), v2.vec_);
    return temp;
}

force_inline simd_vec<float, 8> operator*(float v1, const simd_vec<float, 8> &v2) {
    simd_vec<float, 8> temp;
    temp.vec_ = _mm256_mul_ps(_mm256_set1_ps(v1), v2.vec_);
    return temp;
}

force_inline simd_vec<float, 8> operator/(float v1, const simd_vec<float, 8> &v2) {
    simd_vec<float, 8> temp;
    temp.vec_ = _mm256_div_ps(_mm256_set1_ps(v1), v2.vec_);
    return temp;
}

force_inline simd_vec<float, 8> operator<(const simd_vec<float, 8> &v1, const simd_vec<float, 8> &v2) {
    simd_vec<float, 8> ret;
    ret.vec_ = _mm256_cmp_ps(v1.vec_, v2.vec_, _CMP_LT_OS);
    return ret;
}

force_inline simd_vec<float, 8> operator<=(const simd_vec<float, 8> &v1, const simd_vec<float, 8> &v2) {
    simd_vec<float, 8> ret;
    ret.vec_ = _mm256_cmp_ps(v1.vec_, v2.vec_, _CMP_LE_OS);
    return ret;
}

force_inline simd_vec<float, 8> operator>(const simd_vec<float, 8> &v1, const simd_vec<float, 8> &v2) {
    simd_vec<float, 8> ret;
    ret.vec_ = _mm256_cmp_ps(v1.vec_, v2.vec_, _CMP_GT_OS);
    return ret;
}

force_inline simd_vec<float, 8> operator>=(const simd_vec<float, 8> &v1, const simd_vec<float, 8> &v2) {
    simd_vec<float, 8> ret;
    ret.vec_ = _mm256_cmp_ps(v1.vec_, v2.vec_, _CMP_GE_OS);
    return ret;
}

force_inline simd_vec<float, 8> operator==(const simd_vec<float, 8> &v1, const simd_vec<float, 8> &v2) {
    simd_vec<float, 8> ret;
    ret.vec_ = _mm256_cmp_ps(v1.vec_, v2.vec_, _CMP_EQ_OS);
    return ret;
}

force_inline simd_vec<float, 8> operator!=(const simd_vec<float, 8> &v1, const simd_vec<float, 8> &v2) {
    simd_vec<float, 8> ret;
    ret.vec_ = _mm256_cmp_ps(v1.vec_, v2.vec_, _CMP_NEQ_OS);
    return ret;
}

force_inline simd_vec<float, 8> operator<(const simd_vec<float, 8> &v1, float v2) {
    simd_vec<float, 8> ret;
    ret.vec_ = _mm256_cmp_ps(v1.vec_, _mm256_set1_ps(v2), _CMP_LT_OS);
    return ret;
}

force_inline simd_vec<float, 8> operator<=(const simd_vec<float, 8> &v1, float v2) {
    simd_vec<float, 8> ret;
    ret.vec_ = _mm256_cmp_ps(v1.vec_, _mm256_set1_ps(v2), _CMP_LE_OS);
    return ret;
}

force_inline simd_vec<float, 8> operator>(const simd_vec<float, 8> &v1, float v2) {
    simd_vec<float, 8> ret;
    ret.vec_ = _mm256_cmp_ps(v1.vec_, _mm256_set1_ps(v2), _CMP_GT_OS);
    return ret;
}

force_inline simd_vec<float, 8> operator>=(const simd_vec<float, 8> &v1, float v2) {
    simd_vec<float, 8> ret;
    ret.vec_ = _mm256_cmp_ps(v1.vec_, _mm256_set1_ps(v2), _CMP_GE_OS);
    return ret;
}

force_inline simd_vec<float, 8> operator==(const simd_vec<float, 8>& v1, float v2) {
    simd_vec<float, 8> ret;
    ret.vec_ = _mm256_cmp_ps(v1.vec_, _mm256_set1_ps(v2), _CMP_EQ_OS);
    return ret;
}

force_inline simd_vec<float, 8> operator!=(const simd_vec<float, 8> &v1, float v2) {
    simd_vec<float, 8> ret;
    ret.vec_ = _mm256_cmp_ps(v1.vec_, _mm256_set1_ps(v2), _CMP_NEQ_OS);
    return ret;
}

force_inline simd_vec<float, 8> clamp(const simd_vec<float, 8> &v1, float min, float max) {
    simd_vec<float, 8> ret;
    ret.vec_ = _mm256_max_ps(_mm256_set1_ps(min), _mm256_min_ps(v1.vec_, _mm256_set1_ps(max)));
    return ret;
}

force_inline simd_vec<float, 8> pow(const simd_vec<float, 8> &v1, const simd_vec<float, 8> &v2) {
    simd_vec<float, 8> ret;
    ITERATE_8({ ret.comp_[i] = std::pow(v1.comp_[i], v2.comp_[i]); })
    return ret;
}

force_inline simd_vec<float, 8> normalize(const simd_vec<float, 8> &v1) { return v1 / v1.length(); }

#if defined(USE_AVX2) || defined(USE_AVX512)
force_inline simd_vec<float, 8> fmadd(const simd_vec<float, 8> &a, const simd_vec<float, 8> &b,
                                      const simd_vec<float, 8> &c) {
    simd_vec<float, 8> ret;
    ret.vec_ = _mm256_fmadd_ps(a.vec_, b.vec_, c.vec_);
    return ret;
}

force_inline simd_vec<float, 8> fmadd(const simd_vec<float, 8> &a, const float b, const simd_vec<float, 8> &c) {
    simd_vec<float, 8> ret;
    ret.vec_ = _mm256_fmadd_ps(a.vec_, _mm256_set1_ps(b), c.vec_);
    return ret;
}

force_inline simd_vec<float, 8> fmadd(const float a, const simd_vec<float, 8> &b, const float c) {
    simd_vec<float, 8> ret;
    ret.vec_ = _mm256_fmadd_ps(_mm256_set1_ps(a), b.vec_, _mm256_set1_ps(c));
    return ret;
}

force_inline simd_vec<float, 8> fmsub(const simd_vec<float, 8> &a, const simd_vec<float, 8> &b,
                                      const simd_vec<float, 8> &c) {
    simd_vec<float, 8> ret;
    ret.vec_ = _mm256_fmsub_ps(a.vec_, b.vec_, c.vec_);
    return ret;
}

force_inline simd_vec<float, 8> fmsub(const simd_vec<float, 8> &a, const float b, const simd_vec<float, 8> &c) {
    simd_vec<float, 8> ret;
    ret.vec_ = _mm256_fmsub_ps(a.vec_, _mm256_set1_ps(b), c.vec_);
    return ret;
}

force_inline simd_vec<float, 8> fmsub(const float a, const simd_vec<float, 8> &b, const float c) {
    simd_vec<float, 8> ret;
    ret.vec_ = _mm256_fmsub_ps(_mm256_set1_ps(a), b.vec_, _mm256_set1_ps(c));
    return ret;
}

template <int Scale>
force_inline simd_vec<float, 8> gather(const float *base_addr, const simd_vec<int, 8> &vindex) {
    simd_vec<float, 8> ret;
    ret.vec_ = _mm256_i32gather_ps(base_addr, vindex.vec_, Scale * sizeof(float));
    return ret;
}

template <int Scale> force_inline simd_vec<int, 8> gather(const int *base_addr, const simd_vec<int, 8> &vindex) {
    simd_vec<int, 8> ret;
    ret.vec_ = _mm256_i32gather_epi32(base_addr, vindex.vec_, Scale * sizeof(int));
    return ret;
}

#endif

} // namespace NS
} // namespace Ray

#pragma warning(pop)

#ifdef __GNUC__
#pragma GCC pop_options
#pragma clang attribute pop
#endif
