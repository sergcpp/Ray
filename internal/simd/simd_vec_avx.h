//#pragma once

#include "simd_vec_sse.h"

#include <immintrin.h>

#ifdef __GNUC__
#pragma GCC push_options
#pragma GCC target ("avx")
#endif

#if defined(__GNUC__)
#define _mm256_test_all_zeros(mask, val) \
              _mm256_testz_si256((mask), (val))
#endif

#pragma warning(push)
#pragma warning(disable : 4752)

namespace Ray {
namespace NS {

template <int S>
class simd_vec<typename std::enable_if<S == 8, float>::type, S> {
    public:
    union {
        __m256 vec_;
        float comp_[8];
    };

    friend class simd_vec<int, S>;
public:
    force_inline simd_vec() = default;
    force_inline simd_vec(float f) {
        vec_ = _mm256_set1_ps(f);
    }
    force_inline simd_vec(float f1, float f2, float f3, float f4, float f5, float f6, float f7, float f8) {
        vec_ = _mm256_setr_ps(f1, f2, f3, f4, f5, f6, f7, f8);
    }
    force_inline simd_vec(const float *f) {
        vec_ = _mm256_loadu_ps(f);
    }
    force_inline simd_vec(const float *f, simd_mem_aligned_tag) {
        vec_ = _mm256_load_ps(f);
    }

    force_inline float &operator[](int i) { return comp_[i]; }
    force_inline float operator[](int i) const { return comp_[i]; }

    force_inline simd_vec<float, S> &operator+=(const simd_vec<float, S> &rhs) {
        vec_ = _mm256_add_ps(vec_, rhs.vec_);
        return *this;
    }

    force_inline simd_vec<float, S> &operator+=(float rhs) {
        __m256 _rhs = _mm256_set1_ps(rhs);
        vec_ = _mm256_add_ps(vec_, _rhs);
        return *this;
    }

    force_inline simd_vec<float, S> &operator-=(const simd_vec<float, S> &rhs) {
        vec_ = _mm256_sub_ps(vec_, rhs.vec_);
        return *this;
    }

    force_inline simd_vec<float, S> &operator-=(float rhs) {
        vec_ = _mm256_sub_ps(vec_, _mm256_set1_ps(rhs));
        return *this;
    }

    force_inline simd_vec<float, S> &operator*=(const simd_vec<float, S> &rhs) {
        vec_ = _mm256_mul_ps(vec_, rhs.vec_);
        return *this;
    }

    force_inline simd_vec<float, S> &operator*=(float rhs) {
        vec_ = _mm256_mul_ps(vec_, _mm256_set1_ps(rhs));
        return *this;
    }

    force_inline simd_vec<float, S> &operator/=(const simd_vec<float, S> &rhs) {
        vec_ = _mm256_div_ps(vec_, rhs.vec_);
        return *this;
    }

    force_inline simd_vec<float, S> &operator/=(float rhs) {
        __m256 _rhs = _mm256_set1_ps(rhs);
        vec_ = _mm256_div_ps(vec_, _rhs);
        return *this;
    }

    force_inline simd_vec<float, S> operator-() const {
        simd_vec<float, S> temp;
        __m256 m = _mm256_set1_ps(-0.0f);
        temp.vec_ = _mm256_xor_ps(vec_, m);
        return temp;
    }

    force_inline operator simd_vec<int, S>() const {
        simd_vec<int, S> ret;
        ret.vec_ = _mm256_cvtps_epi32(vec_);
        return ret;
    }

    force_inline simd_vec<float, S> sqrt() const {
        simd_vec<float, S> temp;
        temp.vec_ = _mm256_sqrt_ps(vec_);
        return temp;
    }

    force_inline void copy_to(float *f) const {
        _mm256_storeu_ps(f, vec_);
    }

    force_inline void copy_to(float *f, simd_mem_aligned_tag) const {
        _mm256_store_ps(f, vec_);
    }

    force_inline void blend_to(const simd_vec<float, S> &mask, const simd_vec<float, S> &v1) {
        vec_ = _mm256_blendv_ps(vec_, v1.vec_, mask.vec_);
    }

    force_inline static simd_vec<float, S> min(const simd_vec<float, S> &v1, const simd_vec<float, S> &v2) {
        simd_vec<float, S> temp;
        temp.vec_ = _mm256_min_ps(v1.vec_, v2.vec_);
        return temp;
    }

    force_inline static simd_vec<float, S> max(const simd_vec<float, S> &v1, const simd_vec<float, S> &v2) {
        simd_vec<float, S> temp;
        temp.vec_ = _mm256_max_ps(v1.vec_, v2.vec_);
        return temp;
    }

    force_inline static simd_vec<float, S> and_not(const simd_vec<float, S> &v1, const simd_vec<float, S> &v2) {
        simd_vec<float, S> temp;
        temp.vec_ = _mm256_andnot_ps(v1.vec_, v2.vec_);
        return temp;
    }

    force_inline static simd_vec<float, S> floor(const simd_vec<float, S> &v1) {
        simd_vec<float, S> temp;
        temp.vec_ = _mm256_floor_ps(v1.vec_);
        return temp;
    }

    force_inline static simd_vec<float, S> ceil(const simd_vec<float, S> &v1) {
        simd_vec<float, S> temp;
        temp.vec_ = _mm256_ceil_ps(v1.vec_);
        return temp;
    }

    friend force_inline simd_vec<float, S> operator&(const simd_vec<float, S> &v1, const simd_vec<float, S> &v2) {
        simd_vec<float, S> temp;
        temp.vec_ = _mm256_and_ps(v1.vec_, v2.vec_);
        return temp;
    }

    friend force_inline simd_vec<float, S> operator|(const simd_vec<float, S> &v1, const simd_vec<float, S> &v2) {
        simd_vec<float, S> temp;
        temp.vec_ = _mm256_or_ps(v1.vec_, v2.vec_);
        return temp;
    }

    friend force_inline simd_vec<float, S> operator^(const simd_vec<float, S> &v1, const simd_vec<float, S> &v2) {
        simd_vec<float, S> temp;
        temp.vec_ = _mm256_xor_ps(v1.vec_, v2.vec_);
        return temp;
    }

    friend force_inline simd_vec<float, S> operator+(const simd_vec<float, S> &v1, const simd_vec<float, S> &v2) {
        simd_vec<float, S> temp;
        temp.vec_ = _mm256_add_ps(v1.vec_, v2.vec_);
        return temp;
    }

    friend force_inline simd_vec<float, S> operator-(const simd_vec<float, S> &v1, const simd_vec<float, S> &v2) {
        simd_vec<float, S> temp;
        temp.vec_ = _mm256_sub_ps(v1.vec_, v2.vec_);
        return temp;
    }

    friend force_inline simd_vec<float, S> operator*(const simd_vec<float, S> &v1, const simd_vec<float, S> &v2) {
        simd_vec<float, S> temp;
        temp.vec_ = _mm256_mul_ps(v1.vec_, v2.vec_);
        return temp;
    }

    friend force_inline simd_vec<float, S> operator/(const simd_vec<float, S> &v1, const simd_vec<float, S> &v2) {
        simd_vec<float, S> temp;
        temp.vec_ = _mm256_div_ps(v1.vec_, v2.vec_);
        return temp;
    }

    friend force_inline simd_vec<float, S> operator+(const simd_vec<float, S> &v1, float v2) {
        simd_vec<float, S> temp;
        temp.vec_ = _mm256_add_ps(v1.vec_, _mm256_set1_ps(v2));
        return temp;
    }

    friend force_inline simd_vec<float, S> operator-(const simd_vec<float, S> &v1, float v2) {
        simd_vec<float, S> temp;
        temp.vec_ = _mm256_sub_ps(v1.vec_, _mm256_set1_ps(v2));
        return temp;
    }

    friend force_inline simd_vec<float, S> operator*(const simd_vec<float, S> &v1, float v2) {
        simd_vec<float, S> temp;
        temp.vec_ = _mm256_mul_ps(v1.vec_, _mm256_set1_ps(v2));
        return temp;
    }

    friend force_inline simd_vec<float, S> operator/(const simd_vec<float, S> &v1, float v2) {
        simd_vec<float, S> temp;
        temp.vec_ = _mm256_div_ps(v1.vec_, _mm256_set1_ps(v2));
        return temp;
    }

    friend force_inline simd_vec<float, S> operator+(float v1, const simd_vec<float, S> &v2) {
        simd_vec<float, S> temp;
        temp.vec_ = _mm256_add_ps(_mm256_set1_ps(v1), v2.vec_);
        return temp;
    }

    friend force_inline simd_vec<float, S> operator-(float v1, const simd_vec<float, S> &v2) {
        simd_vec<float, S> temp;
        temp.vec_ = _mm256_sub_ps(_mm256_set1_ps(v1), v2.vec_);
        return temp;
    }

    friend force_inline simd_vec<float, S> operator*(float v1, const simd_vec<float, S> &v2) {
        simd_vec<float, S> temp;
        temp.vec_ = _mm256_mul_ps(_mm256_set1_ps(v1), v2.vec_);
        return temp;
    }

    friend force_inline simd_vec<float, S> operator/(float v1, const simd_vec<float, S> &v2) {
        simd_vec<float, S> temp;
        temp.vec_ = _mm256_div_ps(_mm256_set1_ps(v1), v2.vec_);
        return temp;
    }

    friend force_inline simd_vec<float, S> operator<(const simd_vec<float, S> &v1, const simd_vec<float, S> &v2) {
        simd_vec<float, S> ret;
        ret.vec_ = _mm256_cmp_ps(v1.vec_, v2.vec_, _CMP_LT_OS);
        return ret;
    }

    friend force_inline simd_vec<float, S> operator<=(const simd_vec<float, S> &v1, const simd_vec<float, S> &v2) {
        simd_vec<float, S> ret;
        ret.vec_ = _mm256_cmp_ps(v1.vec_, v2.vec_, _CMP_LE_OS);
        return ret;
    }

    friend force_inline simd_vec<float, S> operator>(const simd_vec<float, S> &v1, const simd_vec<float, S> &v2) {
        simd_vec<float, S> ret;
        ret.vec_ = _mm256_cmp_ps(v1.vec_, v2.vec_, _CMP_GT_OS);
        return ret;
    }

    friend force_inline simd_vec<float, S> operator>=(const simd_vec<float, S> &v1, const simd_vec<float, S> &v2) {
        simd_vec<float, S> ret;
        ret.vec_ = _mm256_cmp_ps(v1.vec_, v2.vec_, _CMP_GE_OS);
        return ret;
    }

    friend force_inline simd_vec<float, S> operator<(const simd_vec<float, S> &v1, float v2) {
        simd_vec<float, S> ret;
        ret.vec_ = _mm256_cmp_ps(v1.vec_, _mm256_set1_ps(v2), _CMP_LT_OS);
        return ret;
    }

    friend force_inline simd_vec<float, S> operator<=(const simd_vec<float, S> &v1, float v2) {
        simd_vec<float, S> ret;
        ret.vec_ = _mm256_cmp_ps(v1.vec_, _mm256_set1_ps(v2), _CMP_LE_OS);
        return ret;
    }

    friend force_inline simd_vec<float, S> operator>(const simd_vec<float, S> &v1, float v2) {
        simd_vec<float, S> ret;
        ret.vec_ = _mm256_cmp_ps(v1.vec_, _mm256_set1_ps(v2), _CMP_GT_OS);
        return ret;
    }

    friend force_inline simd_vec<float, S> operator>=(const simd_vec<float, S> &v1, float v2) {
        simd_vec<float, S> ret;
        ret.vec_ = _mm256_cmp_ps(v1.vec_, _mm256_set1_ps(v2), _CMP_GE_OS);
        return ret;
    }

    friend force_inline simd_vec<float, S> clamp(const simd_vec<float, S> &v1, float min, float max) {
        simd_vec<float, S> ret;
        ret.vec_ = _mm256_max_ps(_mm256_set1_ps(min), _mm256_min_ps(v1.vec_, _mm256_set1_ps(max)));
        return ret;
    }

    friend force_inline simd_vec<float, S> pow(const simd_vec<float, S> &v1, const simd_vec<float, S> &v2) {
        simd_vec<float, S> ret;
        ITERATE_8({ ret.comp_[i] = std::pow(v1.comp_[i], v2.comp_[i]); })
        return ret;
    }

    friend force_inline simd_vec<float, S> normalize(const simd_vec<float, S> &v1) {
        return v1 / v1.length();
    }

    friend force_inline const float *value_ptr(const simd_vec<float, S> &v1) {
        return &v1.comp_[0];
    }

    static int size() { return S; }
    static bool is_native() { return true; }
};

template <int S>
class simd_vec<typename std::enable_if<S == 8, int>::type, S> {
    union {
        __m256i vec_;
        int comp_[8];
    };

    friend class simd_vec<float, S>;
public:
    force_inline simd_vec() = default;
    force_inline simd_vec(int f) {
        vec_ = _mm256_set1_epi32(f);
    }
    force_inline simd_vec(int i1, int i2, int i3, int i4, int i5, int i6, int i7, int i8) {
        vec_ = _mm256_setr_epi32(i1, i2, i3, i4, i5, i6, i7, i8);
    }
    force_inline simd_vec(const int *f) {
        vec_ = _mm256_loadu_si256((const __m256i *)f);
    }
    force_inline simd_vec(const int *f, simd_mem_aligned_tag) {
        vec_ = _mm256_load_si256((const __m256i *)f);
    }

    force_inline int &operator[](int i) { return comp_[i]; }
    force_inline int operator[](int i) const { return comp_[i]; }

    force_inline simd_vec<int, S> &operator+=(const simd_vec<int, S> &rhs) {
#if 0 // requires AVX2 support
        vec_ = _mm256_add_epi32(vec_, rhs.vec_);
#else
        ITERATE_8({ comp_[i] = comp_[i] + rhs.comp_[i]; })
#endif
        return *this;
    }

    force_inline simd_vec<int, S> &operator+=(int rhs) {
#if 0 // requires AVX2 support
        vec_ = _mm256_add_epi32(vec_, _mm256_set1_epi32(rhs));
#else
        ITERATE_8({ comp_[i] = comp_[i] + rhs; })
#endif
        return *this;
    }

    force_inline simd_vec<int, S> &operator-=(const simd_vec<int, S> &rhs) {
#if 0 // requires AVX2 support
        vec_ = _mm256_sub_epi32(vec_, rhs.vec_);
#else
        ITERATE_8({ comp_[i] = comp_[i] - rhs.comp_[i]; })
#endif
        return *this;
    }

    force_inline simd_vec<int, S> &operator-=(int rhs) {
#if 0 // requires AVX2 support
        vec_ = _mm256_sub_epi32(vec_, _mm256_set1_epi32(rhs));
#else
        ITERATE_8({ comp_[i] = comp_[i] - rhs; })
#endif
        return *this;
    }

    force_inline simd_vec<int, S> &operator*=(const simd_vec<int, S> &rhs) {
        ITERATE_8({ comp_[i] = comp_[i] * rhs.comp_[i]; })
        return *this;
    }

    force_inline simd_vec<int, S> &operator*=(int rhs) {
        ITERATE_8({ comp_[i] = comp_[i] * rhs; })
        return *this;
    }

    force_inline simd_vec<int, S> &operator/=(const simd_vec<int, S> &rhs) {
        ITERATE_8({ comp_[i] = comp_[i] / rhs.comp_[i]; })
        return *this;
    }

    force_inline simd_vec<int, S> &operator/=(int rhs) {
        ITERATE_8({ comp_[i] = comp_[i] / rhs; })
        return *this;
    }

    force_inline simd_vec<int, S> operator==(int rhs) const {
        simd_vec<int, S> ret;
#if 0 // requires AVX2 support
        ret.vec_ = _mm256_cmpeq_epi32(vec_, _mm256_set1_epi32(rhs));
#else
        ITERATE_8({ ret.comp_[i] = comp_[i] == rhs ? 0xFFFFFFFF : 0; })
#endif
        return ret;
    }

    force_inline simd_vec<int, S> operator==(const simd_vec<int, S> &rhs) const {
        simd_vec<int, S> ret;
#if 0 // requires AVX2 support
        ret.vec_ = _mm256_cmpeq_epi32(vec_, rhs.vec_);
#else
        ITERATE_8({ ret.comp_[i] = comp_[i] == rhs.comp_[i] ? 0xFFFFFFFF : 0; })
#endif
        return ret;
    }

    force_inline simd_vec<int, S> operator!=(int rhs) const {
        simd_vec<int, S> ret;
#if 0 // requires AVX2 support
        ret.vec_ = _mm256_andnot_si256(_mm256_cmpeq_epi32(vec_, _mm256_set1_epi32(rhs)), _mm256_set1_epi32(~0));
#else
        ITERATE_8({ ret.comp_[i] = comp_[i] != rhs ? 0xFFFFFFFF : 0; })
#endif
        return ret;
    }

    force_inline simd_vec<int, S> operator!=(const simd_vec<int, S> &rhs) const {
        simd_vec<int, S> ret;
#if 0 // requires AVX2 support
        ret.vec_ = _mm256_andnot_si256(_mm256_cmpeq_epi32(vec_, rhs.vec_), _mm256_set1_epi32(~0));
#else
        ITERATE_8({ ret.comp_[i] = comp_[i] != rhs.comp_[i] ? 0xFFFFFFFF : 0; })
#endif
        return ret;
    }

    force_inline operator simd_vec<float, S>() const {
        simd_vec<float, S> ret;
        ret.vec_ = _mm256_cvtepi32_ps(vec_);
        return ret;
    }

    force_inline void copy_to(int *f) const {
        _mm256_storeu_si256((__m256i *)f, vec_);
    }

    force_inline void copy_to(int *f, simd_mem_aligned_tag) const {
        _mm256_store_si256((__m256i *)f, vec_);
    }

    force_inline void blend_to(const simd_vec<int, S> &mask, const simd_vec<int, S> &v1) {
        vec_ = _mm256_castps_si256(_mm256_blendv_ps(_mm256_castsi256_ps(vec_), _mm256_castsi256_ps(v1.vec_), _mm256_castsi256_ps(mask.vec_)));
    }

    force_inline bool all_zeros() const {
        return _mm256_test_all_zeros(vec_, vec_) != 0;
    }

    force_inline bool all_zeros(const simd_vec<int, S> &mask) const {
        return _mm256_test_all_zeros(vec_, mask.vec_) != 0;
    }

    force_inline bool not_all_zeros() const {
        volatile int res = _mm256_test_all_zeros(vec_, vec_);
        return res == 0;
    }

    force_inline static simd_vec<int, S> min(const simd_vec<int, S> &v1, const simd_vec<int, S> &v2) {
        simd_vec<int, S> temp;
#if 0 // requires AVX2 support
        temp.vec_ = _mm256_min_epi32(v1.vec_, v2.vec_);
#else
        ITERATE_8({ temp.comp_[i] = std::min(v1.comp_[i], v2.comp_[i]); })
#endif
        return temp;
    }

    force_inline static simd_vec<int, S> max(const simd_vec<int, S> &v1, const simd_vec<int, S> &v2) {
        simd_vec<int, S> temp;
#if 0 // requires AVX2 support
        temp.vec_ = _mm256_max_epi32(v1.vec_, v2.vec_);
#else
        ITERATE_8({ temp.comp_[i] = std::max(v1.comp_[i], v2.comp_[i]); })
#endif
        return temp;
    }

    force_inline static simd_vec<int, S> and_not(const simd_vec<int, S> &v1, const simd_vec<int, S> &v2) {
        simd_vec<int, S> temp;
        temp.vec_ = _mm256_andnot_si256(v1.vec_, v2.vec_);
        return temp;
    }

    friend force_inline simd_vec<int, S> operator&(const simd_vec<int, S> &v1, const simd_vec<int, S> &v2) {
        simd_vec<int, S> temp;
        temp.vec_ = _mm256_and_si256(v1.vec_, v2.vec_);
        return temp;
    }

    friend force_inline simd_vec<int, S> operator|(const simd_vec<int, S> &v1, const simd_vec<int, S> &v2) {
        simd_vec<int, S> temp;
        temp.vec_ = _mm256_or_si256(v1.vec_, v2.vec_);
        return temp;
    }

    friend force_inline simd_vec<int, S> operator^(const simd_vec<int, S> &v1, const simd_vec<int, S> &v2) {
        simd_vec<int, S> temp;
        temp.vec_ = _mm256_xor_si256(v1.vec_, v2.vec_);
        return temp;
    }

    friend force_inline simd_vec<int, S> operator+(const simd_vec<int, S> &v1, const simd_vec<int, S> &v2) {
        simd_vec<int, S> temp;
#if 0 // requires AVX2 support
        temp.vec_ = _mm256_add_epi32(v1.vec_, v2.vec_);
#else
        ITERATE_8({ temp.comp_[i] = v1.comp_[i] + v2.comp_[i]; })
#endif
        return temp;
    }

    friend force_inline simd_vec<int, S> operator-(const simd_vec<int, S> &v1, const simd_vec<int, S> &v2) {
        simd_vec<int, S> temp;
#if 0 // requires AVX2 support
        temp.vec_ = _mm256_sub_epi32(v1.vec_, v2.vec_);
#else
        ITERATE_8({ temp.comp_[i] = v1.comp_[i] - v2.comp_[i]; })
#endif
        return temp;
    }

    friend force_inline simd_vec<int, S> operator*(const simd_vec<int, S> &v1, const simd_vec<int, S> &v2) {
        simd_vec<int, S> temp;
        ITERATE_8({ temp.comp_[i] = v1.comp_[i] * v2.comp_[i]; })
        return temp;
    }

    friend force_inline simd_vec<int, S> operator/(const simd_vec<int, S> &v1, const simd_vec<int, S> &v2) {
        simd_vec<int, S> temp;
        ITERATE_8({ temp.comp_[i] = v1.comp_[i] / v2.comp_[i]; })
        return temp;
    }

    friend force_inline simd_vec<int, S> operator+(const simd_vec<int, S> &v1, int v2) {
        simd_vec<int, S> temp;
#if 0 // requires AVX2 support
        temp.vec_ = _mm256_add_epi32(v1.vec_, _mm256_set1_epi32(v2));
#else
        ITERATE_8({ temp.comp_[i] = v1.comp_[i] + v2; })
#endif
        return temp;
    }

    friend force_inline simd_vec<int, S> operator-(const simd_vec<int, S> &v1, int v2) {
        simd_vec<int, S> temp;
#if 0 // requires AVX2 support
        temp.vec_ = _mm256_sub_epi32(v1.vec_, _mm256_set1_epi32(v2));
#else
        ITERATE_8({ temp.comp_[i] = v1.comp_[i] - v2; })
#endif
        return temp;
    }

    friend force_inline simd_vec<int, S> operator*(const simd_vec<int, S> &v1, int v2) {
        simd_vec<int, S> temp;
        ITERATE_8({ temp.comp_[i] = v1.comp_[i] * v2; })
        return temp;
    }

    friend force_inline simd_vec<int, S> operator/(const simd_vec<int, S> &v1, int v2) {
        simd_vec<int, S> temp;
        ITERATE_8({ temp.comp_[i] = v1.comp_[i] / v2; })
        return temp;
    }

    friend force_inline simd_vec<int, S> operator+(int v1, const simd_vec<int, S> &v2) {
        simd_vec<int, S> temp;
#if 0 // requires AVX2 support
        temp.vec_ = _mm256_add_epi32(_mm256_set1_epi32(v1), v2.vec_);
#else
        ITERATE_8({ temp.comp_[i] = v1 + v2.comp_[i]; })
#endif
        return temp;
    }

    friend force_inline simd_vec<int, S> operator-(int v1, const simd_vec<int, S> &v2) {
        simd_vec<int, S> temp;
#if 0 // requires AVX2 support
        temp.vec_ = _mm256_sub_epi32(_mm256_set1_epi32(v1), v2.vec_);
#else
        ITERATE_8({ temp.comp_[i] = v1 - v2.comp_[i]; })
#endif
        return temp;
    }

    friend force_inline simd_vec<int, S> operator*(int v1, const simd_vec<int, S> &v2) {
        simd_vec<int, S> temp;
        ITERATE_8({ temp.comp_[i] = v1 * v2.comp_[i]; })
        return temp;
    }

    friend force_inline simd_vec<int, S> operator/(int v1, const simd_vec<int, S> &v2) {
        simd_vec<int, S> temp;
        ITERATE_8({ temp.comp_[i] = v1 / v2.comp_[i]; })
        return temp;
    }

    friend force_inline simd_vec<int, S> operator<(const simd_vec<int, S> &v1, const simd_vec<int, S> &v2) {
        simd_vec<int, S> ret;
#if 0 // requires AVX2 support
        ret.vec_ = _mm256_cmplt_epi32(v1.vec_, v2.vec_);
#else
        ITERATE_8({ ret.comp_[i] = v1.comp_[i] < v2.comp_[i] ? 0xFFFFFFFF : 0; })
#endif
        return ret;
    }

    friend force_inline simd_vec<int, S> operator>(const simd_vec<int, S> &v1, const simd_vec<int, S> &v2) {
        simd_vec<int, S> ret;
#if 0 // requires AVX2 support
        ret.vec_ = _mm256_cmpgt_epi32(v1.vec_, v2.vec_);
#else
        ITERATE_8({ ret.comp_[i] = v1.comp_[i] > v2.comp_[i] ? 0xFFFFFFFF : 0; })
#endif
        return ret;
    }

    friend force_inline simd_vec<int, S> operator<(const simd_vec<int, S> &v1, int v2) {
        simd_vec<int, S> ret;
#if 0 // requires AVX2 support
        ret.vec_ = _mm256_cmpgt_epi32(_mm256_set1_epi32(v2), v1.vec_);
#else
        ITERATE_8({ ret.comp_[i] = v1.comp_[i] < v2 ? 0xFFFFFFFF : 0; })
#endif
        return ret;
    }

    friend force_inline simd_vec<int, S> operator>(const simd_vec<int, S> &v1, int v2) {
        simd_vec<int, S> ret;
#if 0 // requires AVX2 support
        ret.vec_ = _mm256_cmpgt_epi32(v1.vec_, _mm256_set1_epi32(v2));
#else
        ITERATE_8({ ret.comp_[i] = v1.comp_[i] > v2 ? 0xFFFFFFFF : 0; })
#endif
        return ret;
    }

    friend force_inline simd_vec<int, S> operator>>(const simd_vec<int, S> &v1, const simd_vec<int, S> &v2) {
        simd_vec<int, S> ret;
#if 0 // requires AVX2 support
        ret.vec_ = _mm256_srlv_epi32(v1.vec_, v2.vec_);
#else
        ITERATE_8({ ret.comp_[i] = v1.comp_[i] >> v2.comp_[i]; })
#endif
        return ret;
    }

    friend force_inline simd_vec<int, S> operator>>(const simd_vec<int, S> &v1, int v2) {
        simd_vec<int, S> ret;
#if 0 // requires AVX2 support
        ret.vec_ = _mm256_srli_epi32(v1.vec_, v2);
#else
        ITERATE_8({ ret.comp_[i] = v1.comp_[i] >> v2; })
#endif
        return ret;
    }

    friend force_inline simd_vec<int, S> operator<<(const simd_vec<int, S> &v1, const simd_vec<int, S> &v2) {
        simd_vec<int, S> ret;
#if 0 // requires AVX2 support
        ret.vec_ = _mm256_sllv_epi32(v1.vec_, v2.vec_);
#else
        ITERATE_8({ ret.comp_[i] = v1.comp_[i] << v2.comp_[i]; })
#endif
        return ret;
    }

    friend force_inline simd_vec<int, S> operator<<(const simd_vec<int, S> &v1, int v2) {
        simd_vec<int, S> ret;
#if 0 // requires AVX2 support
        ret.vec_ = _mm256_slli_epi32(v1.vec_, v2);
#else
        ITERATE_8({ ret.comp_[i] = v1.comp_[i] << v2; })
#endif
        return ret;
    }

    static int size() { return S; }
    static bool is_native() { return true; }
};

#if defined(USE_AVX)
using native_simd_fvec = simd_fvec<8>;
using native_simd_ivec = simd_ivec<8>;
#endif

}
}

#pragma warning(pop)

#ifdef __GNUC__
#pragma GCC pop_options
#endif
