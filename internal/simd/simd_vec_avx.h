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

namespace ray {
namespace NS {

template <int S>
class simd_vec<typename std::enable_if<S % 8 == 0, float>::type, S> {
    public:
    union {
        __m256 vec_[S / 8];
        float comp_[S];
    };

    friend class simd_vec<int, S>;
public:
    force_inline simd_vec() = default;
    force_inline simd_vec(float f) {
        ITERATE(S/8, { vec_[i] = _mm256_set1_ps(f); })
    }
    template <typename... Tail>
    force_inline simd_vec(typename std::enable_if<sizeof...(Tail)+1 == S, float>::type head, Tail... tail) {
        const float _tail[] = { tail... };
        vec_[0] = _mm256_setr_ps(head, _tail[0], _tail[1], _tail[2], _tail[3], _tail[4], _tail[5], _tail[6]);
        if (S > 8) {
            vec_[1] = _mm256_setr_ps(_tail[7], _tail[8], _tail[9], _tail[10], _tail[11], _tail[12], _tail[13], _tail[14]);
        }

        for (int i = 15; i < S - 1; i += 8) {
            vec_[(i + 1) / 8] = _mm256_setr_ps(_tail[i], _tail[i + 1], _tail[i + 2], _tail[i + 3], _tail[i + 4], _tail[i + 5], _tail[i + 6], _tail[i + 7]);
        }
    }
    force_inline simd_vec(const float *f) {
        ITERATE(S/8, {
            vec_[i] = _mm256_loadu_ps(f);
            f += 8;
        })
    }
    force_inline simd_vec(const float *f, simd_mem_aligned_tag) {
        ITERATE(S/8, {
            vec_[i] = _mm256_load_ps(f);
            f += 8;
        })
    }

    force_inline float &operator[](int i) { return comp_[i]; }
    force_inline float operator[](int i) const { return comp_[i]; }

    force_inline simd_vec<float, S> &operator+=(const simd_vec<float, S> &rhs) {
        ITERATE(S/8,  { vec_[i] = _mm256_add_ps(vec_[i], rhs.vec_[i]); })
        return *this;
    }

    force_inline simd_vec<float, S> &operator+=(float rhs) {
        __m256 _rhs = _mm256_set1_ps(rhs);
        ITERATE(S/8,  { vec_[i] = _mm256_add_ps(vec_[i], _rhs); })
        return *this;
    }

    force_inline simd_vec<float, S> &operator-=(const simd_vec<float, S> &rhs) {
        ITERATE(S/8, { vec_[i] = _mm256_sub_ps(vec_[i], rhs.vec_[i]); })
        return *this;
    }

    force_inline simd_vec<float, S> &operator-=(float rhs) {
        ITERATE(S/8, { vec_[i] = _mm256_sub_ps(vec_[i], _mm256_set1_ps(rhs)); })
        return *this;
    }

    force_inline simd_vec<float, S> &operator*=(const simd_vec<float, S> &rhs) {
        ITERATE(S/8, { vec_[i] = _mm256_mul_ps(vec_[i], rhs.vec_[i]); })
        return *this;
    }

    force_inline simd_vec<float, S> &operator*=(float rhs) {
        ITERATE(S/8, { vec_[i] = _mm256_mul_ps(vec_[i], _mm256_set1_ps(rhs)); })
        return *this;
    }

    force_inline simd_vec<float, S> &operator/=(const simd_vec<float, S> &rhs) {
        ITERATE(S/8, { vec_[i] = _mm256_div_ps(vec_[i], rhs.vec_[i]); })
        return *this;
    }

    force_inline simd_vec<float, S> &operator/=(float rhs) {
        __m256 _rhs = _mm256_set1_ps(rhs);
        ITERATE(S/8, { vec_[i] = _mm256_div_ps(vec_[i], _rhs); })
        return *this;
    }

    force_inline simd_vec<float, S> operator-() const {
        simd_vec<float, S> temp;
        __m256 m = _mm256_set1_ps(-0.0f);
        ITERATE(S/8, { temp.vec_[i] = _mm256_xor_ps(vec_[i], m); })
        return temp;
    }

    force_inline operator simd_vec<int, S>() const {
        simd_vec<int, S> ret;
        ITERATE(S/8, { ret.vec_[i] = _mm256_cvtps_epi32(vec_[i]); })
        return ret;
    }

    force_inline simd_vec<float, S> sqrt() const {
        simd_vec<float, S> temp;
        ITERATE(S/8, { temp.vec_[i] = _mm256_sqrt_ps(vec_[i]); })
        return temp;
    }

    force_inline void copy_to(float *f) const {
        ITERATE(S/8, {
            _mm256_storeu_ps(f, vec_[i]);
            f += 8;
        })
    }

    force_inline void copy_to(float *f, simd_mem_aligned_tag) const {
        ITERATE(S/8, {
            _mm256_store_ps(f, vec_[i]);
            f += 8;
        })
    }

    force_inline void blend_to(const simd_vec<float, S> &mask, const simd_vec<float, S> &v1) {
        ITERATE(S/8, { vec_[i] = _mm256_blendv_ps(vec_[i], v1.vec_[i], mask.vec_[i]); })
    }

    force_inline static simd_vec<float, S> min(const simd_vec<float, S> &v1, const simd_vec<float, S> &v2) {
        simd_vec<float, S> temp;
        ITERATE(S/8, { temp.vec_[i] = _mm256_min_ps(v1.vec_[i], v2.vec_[i]); })
        return temp;
    }

    force_inline static simd_vec<float, S> max(const simd_vec<float, S> &v1, const simd_vec<float, S> &v2) {
        simd_vec<float, S> temp;
        ITERATE(S/8, { temp.vec_[i] = _mm256_max_ps(v1.vec_[i], v2.vec_[i]); })
        return temp;
    }

    force_inline static simd_vec<float, S> and_not(const simd_vec<float, S> &v1, const simd_vec<float, S> &v2) {
        simd_vec<float, S> temp;
        ITERATE(S/8, { temp.vec_[i] = _mm256_andnot_ps(v1.vec_[i], v2.vec_[i]); })
        return temp;
    }

    force_inline static simd_vec<float, S> floor(const simd_vec<float, S> &v1) {
        simd_vec<float, S> temp;
        ITERATE(S/8, { temp.vec_[i] = _mm256_floor_ps(v1.vec_[i]); })
        return temp;
    }

    force_inline static simd_vec<float, S> ceil(const simd_vec<float, S> &v1) {
        simd_vec<float, S> temp;
        ITERATE(S/8, { temp.vec_[i] = _mm256_ceil_ps(v1.vec_[i]); })
        return temp;
    }

    friend force_inline simd_vec<float, S> operator&(const simd_vec<float, S> &v1, const simd_vec<float, S> &v2) {
        simd_vec<float, S> temp;
        ITERATE(S/8, { temp.vec_[i] = _mm256_and_ps(v1.vec_[i], v2.vec_[i]); })
        return temp;
    }

    friend force_inline simd_vec<float, S> operator|(const simd_vec<float, S> &v1, const simd_vec<float, S> &v2) {
        simd_vec<float, S> temp;
        ITERATE(S/8, { temp.vec_[i] = _mm256_or_ps(v1.vec_[i], v2.vec_[i]); })
        return temp;
    }

    friend force_inline simd_vec<float, S> operator^(const simd_vec<float, S> &v1, const simd_vec<float, S> &v2) {
        simd_vec<float, S> temp;
        ITERATE(S/8, { temp.vec_[i] = _mm256_xor_ps(v1.vec_[i], v2.vec_[i]); })
        return temp;
    }

    friend force_inline simd_vec<float, S> operator+(const simd_vec<float, S> &v1, const simd_vec<float, S> &v2) {
        simd_vec<float, S> temp;
        ITERATE(S/8, { temp.vec_[i] = _mm256_add_ps(v1.vec_[i], v2.vec_[i]); })
        return temp;
    }

    friend force_inline simd_vec<float, S> operator-(const simd_vec<float, S> &v1, const simd_vec<float, S> &v2) {
        simd_vec<float, S> temp;
        ITERATE(S/8, { temp.vec_[i] = _mm256_sub_ps(v1.vec_[i], v2.vec_[i]); })
        return temp;
    }

    friend force_inline simd_vec<float, S> operator*(const simd_vec<float, S> &v1, const simd_vec<float, S> &v2) {
        simd_vec<float, S> temp;
        ITERATE(S/8, { temp.vec_[i] = _mm256_mul_ps(v1.vec_[i], v2.vec_[i]); })
        return temp;
    }

    friend force_inline simd_vec<float, S> operator/(const simd_vec<float, S> &v1, const simd_vec<float, S> &v2) {
        simd_vec<float, S> temp;
        ITERATE(S/8, { temp.vec_[i] = _mm256_div_ps(v1.vec_[i], v2.vec_[i]); })
        return temp;
    }

    friend force_inline simd_vec<float, S> operator+(const simd_vec<float, S> &v1, float v2) {
        simd_vec<float, S> temp;
        ITERATE(S/8, { temp.vec_[i] = _mm256_add_ps(v1.vec_[i], _mm256_set1_ps(v2)); })
        return temp;
    }

    friend force_inline simd_vec<float, S> operator-(const simd_vec<float, S> &v1, float v2) {
        simd_vec<float, S> temp;
        ITERATE(S/8, { temp.vec_[i] = _mm256_sub_ps(v1.vec_[i], _mm256_set1_ps(v2)); })
        return temp;
    }

    friend force_inline simd_vec<float, S> operator*(const simd_vec<float, S> &v1, float v2) {
        simd_vec<float, S> temp;
        ITERATE(S/8, { temp.vec_[i] = _mm256_mul_ps(v1.vec_[i], _mm256_set1_ps(v2)); })
        return temp;
    }

    friend force_inline simd_vec<float, S> operator/(const simd_vec<float, S> &v1, float v2) {
        simd_vec<float, S> temp;
        ITERATE(S/8, { temp.vec_[i] = _mm256_div_ps(v1.vec_[i], _mm256_set1_ps(v2)); })
        return temp;
    }

    friend force_inline simd_vec<float, S> operator+(float v1, const simd_vec<float, S> &v2) {
        simd_vec<float, S> temp;
        ITERATE(S/8, { temp.vec_[i] = _mm256_add_ps(_mm256_set1_ps(v1), v2.vec_[i]); })
        return temp;
    }

    friend force_inline simd_vec<float, S> operator-(float v1, const simd_vec<float, S> &v2) {
        simd_vec<float, S> temp;
        ITERATE(S/8, { temp.vec_[i] = _mm256_sub_ps(_mm256_set1_ps(v1), v2.vec_[i]); })
        return temp;
    }

    friend force_inline simd_vec<float, S> operator*(float v1, const simd_vec<float, S> &v2) {
        simd_vec<float, S> temp;
        ITERATE(S/8, { temp.vec_[i] = _mm256_mul_ps(_mm256_set1_ps(v1), v2.vec_[i]); })
        return temp;
    }

    friend force_inline simd_vec<float, S> operator/(float v1, const simd_vec<float, S> &v2) {
        simd_vec<float, S> temp;
        ITERATE(S/8, { temp.vec_[i] = _mm256_div_ps(_mm256_set1_ps(v1), v2.vec_[i]); })
        return temp;
    }

    friend force_inline simd_vec<float, S> operator<(const simd_vec<float, S> &v1, const simd_vec<float, S> &v2) {
        simd_vec<float, S> ret;
        ITERATE(S/8, { ret.vec_[i] = _mm256_cmp_ps(v1.vec_[i], v2.vec_[i], _CMP_LT_OS); })
        return ret;
    }

    friend force_inline simd_vec<float, S> operator<=(const simd_vec<float, S> &v1, const simd_vec<float, S> &v2) {
        simd_vec<float, S> ret;
        ITERATE(S/8, { ret.vec_[i] = _mm256_cmp_ps(v1.vec_[i], v2.vec_[i], _CMP_LE_OS); })
        return ret;
    }

    friend force_inline simd_vec<float, S> operator>(const simd_vec<float, S> &v1, const simd_vec<float, S> &v2) {
        simd_vec<float, S> ret;
        ITERATE(S/8, { ret.vec_[i] = _mm256_cmp_ps(v1.vec_[i], v2.vec_[i], _CMP_GT_OS); })
        return ret;
    }

    friend force_inline simd_vec<float, S> operator>=(const simd_vec<float, S> &v1, const simd_vec<float, S> &v2) {
        simd_vec<float, S> ret;
        ITERATE(S/8, { ret.vec_[i] = _mm256_cmp_ps(v1.vec_[i], v2.vec_[i], _CMP_GE_OS); })
        return ret;
    }

    friend force_inline simd_vec<float, S> operator<(const simd_vec<float, S> &v1, float v2) {
        simd_vec<float, S> ret;
        ITERATE(S/8, { ret.vec_[i] = _mm256_cmp_ps(v1.vec_[i], _mm256_set1_ps(v2), _CMP_LT_OS); })
        return ret;
    }

    friend force_inline simd_vec<float, S> operator<=(const simd_vec<float, S> &v1, float v2) {
        simd_vec<float, S> ret;
        ITERATE(S/8, { ret.vec_[i] = _mm256_cmp_ps(v1.vec_[i], _mm256_set1_ps(v2), _CMP_LE_OS); })
        return ret;
    }

    friend force_inline simd_vec<float, S> operator>(const simd_vec<float, S> &v1, float v2) {
        simd_vec<float, S> ret;
        ITERATE(S/8, { ret.vec_[i] = _mm256_cmp_ps(v1.vec_[i], _mm256_set1_ps(v2), _CMP_GT_OS); })
        return ret;
    }

    friend force_inline simd_vec<float, S> operator>=(const simd_vec<float, S> &v1, float v2) {
        simd_vec<float, S> ret;
        ITERATE(S/8, { ret.vec_[i] = _mm256_cmp_ps(v1.vec_[i], _mm256_set1_ps(v2), _CMP_GE_OS); })
        return ret;
    }

    static const size_t alignment = alignof(__m256);

    static int size() { return S; }
    static int native_count() { return S / 8; }
    static bool is_native() { return native_count() == 1; }
};

template <int S>
class simd_vec<typename std::enable_if<S % 8 == 0, int>::type, S> {
    union {
        __m256i vec_[S / 8];
        int comp_[S];
    };

    friend class simd_vec<float, S>;
public:
    force_inline simd_vec() = default;
    force_inline simd_vec(int f) {
        ITERATE(S/8, {
            vec_[i] = _mm256_set1_epi32(f);
        })
    }
    template <typename... Tail>
    force_inline simd_vec(typename std::enable_if<sizeof...(Tail)+1 == S, int>::type head, Tail... tail) {
        const int _tail[] = { tail... };
        vec_[0] = _mm256_setr_epi32(head, _tail[0], _tail[1], _tail[2], _tail[3], _tail[4], _tail[5], _tail[6]);
        if (S > 8) {
            vec_[1] = _mm256_setr_epi32(_tail[7], _tail[8], _tail[9], _tail[10], _tail[11], _tail[12], _tail[13], _tail[14]);
        }

        for (int i = 15; i < S - 1; i += 8) {
            vec_[(i + 1) / 8] = _mm256_setr_epi32(_tail[i], _tail[i + 1], _tail[i + 2], _tail[i + 3], _tail[i + 4], _tail[i + 5], _tail[i + 6], _tail[i + 7]);
        }
    }
    force_inline simd_vec(const int *f) {
        ITERATE(S/8, {
            vec_[i] = _mm256_loadu_si256((const __m256i *)f);
            f += 8;
        })
    }
    force_inline simd_vec(const int *f, simd_mem_aligned_tag) {
        ITERATE(S/8, {
            vec_[i] = _mm256_load_si256((const __m256i *)f);
            f += 8;
        })
    }

    force_inline int &operator[](int i) { return comp_[i]; }
    force_inline int operator[](int i) const { return comp_[i]; }

    force_inline simd_vec<int, S> &operator+=(const simd_vec<int, S> &rhs) {
        ITERATE(S/8, { vec_[i] = _mm256_add_epi32(vec_[i], rhs.vec_[i]); })
        return *this;
    }

    force_inline simd_vec<int, S> &operator+=(int rhs) {
        ITERATE(S/8, { vec_[i] = _mm256_add_epi32(vec_[i], _mm256_set1_epi32(rhs)); })
        return *this;
    }

    force_inline simd_vec<int, S> &operator-=(const simd_vec<int, S> &rhs) {
        ITERATE(S/8, { vec_[i] = _mm256_sub_epi32(vec_[i], rhs.vec_[i]); })
        return *this;
    }

    force_inline simd_vec<int, S> &operator-=(int rhs) {
        ITERATE(S/8, { vec_[i] = _mm256_sub_epi32(vec_[i], _mm256_set1_epi32(rhs)); })
        return *this;
    }

    force_inline simd_vec<int, S> &operator*=(const simd_vec<int, S> &rhs) {
        ITERATE(S, { comp_[i] = comp_[i] * rhs.comp_[i]; })
        return *this;
    }

    force_inline simd_vec<int, S> &operator*=(int rhs) {
        ITERATE(S, { comp_[i] = comp_[i] * rhs; })
        return *this;
    }

    force_inline simd_vec<int, S> &operator/=(const simd_vec<int, S> &rhs) {
        ITERATE(S, { comp_[i] = comp_[i] / rhs.comp_[i]; })
        return *this;
    }

    force_inline simd_vec<int, S> &operator/=(int rhs) {
        ITERATE(S, { comp_[i] = comp_[i] / rhs; })
        return *this;
    }

    force_inline simd_vec<int, S> operator==(int rhs) const {
        simd_vec<int, S> ret;
        ITERATE(S/8, { ret.vec_[i] = _mm256_cmpeq_epi32(vec_[i], _mm256_set1_epi32(rhs)); })
        return ret;
    }

    force_inline operator simd_vec<float, S>() const {
        simd_vec<float, S> ret;
        ITERATE(S/8, { ret.vec_[i] = _mm256_cvtepi32_ps(vec_[i]); })
        return ret;
    }

    force_inline void copy_to(int *f) const {
        ITERATE(S/8, {
            _mm256_storeu_si256((__m256i *)f, vec_[i]);
            f += 8;
        })
    }

    force_inline void copy_to(int *f, simd_mem_aligned_tag) const {
        ITERATE(S/8, {
            _mm256_store_si256((__m256i *)f, vec_[i]);
            f += 8;
        })
    }

    force_inline void blend_to(const simd_vec<int, S> &mask, const simd_vec<int, S> &v1) {
        ITERATE(S/8, { vec_[i] = _mm256_blendv_epi8(vec_[i], v1.vec_[i], mask.vec_[i]); })
    }

    force_inline bool all_zeros() const {
        int res = 1;
        ITERATE(S/8, { res &= _mm256_test_all_zeros(vec_[i], vec_[i]); })
        return res != 0;
    }

    force_inline bool all_zeros(const simd_vec<int, S> &mask) const {
        int res = 1;
        ITERATE(S/8, { res &= _mm256_test_all_zeros(vec_[i], mask.vec_[i]); })
        return res != 0;
    }

    force_inline bool not_all_zeros() const {
        ITERATE(S/8, { if (!_mm256_test_all_zeros(vec_[i], vec_[i])) return true; })
        return false;
    }

    force_inline static simd_vec<int, S> min(const simd_vec<int, S> &v1, const simd_vec<int, S> &v2) {
        simd_vec<int, S> temp;
        ITERATE(S/8, { temp.vec_[i] = _mm256_min_epi32(v1.vec_[i], v2.vec_[i]); })
        return temp;
    }

    force_inline static simd_vec<int, S> max(const simd_vec<int, S> &v1, const simd_vec<int, S> &v2) {
        simd_vec<int, S> temp;
        ITERATE(S/8, { temp.vec_[i] = _mm256_max_epi32(v1.vec_[i], v2.vec_[i]); })
        return temp;
    }

    force_inline static simd_vec<int, S> and_not(const simd_vec<int, S> &v1, const simd_vec<int, S> &v2) {
        simd_vec<int, S> temp;
        ITERATE(S/8, { temp.vec_[i] = _mm256_andnot_si256(v1.vec_[i], v2.vec_[i]); })
        return temp;
    }

    friend force_inline simd_vec<int, S> operator&(const simd_vec<int, S> &v1, const simd_vec<int, S> &v2) {
        simd_vec<int, S> temp;
        ITERATE(S/8, { temp.vec_[i] = _mm256_and_si256(v1.vec_[i], v2.vec_[i]); })
        return temp;
    }

    friend force_inline simd_vec<int, S> operator|(const simd_vec<int, S> &v1, const simd_vec<int, S> &v2) {
        simd_vec<int, S> temp;
        ITERATE(S/8, { temp.vec_[i] = _mm256_or_si256(v1.vec_[i], v2.vec_[i]); })
        return temp;
    }

    friend force_inline simd_vec<int, S> operator^(const simd_vec<int, S> &v1, const simd_vec<int, S> &v2) {
        simd_vec<int, S> temp;
        ITERATE(S/8, { temp.vec_[i] = _mm256_xor_si256(v1.vec_[i], v2.vec_[i]); });
        return temp;
    }

    friend force_inline simd_vec<int, S> operator+(const simd_vec<int, S> &v1, const simd_vec<int, S> &v2) {
        simd_vec<int, S> temp;
        ITERATE(S/8, { temp.vec_[i] = _mm256_add_epi32(v1.vec_[i], v2.vec_[i]); })
        return temp;
    }

    friend force_inline simd_vec<int, S> operator-(const simd_vec<int, S> &v1, const simd_vec<int, S> &v2) {
        simd_vec<int, S> temp;
        ITERATE(S/8, { temp.vec_[i] = _mm256_sub_epi32(v1.vec_[i], v2.vec_[i]); })
        return temp;
    }

    friend force_inline simd_vec<int, S> operator*(const simd_vec<int, S> &v1, const simd_vec<int, S> &v2) {
        simd_vec<int, S> temp;
        ITERATE(S, { temp.comp_[i] = v1.comp_[i] * v2.comp_[i]; })
        return temp;
    }

    friend force_inline simd_vec<int, S> operator/(const simd_vec<int, S> &v1, const simd_vec<int, S> &v2) {
        simd_vec<int, S> temp;
        ITERATE(S/8, { temp.comp_[i] = v1.vec_[i] / v2.vec_[i]; })
        return temp;
    }

    friend force_inline simd_vec<int, S> operator+(const simd_vec<int, S> &v1, int v2) {
        simd_vec<int, S> temp;
        ITERATE(S/8, { temp.vec_[i] = _mm256_add_epi32(v1.vec_[i], _mm256_set1_epi32(v2)); })
        return temp;
    }

    friend force_inline simd_vec<int, S> operator-(const simd_vec<int, S> &v1, int v2) {
        simd_vec<int, S> temp;
        ITERATE(S/8, { temp.vec_[i] = _mm256_sub_epi32(v1.vec_[i], _mm256_set1_epi32(v2)); })
        return temp;
    }

    friend force_inline simd_vec<int, S> operator*(const simd_vec<int, S> &v1, int v2) {
        simd_vec<int, S> temp;
        ITERATE(S, { temp.comp_[i] = v1.comp_[i] * v2; })
        return temp;
    }

    friend force_inline simd_vec<int, S> operator/(const simd_vec<int, S> &v1, int v2) {
        simd_vec<int, S> temp;
        ITERATE(S, { temp.comp_[i] = v1.comp_[i] / v2; })
        return temp;
    }

    friend force_inline simd_vec<int, S> operator+(int v1, const simd_vec<int, S> &v2) {
        simd_vec<int, S> temp;
        ITERATE(S/8, { temp.vec_[i] = _mm256_add_epi32(_mm256_set1_epi32(v1), v2.vec_[i]); })
        return temp;
    }

    friend force_inline simd_vec<int, S> operator-(int v1, const simd_vec<int, S> &v2) {
        simd_vec<int, S> temp;
        ITERATE(S/8, { temp.vec_[i] = _mm256_sub_epi32(_mm256_set1_epi32(v1), v2.vec_[i]); })
        return temp;
    }

    friend force_inline simd_vec<int, S> operator*(int v1, const simd_vec<int, S> &v2) {
        simd_vec<int, S> temp;
        ITERATE(S, { temp.comp_[i] = v1 * v2.comp_[i]; })
        return temp;
    }

    friend force_inline simd_vec<int, S> operator/(int v1, const simd_vec<int, S> &v2) {
        simd_vec<int, S> temp;
        ITERATE(S, { temp.comp_[i] = v1 / v2.comp_[i]; })
        return temp;
    }

    friend force_inline simd_vec<int, S> operator<(const simd_vec<int, S> &v1, const simd_vec<int, S> &v2) {
        simd_vec<int, S> ret;
        ITERATE(S/8, { ret.vec_[i] = _mm256_cmplt_epi32(v1.vec_[i], v2.vec_[i]); })
        return ret;
    }

    friend force_inline simd_vec<int, S> operator>(const simd_vec<int, S> &v1, const simd_vec<int, S> &v2) {
        simd_vec<int, S> ret;
        ITERATE(S/8, { ret.vec_[i] = _mm256_cmpgt_epi32(v1.vec_[i], v2.vec_[i]); })
        return ret;
    }

    friend force_inline simd_vec<int, S> operator<(const simd_vec<int, S> &v1, int v2) {
        simd_vec<int, S> ret;
        ITERATE(S/8, { ret.vec_[i] = _mm256_cmpgt_epi32(_mm256_set1_epi32(v2), v1.vec_[i]); })
        return ret;
    }

    friend force_inline simd_vec<int, S> operator>(const simd_vec<int, S> &v1, int v2) {
        simd_vec<int, S> ret;
        ITERATE(S/8, { ret.vec_[i] = _mm256_cmpgt_epi32(v1.vec_[i], _mm256_set1_epi32(v2)); })
        return ret;
    }

    friend force_inline simd_vec<int, S> operator>>(const simd_vec<int, S> &v1, int v2) {
        simd_vec<int, S> temp;
        ITERATE(S/8, { temp.vec_[i] = _mm256_srli_epi32(v1.vec_[i], v2); })
        return temp;
    }

    static const size_t alignment = alignof(__m256i);

    static int size() { return S; }
    static int native_count() { return S / 8; }
    static bool is_native() { return native_count() == 1; }
};

#if defined(USE_AVX)
using native_simd_fvec = simd_fvec<8>;
using native_simd_ivec = simd_ivec<8>;
#endif

}
}

#ifdef __GNUC__
#pragma GCC pop_options
#endif
