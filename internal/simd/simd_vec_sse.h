//#pragma once

#include <type_traits>

#include <immintrin.h>
#include <xmmintrin.h>

#ifdef __GNUC__
#pragma GCC push_options
#pragma GCC target("sse2")
#pragma clang attribute push(__attribute__((target("sse2"))), apply_to = function)
#endif

namespace Ray {
namespace NS {

template <> class simd_vec<int, 4>;

template <> class simd_vec<float, 4> {
    union {
        __m128 vec_;
        float comp_[4];
    };

    friend class simd_vec<int, 4>;

  public:
    force_inline simd_vec() = default;
    force_inline simd_vec(const float f) { vec_ = _mm_set1_ps(f); }
    template <typename... Tail> force_inline simd_vec(const float f1, const float f2, const float f3, const float f4) {
        vec_ = _mm_setr_ps(f1, f2, f3, f4);
    }
    force_inline explicit simd_vec(const float *f) { vec_ = _mm_loadu_ps(f); }
    force_inline simd_vec(const float *f, simd_mem_aligned_tag) { vec_ = _mm_load_ps(f); }

    force_inline float &operator[](const int i) { return comp_[i]; }
    force_inline const float &operator[](const int i) const { return comp_[i]; }

    force_inline simd_vec<float, 4> &operator+=(const simd_vec<float, 4> &rhs) {
        vec_ = _mm_add_ps(vec_, rhs.vec_);
        return *this;
    }

    force_inline simd_vec<float, 4> &operator+=(const float rhs) {
        vec_ = _mm_add_ps(vec_, _mm_set1_ps(rhs));
        return *this;
    }

    force_inline simd_vec<float, 4> &operator-=(const simd_vec<float, 4> &rhs) {
        vec_ = _mm_sub_ps(vec_, rhs.vec_);
        return *this;
    }

    force_inline simd_vec<float, 4> &operator-=(const float rhs) {
        vec_ = _mm_sub_ps(vec_, _mm_set1_ps(rhs));
        return *this;
    }

    force_inline simd_vec<float, 4> &operator*=(const simd_vec<float, 4> &rhs) {
        vec_ = _mm_mul_ps(vec_, rhs.vec_);
        return *this;
    }

    force_inline simd_vec<float, 4> &operator*=(const float rhs) {
        vec_ = _mm_mul_ps(vec_, _mm_set1_ps(rhs));
        return *this;
    }

    force_inline simd_vec<float, 4> &operator/=(const simd_vec<float, 4> &rhs) {
        vec_ = _mm_div_ps(vec_, rhs.vec_);
        return *this;
    }

    force_inline simd_vec<float, 4> &operator/=(const float rhs) {
        vec_ = _mm_div_ps(vec_, _mm_set1_ps(rhs));
        return *this;
    }

    force_inline simd_vec<float, 4> &operator|=(const simd_vec<float, 4> &rhs) {
        vec_ = _mm_or_ps(vec_, rhs.vec_);
        return *this;
    }

    force_inline simd_vec<float, 4> &operator|=(const float rhs) {
        vec_ = _mm_or_ps(vec_, _mm_set1_ps(rhs));
        return *this;
    }

    force_inline simd_vec<float, 4> operator-() const {
        simd_vec<float, 4> temp;
        __m128 m = _mm_set1_ps(-0.0f);
        temp.vec_ = _mm_xor_ps(vec_, m);
        return temp;
    }

    force_inline simd_vec<float, 4> operator<(const simd_vec<float, 4> &rhs) const {
        simd_vec<float, 4> ret;
        ret.vec_ = _mm_cmplt_ps(vec_, rhs.vec_);
        return ret;
    }

    force_inline simd_vec<float, 4> operator<=(const simd_vec<float, 4> &rhs) const {
        simd_vec<float, 4> ret;
        ret.vec_ = _mm_cmple_ps(vec_, rhs.vec_);
        return ret;
    }

    force_inline simd_vec<float, 4> operator>(const simd_vec<float, 4> &rhs) const {
        simd_vec<float, 4> ret;
        ret.vec_ = _mm_cmpgt_ps(vec_, rhs.vec_);
        return ret;
    }

    force_inline simd_vec<float, 4> operator>=(const simd_vec<float, 4> &rhs) const {
        simd_vec<float, 4> ret;
        ret.vec_ = _mm_cmpge_ps(vec_, rhs.vec_);
        return ret;
    }

    force_inline simd_vec<float, 4> operator~() const {
        simd_vec<float, 4> ret;
        ret.vec_ = _mm_castsi128_ps(_mm_andnot_si128(_mm_castps_si128(vec_), _mm_set1_epi32(~0)));
        return ret;
    }

    force_inline simd_vec<float, 4> operator<(float rhs) const {
        simd_vec<float, 4> ret;
        ret.vec_ = _mm_cmplt_ps(vec_, _mm_set1_ps(rhs));
        return ret;
    }

    force_inline simd_vec<float, 4> operator<=(float rhs) const {
        simd_vec<float, 4> ret;
        ret.vec_ = _mm_cmple_ps(vec_, _mm_set1_ps(rhs));
        return ret;
    }

    force_inline simd_vec<float, 4> operator>(float rhs) const {
        simd_vec<float, 4> ret;
        ret.vec_ = _mm_cmpgt_ps(vec_, _mm_set1_ps(rhs));
        return ret;
    }

    force_inline simd_vec<float, 4> operator>=(float rhs) const {
        simd_vec<float, 4> ret;
        ret.vec_ = _mm_cmpge_ps(vec_, _mm_set1_ps(rhs));
        return ret;
    }

    force_inline operator simd_vec<int, 4>() const;

    force_inline simd_vec<float, 4> sqrt() const {
        simd_vec<float, 4> temp;
        temp.vec_ = _mm_sqrt_ps(vec_);
        return temp;
    }

    force_inline float length() const {
        __m128 r1, r2;
        r1 = _mm_mul_ps(vec_, vec_);

        r2 = _mm_shuffle_ps(r1, r1, _MM_SHUFFLE(2, 3, 0, 1));
        r1 = _mm_add_ps(r1, r2);
        r2 = _mm_shuffle_ps(r1, r1, _MM_SHUFFLE(0, 1, 2, 3));
        r1 = _mm_add_ps(r1, r2);

        return _mm_cvtss_f32(_mm_sqrt_ss(r1));
    }

    force_inline float length2() const {
        __m128 r1, r2;
        r1 = _mm_mul_ps(vec_, vec_);

        r2 = _mm_shuffle_ps(r1, r1, _MM_SHUFFLE(2, 3, 0, 1));
        r1 = _mm_add_ps(r1, r2);
        r2 = _mm_shuffle_ps(r1, r1, _MM_SHUFFLE(0, 1, 2, 3));
        r1 = _mm_add_ps(r1, r2);

        return _mm_cvtss_f32(r1);
    }

    force_inline void copy_to(float *f) const { _mm_storeu_ps(f, vec_); }
    force_inline void copy_to(float *f, simd_mem_aligned_tag) const { _mm_store_ps(f, vec_); }

    force_inline void blend_to(const simd_vec<float, 4> &mask, const simd_vec<float, 4> &v1) {
#if defined(USE_SSE41)
        vec_ = _mm_blendv_ps(vec_, v1.vec_, mask.vec_);
#else
        __m128 temp1 = _mm_and_ps(mask.vec_, v1.vec_);
        __m128 temp2 = _mm_andnot_ps(mask.vec_, vec_);
        vec_ = _mm_or_ps(temp1, temp2);
#endif
    }

    force_inline void blend_inv_to(const simd_vec<float, 4> &mask, const simd_vec<float, 4> &v1) {
#if defined(USE_SSE41)
        vec_ = _mm_blendv_ps(v1.vec_, vec_, mask.vec_);
#else
        __m128 temp1 = _mm_andnot_ps(mask.vec_, v1.vec_);
        __m128 temp2 = _mm_and_ps(mask.vec_, vec_);
        vec_ = _mm_or_ps(temp1, temp2);
#endif
    }

    force_inline static simd_vec<float, 4> min(const simd_vec<float, 4> &v1, const simd_vec<float, 4> &v2) {
        simd_vec<float, 4> temp;
        temp.vec_ = _mm_min_ps(v1.vec_, v2.vec_);
        return temp;
    }

    force_inline static simd_vec<float, 4> max(const simd_vec<float, 4> &v1, const simd_vec<float, 4> &v2) {
        simd_vec<float, 4> temp;
        temp.vec_ = _mm_max_ps(v1.vec_, v2.vec_);
        return temp;
    }

    force_inline static simd_vec<float, 4> and_not(const simd_vec<float, 4> &v1, const simd_vec<float, 4> &v2) {
        simd_vec<float, 4> temp;
        temp.vec_ = _mm_andnot_ps(v1.vec_, v2.vec_);
        return temp;
    }

    force_inline static simd_vec<float, 4> floor(const simd_vec<float, 4> &v1) {
        simd_vec<float, 4> temp;
#if defined(USE_SSE41)
        temp.vec_ = _mm_floor_ps(v1.vec_);
#else
        __m128 t = _mm_cvtepi32_ps(_mm_cvttps_epi32(v1.vec_));
        __m128 r = _mm_sub_ps(t, _mm_and_ps(_mm_cmplt_ps(v1.vec_, t), _mm_set1_ps(1.0f)));
        temp.vec_ = r;
#endif
        return temp;
    }

    force_inline static simd_vec<float, 4> ceil(const simd_vec<float, 4> &v1) {
        simd_vec<float, 4> temp;
        __m128 t = _mm_cvtepi32_ps(_mm_cvttps_epi32(v1.vec_));
        __m128 r = _mm_add_ps(t, _mm_and_ps(_mm_cmpgt_ps(v1.vec_, t), _mm_set1_ps(1.0f)));
        temp.vec_ = r;
        return temp;
    }

    friend force_inline simd_vec<float, 4> operator&(const simd_vec<float, 4> &v1, const simd_vec<float, 4> &v2) {
        simd_vec<float, 4> temp;
        temp.vec_ = _mm_and_ps(v1.vec_, v2.vec_);
        return temp;
    }

    friend force_inline simd_vec<float, 4> operator|(const simd_vec<float, 4> &v1, const simd_vec<float, 4> &v2) {
        simd_vec<float, 4> temp;
        temp.vec_ = _mm_or_ps(v1.vec_, v2.vec_);
        return temp;
    }

    friend force_inline simd_vec<float, 4> operator^(const simd_vec<float, 4> &v1, const simd_vec<float, 4> &v2) {
        simd_vec<float, 4> temp;
        temp.vec_ = _mm_xor_ps(v1.vec_, v2.vec_);
        return temp;
    }

    friend force_inline simd_vec<float, 4> operator+(const simd_vec<float, 4> &v1, const simd_vec<float, 4> &v2) {
        simd_vec<float, 4> ret;
        ret.vec_ = _mm_add_ps(v1.vec_, v2.vec_);
        return ret;
    }

    friend force_inline simd_vec<float, 4> operator-(const simd_vec<float, 4> &v1, const simd_vec<float, 4> &v2) {
        simd_vec<float, 4> ret;
        ret.vec_ = _mm_sub_ps(v1.vec_, v2.vec_);
        return ret;
    }

    force_inline simd_vec<float, 4> operator==(float rhs) const {
        simd_vec<float, 4> ret;
        ret.vec_ = _mm_cmpeq_ps(vec_, _mm_set1_ps(rhs));
        return ret;
    }

    force_inline simd_vec<float, 4> operator==(const simd_vec<float, 4> &rhs) const {
        simd_vec<float, 4> ret;
        ret.vec_ = _mm_cmpeq_ps(vec_, rhs.vec_);
        return ret;
    }

    force_inline simd_vec<float, 4> operator!=(float rhs) const {
        simd_vec<float, 4> ret;
        ret.vec_ = _mm_cmpneq_ps(vec_, _mm_set1_ps(rhs));
        return ret;
    }

    force_inline simd_vec<float, 4> operator!=(const simd_vec<float, 4> &rhs) const {
        simd_vec<float, 4> ret;
        ret.vec_ = _mm_cmpneq_ps(vec_, rhs.vec_);
        return ret;
    }

    friend force_inline simd_vec<float, 4> operator*(const simd_vec<float, 4> &v1, const simd_vec<float, 4> &v2) {
        simd_vec<float, 4> ret;
        ret.vec_ = _mm_mul_ps(v1.vec_, v2.vec_);
        return ret;
    }

    friend force_inline simd_vec<float, 4> operator/(const simd_vec<float, 4> &v1, const simd_vec<float, 4> &v2) {
        simd_vec<float, 4> ret;
        ret.vec_ = _mm_div_ps(v1.vec_, v2.vec_);
        return ret;
    }

    friend force_inline simd_vec<float, 4> operator+(const simd_vec<float, 4> &v1, float v2) {
        simd_vec<float, 4> ret;
        ret.vec_ = _mm_add_ps(v1.vec_, _mm_set1_ps(v2));
        return ret;
    }

    friend force_inline simd_vec<float, 4> operator-(const simd_vec<float, 4> &v1, float v2) {
        simd_vec<float, 4> ret;
        ret.vec_ = _mm_sub_ps(v1.vec_, _mm_set1_ps(v2));
        return ret;
    }

    friend force_inline simd_vec<float, 4> operator*(const simd_vec<float, 4> &v1, float v2) {
        simd_vec<float, 4> ret;
        ret.vec_ = _mm_mul_ps(v1.vec_, _mm_set1_ps(v2));
        return ret;
    }

    friend force_inline simd_vec<float, 4> operator/(const simd_vec<float, 4> &v1, float v2) {
        simd_vec<float, 4> ret;
        ret.vec_ = _mm_div_ps(v1.vec_, _mm_set1_ps(v2));
        return ret;
    }

    friend force_inline simd_vec<float, 4> operator+(float v1, const simd_vec<float, 4> &v2) {
        return operator+(v2, v1);
    }

    friend force_inline simd_vec<float, 4> operator-(float v1, const simd_vec<float, 4> &v2) {
        simd_vec<float, 4> ret;
        ret.vec_ = _mm_sub_ps(_mm_set1_ps(v1), v2.vec_);
        return ret;
    }

    friend force_inline simd_vec<float, 4> operator*(float v1, const simd_vec<float, 4> &v2) {
        simd_vec<float, 4> ret;
        ret.vec_ = _mm_mul_ps(_mm_set1_ps(v1), v2.vec_);
        return ret;
    }

    friend force_inline simd_vec<float, 4> operator/(float v1, const simd_vec<float, 4> &v2) {
        simd_vec<float, 4> ret;
        ret.vec_ = _mm_div_ps(_mm_set1_ps(v1), v2.vec_);
        return ret;
    }

    friend force_inline float dot(const simd_vec<float, 4> &v1, const simd_vec<float, 4> &v2) {
        __m128 r1, r2;
        r1 = _mm_mul_ps(v1.vec_, v2.vec_);
        r2 = _mm_shuffle_ps(r1, r1, _MM_SHUFFLE(2, 3, 0, 1));
        r1 = _mm_add_ps(r1, r2);
        r2 = _mm_shuffle_ps(r1, r1, _MM_SHUFFLE(0, 1, 2, 3));
        r1 = _mm_add_ps(r1, r2);
        return _mm_cvtss_f32(r1);
    }

    friend force_inline simd_vec<float, 4> clamp(const simd_vec<float, 4> &v1, float min, float max) {
        simd_vec<float, 4> ret;
        ret.vec_ = _mm_max_ps(_mm_set1_ps(min), _mm_min_ps(v1.vec_, _mm_set1_ps(max)));
        return ret;
    }

    friend force_inline simd_vec<float, 4> pow(const simd_vec<float, 4> &v1, const simd_vec<float, 4> &v2) {
        simd_vec<float, 4> ret;
        ITERATE_4({ ret.comp_[i] = std::pow(v1.comp_[i], v2.comp_[i]); })
        return ret;
    }

    friend force_inline simd_vec<float, 4> normalize(const simd_vec<float, 4> &v1) { return v1 / v1.length(); }

    friend force_inline const float *value_ptr(const simd_vec<float, 4> &v1) { return &v1.comp_[0]; }

    static int size() { return 4; }
    static bool is_native() { return true; }
};

template <> class simd_vec<int, 4> {
    union {
        __m128i vec_;
        int comp_[4];
    };

    friend class simd_vec<float, 4>;

  public:
    force_inline simd_vec() = default;
    force_inline simd_vec(int f) { vec_ = _mm_set1_epi32(f); }
    force_inline simd_vec(int i1, int i2, int i3, int i4) { vec_ = _mm_setr_epi32(i1, i2, i3, i4); }
    force_inline explicit simd_vec(const int *f) { vec_ = _mm_loadu_si128((const __m128i *)f); }
    force_inline simd_vec(const int *f, simd_mem_aligned_tag) { vec_ = _mm_load_si128((const __m128i *)f); }

    force_inline int &operator[](int i) { return comp_[i]; }
    force_inline const int &operator[](int i) const { return comp_[i]; }

    force_inline simd_vec<int, 4> &operator+=(const simd_vec<int, 4> &rhs) {
        vec_ = _mm_add_epi32(vec_, rhs.vec_);
        return *this;
    }

    force_inline simd_vec<int, 4> &operator+=(int rhs) {
        vec_ = _mm_add_epi32(vec_, _mm_set1_epi32(rhs));
        return *this;
    }

    force_inline simd_vec<int, 4> &operator-=(const simd_vec<int, 4> &rhs) {
        vec_ = _mm_sub_epi32(vec_, rhs.vec_);
        return *this;
    }

    force_inline simd_vec<int, 4> &operator-=(int rhs) {
        vec_ = _mm_sub_epi32(vec_, _mm_set1_epi32(rhs));
        return *this;
    }

    force_inline simd_vec<int, 4> &operator*=(const simd_vec<int, 4> &rhs) {
        ITERATE_4({ comp_[i] = comp_[i] * rhs.comp_[i]; })
        return *this;
    }

    force_inline simd_vec<int, 4> &operator*=(int rhs) {
        ITERATE_4({ comp_[i] = comp_[i] * rhs; })
        return *this;
    }

    force_inline simd_vec<int, 4> &operator/=(const simd_vec<int, 4> &rhs) {
        ITERATE_4({ comp_[i] = comp_[i] / rhs.comp_[i]; })
        return *this;
    }

    force_inline simd_vec<int, 4> &operator/=(int rhs) {
        ITERATE_4({ comp_[i] = comp_[i] / rhs; })
        return *this;
    }

    force_inline simd_vec<int, 4> &operator|=(const simd_vec<int, 4> &rhs) {
        vec_ = _mm_or_si128(vec_, rhs.vec_);
        return *this;
    }

    force_inline simd_vec<int, 4> &operator|=(const int rhs) {
        vec_ = _mm_or_si128(vec_, _mm_set1_epi32(rhs));
        return *this;
    }

    force_inline simd_vec<int, 4> operator-() const {
        simd_vec<int, 4> temp;
        temp.vec_ = _mm_sub_epi32(_mm_setzero_si128(), vec_);
        return temp;
    }

    force_inline simd_vec<int, 4> operator==(int rhs) const {
        simd_vec<int, 4> ret;
        ret.vec_ = _mm_cmpeq_epi32(vec_, _mm_set1_epi32(rhs));
        return ret;
    }

    force_inline simd_vec<int, 4> operator==(const simd_vec<int, 4> &rhs) const {
        simd_vec<int, 4> ret;
        ret.vec_ = _mm_cmpeq_epi32(vec_, rhs.vec_);
        return ret;
    }

    force_inline simd_vec<int, 4> operator!=(int rhs) const {
        simd_vec<int, 4> ret;
        ret.vec_ = _mm_andnot_si128(_mm_cmpeq_epi32(vec_, _mm_set1_epi32(rhs)), _mm_set1_epi32(~0));
        return ret;
    }

    force_inline simd_vec<int, 4> operator!=(const simd_vec<int, 4> &rhs) const {
        simd_vec<int, 4> ret;
        ret.vec_ = _mm_andnot_si128(_mm_cmpeq_epi32(vec_, rhs.vec_), _mm_set1_epi32(~0));
        return ret;
    }

    force_inline simd_vec<int, 4> operator<(const simd_vec<int, 4> &rhs) const {
        simd_vec<int, 4> ret;
        ret.vec_ = _mm_cmplt_epi32(vec_, rhs.vec_);
        return ret;
    }

    force_inline simd_vec<int, 4> operator<=(const simd_vec<int, 4> &rhs) const {
        simd_vec<int, 4> ret;
        ret.vec_ = _mm_andnot_si128(_mm_cmpgt_epi32(vec_, rhs.vec_), _mm_set_epi32(~0, ~0, ~0, ~0));
        return ret;
    }

    force_inline simd_vec<int, 4> operator>(const simd_vec<int, 4> &rhs) const {
        simd_vec<int, 4> ret;
        ret.vec_ = _mm_cmpgt_epi32(vec_, rhs.vec_);
        return ret;
    }

    force_inline simd_vec<int, 4> operator>=(const simd_vec<int, 4> &rhs) const {
        simd_vec<int, 4> ret;
        ret.vec_ = _mm_andnot_si128(_mm_cmplt_epi32(vec_, rhs.vec_), _mm_set_epi32(~0, ~0, ~0, ~0));
        return ret;
    }

    force_inline simd_vec<int, 4> &operator&=(const simd_vec<int, 4> &rhs) {
        vec_ = _mm_and_si128(vec_, rhs.vec_);
        return *this;
    }

    force_inline simd_vec<int, 4> operator<(int rhs) const {
        simd_vec<int, 4> ret;
        ret.vec_ = _mm_cmplt_epi32(vec_, _mm_set1_epi32(rhs));
        return ret;
    }

    force_inline simd_vec<int, 4> operator<=(int rhs) const {
        simd_vec<int, 4> ret;
        ret.vec_ = _mm_andnot_si128(_mm_cmpgt_epi32(vec_, _mm_set1_epi32(rhs)), _mm_set_epi32(~0, ~0, ~0, ~0));
        return ret;
    }

    force_inline simd_vec<int, 4> operator>(int rhs) const {
        simd_vec<int, 4> ret;
        ret.vec_ = _mm_cmpgt_epi32(vec_, _mm_set1_epi32(rhs));
        return ret;
    }

    force_inline simd_vec<int, 4> operator>=(int rhs) const {
        simd_vec<int, 4> ret;
        ret.vec_ = _mm_andnot_si128(_mm_cmplt_epi32(vec_, _mm_set1_epi32(rhs)), _mm_set_epi32(~0, ~0, ~0, ~0));
        return ret;
    }

    force_inline simd_vec<int, 4> &operator&=(const int rhs) {
        vec_ = _mm_and_si128(vec_, _mm_set1_epi32(rhs));
        return *this;
    }

    force_inline simd_vec<int, 4> operator~() const {
        simd_vec<int, 4> ret;
        ret.vec_ = _mm_andnot_si128(vec_, _mm_set1_epi32(~0));
        return ret;
    }

    force_inline operator simd_vec<float, 4>() const {
        simd_vec<float, 4> ret;
        ret.vec_ = _mm_cvtepi32_ps(vec_);
        return ret;
    }

    force_inline void copy_to(int *f) const { _mm_storeu_si128((__m128i *)f, vec_); }
    force_inline void copy_to(int *f, simd_mem_aligned_tag) const { _mm_store_si128((__m128i *)f, vec_); }

    force_inline void blend_to(const simd_vec<int, 4> &mask, const simd_vec<int, 4> &v1) {
#if defined(USE_SSE41)
        vec_ = _mm_blendv_epi8(vec_, v1.vec_, mask.vec_);
#else
        __m128i temp1 = _mm_and_si128(mask.vec_, v1.vec_);
        __m128i temp2 = _mm_andnot_si128(mask.vec_, vec_);
        vec_ = _mm_or_si128(temp1, temp2);
#endif
    }

    force_inline void blend_inv_to(const simd_vec<int, 4> &mask, const simd_vec<int, 4> &v1) {
#if defined(USE_SSE41)
        vec_ = _mm_blendv_epi8(v1.vec_, vec_, mask.vec_);
#else
        __m128i temp1 = _mm_andnot_si128(mask.vec_, v1.vec_);
        __m128i temp2 = _mm_and_si128(mask.vec_, vec_);
        vec_ = _mm_or_si128(temp1, temp2);
#endif
    }

    force_inline int movemask() const { return _mm_movemask_ps(_mm_castsi128_ps(vec_)); }

    force_inline bool all_zeros() const {
#if defined(USE_SSE41)
        return _mm_test_all_zeros(vec_, vec_);
#else
        return _mm_movemask_epi8(_mm_cmpeq_epi32(vec_, _mm_setzero_si128())) == 0xFFFF;
#endif
    }

    force_inline bool all_zeros(const simd_vec<int, 4> &mask) const {
#if defined(USE_SSE41)
        return _mm_test_all_zeros(vec_, mask.vec_);
#else
        return _mm_movemask_epi8(_mm_cmpeq_epi32(_mm_and_si128(vec_, mask.vec_), _mm_setzero_si128())) == 0xFFFF;
#endif
    }

    force_inline bool not_all_zeros() const { return !all_zeros(); }

    force_inline static simd_vec<int, 4> min(const simd_vec<int, 4> &v1, const simd_vec<int, 4> &v2) {
        simd_vec<int, 4> temp;
        temp.vec_ = _mm_min_epi32(v1.vec_, v2.vec_);
        return temp;
    }

    force_inline static simd_vec<int, 4> max(const simd_vec<int, 4> &v1, const simd_vec<int, 4> &v2) {
        simd_vec<int, 4> temp;
        temp.vec_ = _mm_max_epi32(v1.vec_, v2.vec_);
        return temp;
    }

    force_inline static simd_vec<int, 4> and_not(const simd_vec<int, 4> &v1, const simd_vec<int, 4> &v2) {
        simd_vec<int, 4> temp;
        temp.vec_ = _mm_andnot_si128(v1.vec_, v2.vec_);
        return temp;
    }

    friend force_inline simd_vec<int, 4> operator&(const simd_vec<int, 4> &v1, const simd_vec<int, 4> &v2) {
        simd_vec<int, 4> temp;
        temp.vec_ = _mm_and_si128(v1.vec_, v2.vec_);
        return temp;
    }

    friend force_inline simd_vec<int, 4> operator|(const simd_vec<int, 4> &v1, const simd_vec<int, 4> &v2) {
        simd_vec<int, 4> temp;
        temp.vec_ = _mm_or_si128(v1.vec_, v2.vec_);
        return temp;
    }

    friend force_inline simd_vec<int, 4> operator^(const simd_vec<int, 4> &v1, const simd_vec<int, 4> &v2) {
        simd_vec<int, 4> temp;
        temp.vec_ = _mm_xor_si128(v1.vec_, v2.vec_);
        ;
        return temp;
    }

    friend force_inline simd_vec<int, 4> operator+(const simd_vec<int, 4> &v1, const simd_vec<int, 4> &v2) {
        simd_vec<int, 4> ret;
        ret.vec_ = _mm_add_epi32(v1.vec_, v2.vec_);
        return ret;
    }

    friend force_inline simd_vec<int, 4> operator-(const simd_vec<int, 4> &v1, const simd_vec<int, 4> &v2) {
        simd_vec<int, 4> ret;
        ret.vec_ = _mm_sub_epi32(v1.vec_, v2.vec_);
        return ret;
    }

    friend force_inline simd_vec<int, 4> operator*(const simd_vec<int, 4> &v1, const simd_vec<int, 4> &v2) {
        simd_vec<int, 4> ret;
#if defined(USE_SSE41)
        ret.vec_ = _mm_mul_epi32(v1.vec_, v2.vec_);
#else
        ITERATE_4({ ret.comp_[i] = v1.comp_[i] * v2.comp_[i]; })
#endif
        return ret;
    }

    friend force_inline simd_vec<int, 4> operator/(const simd_vec<int, 4> &v1, const simd_vec<int, 4> &v2) {
        simd_vec<int, 4> ret;
        ITERATE_4({ ret.comp_[i] = v1.comp_[i] / v2.comp_[i]; })
        return ret;
    }

    friend force_inline simd_vec<int, 4> operator+(const simd_vec<int, 4> &v1, int v2) {
        simd_vec<int, 4> ret;
        ret.vec_ = _mm_add_epi32(v1.vec_, _mm_set1_epi32(v2));
        return ret;
    }

    friend force_inline simd_vec<int, 4> operator-(const simd_vec<int, 4> &v1, int v2) {
        simd_vec<int, 4> ret;
        ret.vec_ = _mm_sub_epi32(v1.vec_, _mm_set1_epi32(v2));
        return ret;
    }

    friend force_inline simd_vec<int, 4> operator*(const simd_vec<int, 4> &v1, int v2) {
        simd_vec<int, 4> ret;
#if defined(USE_SSE41)
        ret.vec_ = _mm_mul_epi32(v1.vec_, _mm_set1_epi32(v2));
#else
        ITERATE_4({ ret.comp_[i] = v1.comp_[i] * v2; })
#endif
        return ret;
    }

    friend force_inline simd_vec<int, 4> operator/(const simd_vec<int, 4> &v1, int v2) {
        simd_vec<int, 4> ret;
        ITERATE_4({ ret.comp_[i] = v1.comp_[i] / v2; })
        return ret;
    }

    friend force_inline simd_vec<int, 4> operator+(int v1, const simd_vec<int, 4> &v2) { return operator+(v2, v1); }

    friend force_inline simd_vec<int, 4> operator-(int v1, const simd_vec<int, 4> &v2) {
        simd_vec<int, 4> ret;
        ret.vec_ = _mm_sub_epi32(_mm_set1_epi32(v1), v2.vec_);
        return ret;
    }

    friend force_inline simd_vec<int, 4> operator*(int v1, const simd_vec<int, 4> &v2) {
        simd_vec<int, 4> ret;
#if defined(USE_SSE41)
        ret.vec_ = _mm_mul_epi32(_mm_set1_epi32(v1), v2.vec_);
#else
        ITERATE_4({ ret.comp_[i] = v1 * v2.comp_[i]; })
#endif
        return ret;
    }

    friend force_inline simd_vec<int, 4> operator/(int v1, const simd_vec<int, 4> &v2) {
        simd_vec<int, 4> ret;
        ITERATE_4({ ret.comp_[i] = v1 / v2.comp_[i]; })
        return ret;
    }

    friend force_inline simd_vec<int, 4> operator>>(const simd_vec<int, 4> &v1, const simd_vec<int, 4> &v2) {
        simd_vec<int, 4> ret;
#if 0
        ret.vec_ = _mm_srlv_epi32(v1.vec_, v2.vec_);
#else
        ITERATE_4({ ret.comp_[i] = reinterpret_cast<const unsigned &>(v1.comp_[i]) >> v2.comp_[i]; })
#endif
        return ret;
    }

    friend force_inline simd_vec<int, 4> operator>>(const simd_vec<int, 4> &v1, int v2) {
        simd_vec<int, 4> ret;
        ret.vec_ = _mm_srli_epi32(v1.vec_, v2);
        return ret;
    }

    friend force_inline simd_vec<int, 4> operator<<(const simd_vec<int, 4> &v1, const simd_vec<int, 4> &v2) {
        simd_vec<int, 4> ret;
#if 0
        ret.vec_ = _mm_sllv_epi32(v1.vec_, v2.vec_);
#else
        ITERATE_4({ ret.comp_[i] = v1.comp_[i] << v2.comp_[i]; })
#endif
        return ret;
    }

    friend force_inline simd_vec<int, 4> operator<<(const simd_vec<int, 4> &v1, int v2) {
        simd_vec<int, 4> ret;
        ret.vec_ = _mm_slli_epi32(v1.vec_, v2);
        return ret;
    }

    friend force_inline bool is_equal(const simd_vec<int, 4> &v1, const simd_vec<int, 4> &v2) {
        __m128i vcmp = _mm_cmpeq_epi32(v1.vec_, v2.vec_);
        return (_mm_movemask_epi8(vcmp) == 0xffff);
    }

    static int size() { return 4; }
    static bool is_native() { return true; }
};

force_inline simd_vec<float, 4>::operator simd_vec<int, 4>() const {
    simd_vec<int, 4> ret;
    ret.vec_ = _mm_cvtps_epi32(vec_);
    return ret;
}

} // namespace NS
} // namespace Ray

#ifdef __GNUC__
#pragma GCC pop_options
#pragma clang attribute pop
#endif