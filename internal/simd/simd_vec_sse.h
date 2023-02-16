// #pragma once

#include <type_traits>

#include <immintrin.h>
#include <smmintrin.h>
#include <xmmintrin.h>

#ifdef __GNUC__
#pragma GCC push_options
#pragma GCC target("sse2")
#pragma clang attribute push(__attribute__((target("sse2"))), apply_to = function)
#endif

#ifndef NDEBUG
#define validate_mask(m) __assert_valid_mask(m)
#else
#define validate_mask(m) ((void)m)
#endif

namespace Ray {
namespace NS {

template <> class simd_vec<int, 4>;

template <> class simd_vec<float, 4> {
    __m128 vec_;

    friend class simd_vec<int, 4>;

  public:
    force_inline simd_vec() = default;
    force_inline simd_vec(const float f) { vec_ = _mm_set1_ps(f); }
    template <typename... Tail> force_inline simd_vec(const float f1, const float f2, const float f3, const float f4) {
        vec_ = _mm_setr_ps(f1, f2, f3, f4);
    }
    force_inline explicit simd_vec(const float *f) { vec_ = _mm_loadu_ps(f); }
    force_inline simd_vec(const float *f, simd_mem_aligned_tag) { vec_ = _mm_load_ps(f); }

    force_inline float operator[](const int i) const {
#if defined(_MSC_VER) && !defined(__clang__)
        return vec_.m128_f32[i];
#else  // _MSC_VER
        alignas(16) float comp[4];
        _mm_store_ps(comp, vec_);
        return comp[i];
#endif // _MSC_VER
    }

    template <int i> force_inline float get() const {
#if defined(_MSC_VER) && !defined(__clang__)
        return vec_.m128_f32[i];
#else
        const int ndx = (i & 3);
        __m128 temp = _mm_shuffle_ps(vec_, vec_, _MM_SHUFFLE(ndx, ndx, ndx, ndx));
        return _mm_cvtss_f32(temp);
#endif
    }
    template <int i> force_inline void set(const float v) {
#if defined(USE_SSE41)
        vec_ = _mm_insert_ps(vec_, _mm_set_ss(v), i << 4);
#elif defined(_MSC_VER) && !defined(__clang__)
        vec_.m128_f32[i] = v;
#else
        static const int maskl[8] = {0, 0, 0, 0, -1, 0, 0, 0};
        __m128 mask = _mm_castsi128_ps(_mm_loadu_si128((const __m128i *)(maskl + 4 - i)));
        __m128 temp1 = _mm_and_ps(mask, _mm_set1_ps(v));
        __m128 temp2 = _mm_andnot_ps(mask, vec_);
        vec_ = _mm_or_ps(temp1, temp2);
#endif
    }
    force_inline void set(const int i, const float v) {
#if defined(_MSC_VER) && !defined(__clang__)
        vec_.m128_f32[i] = v;
#else // _MSC_VER
        __m128 broad = _mm_set1_ps(v);
        static const int maskl[8] = {0, 0, 0, 0, -1, 0, 0, 0};
        __m128 mask = _mm_castsi128_ps(_mm_loadu_si128((const __m128i *)(maskl + 4 - i)));
#if defined(USE_SSE41)
        vec_ = _mm_blendv_ps(vec_, broad, mask);
#else
        __m128 temp1 = _mm_and_ps(mask, broad);
        __m128 temp2 = _mm_andnot_ps(mask, vec_);
        vec_ = _mm_or_ps(temp1, temp2);
#endif
#endif // _MSC_VER
    }

    force_inline simd_vec<float, 4> &vectorcall operator+=(const simd_vec<float, 4> rhs) {
        vec_ = _mm_add_ps(vec_, rhs.vec_);
        return *this;
    }

    force_inline simd_vec<float, 4> &vectorcall operator+=(const float rhs) {
        vec_ = _mm_add_ps(vec_, _mm_set1_ps(rhs));
        return *this;
    }

    force_inline simd_vec<float, 4> &vectorcall operator-=(const simd_vec<float, 4> rhs) {
        vec_ = _mm_sub_ps(vec_, rhs.vec_);
        return *this;
    }

    force_inline simd_vec<float, 4> &vectorcall operator-=(const float rhs) {
        vec_ = _mm_sub_ps(vec_, _mm_set1_ps(rhs));
        return *this;
    }

    force_inline simd_vec<float, 4> &vectorcall operator*=(const simd_vec<float, 4> rhs) {
        vec_ = _mm_mul_ps(vec_, rhs.vec_);
        return *this;
    }

    force_inline simd_vec<float, 4> &vectorcall operator*=(const float rhs) {
        vec_ = _mm_mul_ps(vec_, _mm_set1_ps(rhs));
        return *this;
    }

    force_inline simd_vec<float, 4> &vectorcall operator/=(const simd_vec<float, 4> rhs) {
        vec_ = _mm_div_ps(vec_, rhs.vec_);
        return *this;
    }

    force_inline simd_vec<float, 4> &vectorcall operator/=(const float rhs) {
        vec_ = _mm_div_ps(vec_, _mm_set1_ps(rhs));
        return *this;
    }

    force_inline simd_vec<float, 4> &vectorcall operator|=(const simd_vec<float, 4> rhs) {
        vec_ = _mm_or_ps(vec_, rhs.vec_);
        return *this;
    }

    force_inline simd_vec<float, 4> &vectorcall operator|=(const float rhs) {
        vec_ = _mm_or_ps(vec_, _mm_set1_ps(rhs));
        return *this;
    }

    force_inline simd_vec<float, 4> vectorcall operator-() const {
        simd_vec<float, 4> temp;
        __m128 m = _mm_set1_ps(-0.0f);
        temp.vec_ = _mm_xor_ps(vec_, m);
        return temp;
    }

    force_inline simd_vec<float, 4> vectorcall operator<(const simd_vec<float, 4> rhs) const {
        simd_vec<float, 4> ret;
        ret.vec_ = _mm_cmplt_ps(vec_, rhs.vec_);
        return ret;
    }

    force_inline simd_vec<float, 4> vectorcall operator<=(const simd_vec<float, 4> rhs) const {
        simd_vec<float, 4> ret;
        ret.vec_ = _mm_cmple_ps(vec_, rhs.vec_);
        return ret;
    }

    force_inline simd_vec<float, 4> vectorcall operator>(const simd_vec<float, 4> rhs) const {
        simd_vec<float, 4> ret;
        ret.vec_ = _mm_cmpgt_ps(vec_, rhs.vec_);
        return ret;
    }

    force_inline simd_vec<float, 4> vectorcall operator>=(const simd_vec<float, 4> rhs) const {
        simd_vec<float, 4> ret;
        ret.vec_ = _mm_cmpge_ps(vec_, rhs.vec_);
        return ret;
    }

    force_inline simd_vec<float, 4> vectorcall operator~() const {
        simd_vec<float, 4> ret;
        ret.vec_ = _mm_castsi128_ps(_mm_andnot_si128(_mm_castps_si128(vec_), _mm_set1_epi32(~0)));
        return ret;
    }

    force_inline simd_vec<float, 4> vectorcall operator<(const float rhs) const {
        simd_vec<float, 4> ret;
        ret.vec_ = _mm_cmplt_ps(vec_, _mm_set1_ps(rhs));
        return ret;
    }

    force_inline simd_vec<float, 4> vectorcall operator<=(const float rhs) const {
        simd_vec<float, 4> ret;
        ret.vec_ = _mm_cmple_ps(vec_, _mm_set1_ps(rhs));
        return ret;
    }

    force_inline simd_vec<float, 4> vectorcall operator>(const float rhs) const {
        simd_vec<float, 4> ret;
        ret.vec_ = _mm_cmpgt_ps(vec_, _mm_set1_ps(rhs));
        return ret;
    }

    force_inline simd_vec<float, 4> vectorcall operator>=(const float rhs) const {
        simd_vec<float, 4> ret;
        ret.vec_ = _mm_cmpge_ps(vec_, _mm_set1_ps(rhs));
        return ret;
    }

    force_inline simd_vec<float, 4> &vectorcall operator&=(const simd_vec<float, 4> rhs) {
        vec_ = _mm_and_ps(vec_, rhs.vec_);
        return *this;
    }

    force_inline explicit vectorcall operator simd_vec<int, 4>() const;

    force_inline simd_vec<float, 4> vectorcall sqrt() const {
        simd_vec<float, 4> temp;
        temp.vec_ = _mm_sqrt_ps(vec_);
        return temp;
    }

    force_inline simd_vec<float, 4> vectorcall log() const {
        alignas(16) float comp[4];
        _mm_store_ps(comp, vec_);
        UNROLLED_FOR(i, 4, { comp[i] = std::log(comp[i]); })
        return simd_vec<float, 4>{comp, simd_mem_aligned};
    }

    force_inline float vectorcall length() const {
        __m128 r1, r2;
        r1 = _mm_mul_ps(vec_, vec_);

        r2 = _mm_shuffle_ps(r1, r1, _MM_SHUFFLE(2, 3, 0, 1));
        r1 = _mm_add_ps(r1, r2);
        r2 = _mm_shuffle_ps(r1, r1, _MM_SHUFFLE(0, 1, 2, 3));
        r1 = _mm_add_ps(r1, r2);

        return _mm_cvtss_f32(_mm_sqrt_ss(r1));
    }

    force_inline float vectorcall length2() const {
        __m128 r1, r2;
        r1 = _mm_mul_ps(vec_, vec_);

        r2 = _mm_shuffle_ps(r1, r1, _MM_SHUFFLE(2, 3, 0, 1));
        r1 = _mm_add_ps(r1, r2);
        r2 = _mm_shuffle_ps(r1, r1, _MM_SHUFFLE(0, 1, 2, 3));
        r1 = _mm_add_ps(r1, r2);

        return _mm_cvtss_f32(r1);
    }

    force_inline void vectorcall copy_to(float *f) const { _mm_storeu_ps(f, vec_); }
    force_inline void vectorcall copy_to(float *f, simd_mem_aligned_tag) const { _mm_store_ps(f, vec_); }

    force_inline void vectorcall blend_to(const simd_vec<float, 4> mask, const simd_vec<float, 4> v1) {
        validate_mask(mask);
#if defined(USE_SSE41)
        vec_ = _mm_blendv_ps(vec_, v1.vec_, mask.vec_);
#else
        __m128 temp1 = _mm_and_ps(mask.vec_, v1.vec_);
        __m128 temp2 = _mm_andnot_ps(mask.vec_, vec_);
        vec_ = _mm_or_ps(temp1, temp2);
#endif
    }

    force_inline void vectorcall blend_inv_to(const simd_vec<float, 4> mask, const simd_vec<float, 4> v1) {
        validate_mask(mask);
#if defined(USE_SSE41)
        vec_ = _mm_blendv_ps(v1.vec_, vec_, mask.vec_);
#else
        __m128 temp1 = _mm_andnot_ps(mask.vec_, v1.vec_);
        __m128 temp2 = _mm_and_ps(mask.vec_, vec_);
        vec_ = _mm_or_ps(temp1, temp2);
#endif
    }

    force_inline static simd_vec<float, 4> vectorcall min(const simd_vec<float, 4> v1, const simd_vec<float, 4> v2) {
        simd_vec<float, 4> temp;
        temp.vec_ = _mm_min_ps(v1.vec_, v2.vec_);
        return temp;
    }

    force_inline static simd_vec<float, 4> vectorcall min(const simd_vec<float, 4> v1, const float v2) {
        simd_vec<float, 4> temp;
        temp.vec_ = _mm_min_ps(v1.vec_, _mm_set1_ps(v2));
        return temp;
    }

    force_inline static simd_vec<float, 4> vectorcall min(const float v1, const simd_vec<float, 4> v2) {
        simd_vec<float, 4> temp;
        temp.vec_ = _mm_min_ps(_mm_set1_ps(v1), v2.vec_);
        return temp;
    }

    force_inline static simd_vec<float, 4> vectorcall max(const simd_vec<float, 4> v1, const simd_vec<float, 4> v2) {
        simd_vec<float, 4> temp;
        temp.vec_ = _mm_max_ps(v1.vec_, v2.vec_);
        return temp;
    }

    force_inline static simd_vec<float, 4> vectorcall max(const simd_vec<float, 4> v1, const float v2) {
        simd_vec<float, 4> temp;
        temp.vec_ = _mm_max_ps(v1.vec_, _mm_set1_ps(v2));
        return temp;
    }

    force_inline static simd_vec<float, 4> vectorcall max(const float v1, const simd_vec<float, 4> v2) {
        simd_vec<float, 4> temp;
        temp.vec_ = _mm_max_ps(_mm_set1_ps(v1), v2.vec_);
        return temp;
    }

    force_inline static simd_vec<float, 4> vectorcall and_not(const simd_vec<float, 4> v1,
                                                              const simd_vec<float, 4> v2) {
        simd_vec<float, 4> temp;
        temp.vec_ = _mm_andnot_ps(v1.vec_, v2.vec_);
        return temp;
    }

    force_inline static simd_vec<float, 4> vectorcall floor(const simd_vec<float, 4> v1) {
        simd_vec<float, 4> temp;
#if defined(USE_SSE41)
        temp.vec_ = _mm_floor_ps(v1.vec_);
#else
        __m128 t = _mm_cvtepi32_ps(_mm_cvttps_epi32(v1.vec_));
        temp.vec_ = _mm_sub_ps(t, _mm_and_ps(_mm_cmplt_ps(v1.vec_, t), _mm_set1_ps(1.0f)));
#endif
        return temp;
    }

    force_inline static simd_vec<float, 4> vectorcall ceil(const simd_vec<float, 4> v1) {
        simd_vec<float, 4> temp;
        __m128 t = _mm_cvtepi32_ps(_mm_cvttps_epi32(v1.vec_));
        __m128 r = _mm_add_ps(t, _mm_and_ps(_mm_cmpgt_ps(v1.vec_, t), _mm_set1_ps(1.0f)));
        temp.vec_ = r;
        return temp;
    }

    friend force_inline simd_vec<float, 4> vectorcall operator&(const simd_vec<float, 4> v1,
                                                                const simd_vec<float, 4> v2) {
        simd_vec<float, 4> temp;
        temp.vec_ = _mm_and_ps(v1.vec_, v2.vec_);
        return temp;
    }

    friend force_inline simd_vec<float, 4> vectorcall operator|(const simd_vec<float, 4> v1,
                                                                const simd_vec<float, 4> v2) {
        simd_vec<float, 4> temp;
        temp.vec_ = _mm_or_ps(v1.vec_, v2.vec_);
        return temp;
    }

    friend force_inline simd_vec<float, 4> vectorcall operator^(const simd_vec<float, 4> v1,
                                                                const simd_vec<float, 4> v2) {
        simd_vec<float, 4> temp;
        temp.vec_ = _mm_xor_ps(v1.vec_, v2.vec_);
        return temp;
    }

    friend force_inline simd_vec<float, 4> vectorcall operator+(const simd_vec<float, 4> v1,
                                                                const simd_vec<float, 4> v2) {
        simd_vec<float, 4> ret;
        ret.vec_ = _mm_add_ps(v1.vec_, v2.vec_);
        return ret;
    }

    friend force_inline simd_vec<float, 4> vectorcall operator-(const simd_vec<float, 4> v1,
                                                                const simd_vec<float, 4> v2) {
        simd_vec<float, 4> ret;
        ret.vec_ = _mm_sub_ps(v1.vec_, v2.vec_);
        return ret;
    }

    force_inline simd_vec<float, 4> vectorcall operator==(const float rhs) const {
        simd_vec<float, 4> ret;
        ret.vec_ = _mm_cmpeq_ps(vec_, _mm_set1_ps(rhs));
        return ret;
    }

    force_inline simd_vec<float, 4> vectorcall operator==(const simd_vec<float, 4> rhs) const {
        simd_vec<float, 4> ret;
        ret.vec_ = _mm_cmpeq_ps(vec_, rhs.vec_);
        return ret;
    }

    force_inline simd_vec<float, 4> vectorcall operator!=(const float rhs) const {
        simd_vec<float, 4> ret;
        ret.vec_ = _mm_cmpneq_ps(vec_, _mm_set1_ps(rhs));
        return ret;
    }

    force_inline simd_vec<float, 4> vectorcall operator!=(const simd_vec<float, 4> rhs) const {
        simd_vec<float, 4> ret;
        ret.vec_ = _mm_cmpneq_ps(vec_, rhs.vec_);
        return ret;
    }

    friend force_inline simd_vec<float, 4> vectorcall operator*(const simd_vec<float, 4> v1,
                                                                const simd_vec<float, 4> v2) {
        simd_vec<float, 4> ret;
        ret.vec_ = _mm_mul_ps(v1.vec_, v2.vec_);
        return ret;
    }

    friend force_inline simd_vec<float, 4> vectorcall operator/(const simd_vec<float, 4> v1,
                                                                const simd_vec<float, 4> v2) {
        simd_vec<float, 4> ret;
        ret.vec_ = _mm_div_ps(v1.vec_, v2.vec_);
        return ret;
    }

    friend force_inline simd_vec<float, 4> vectorcall operator+(const simd_vec<float, 4> v1, const float v2) {
        simd_vec<float, 4> ret;
        ret.vec_ = _mm_add_ps(v1.vec_, _mm_set1_ps(v2));
        return ret;
    }

    friend force_inline simd_vec<float, 4> vectorcall operator-(const simd_vec<float, 4> v1, const float v2) {
        simd_vec<float, 4> ret;
        ret.vec_ = _mm_sub_ps(v1.vec_, _mm_set1_ps(v2));
        return ret;
    }

    friend force_inline simd_vec<float, 4> vectorcall operator*(const simd_vec<float, 4> v1, const float v2) {
        simd_vec<float, 4> ret;
        ret.vec_ = _mm_mul_ps(v1.vec_, _mm_set1_ps(v2));
        return ret;
    }

    friend force_inline simd_vec<float, 4> vectorcall operator/(const simd_vec<float, 4> v1, const float v2) {
        simd_vec<float, 4> ret;
        ret.vec_ = _mm_div_ps(v1.vec_, _mm_set1_ps(v2));
        return ret;
    }

    friend force_inline simd_vec<float, 4> vectorcall operator+(const float v1, const simd_vec<float, 4> v2) {
        return operator+(v2, v1);
    }

    friend force_inline simd_vec<float, 4> vectorcall operator-(const float v1, const simd_vec<float, 4> v2) {
        simd_vec<float, 4> ret;
        ret.vec_ = _mm_sub_ps(_mm_set1_ps(v1), v2.vec_);
        return ret;
    }

    friend force_inline simd_vec<float, 4> vectorcall operator*(const float v1, const simd_vec<float, 4> v2) {
        simd_vec<float, 4> ret;
        ret.vec_ = _mm_mul_ps(_mm_set1_ps(v1), v2.vec_);
        return ret;
    }

    friend force_inline simd_vec<float, 4> vectorcall operator/(const float v1, const simd_vec<float, 4> v2) {
        simd_vec<float, 4> ret;
        ret.vec_ = _mm_div_ps(_mm_set1_ps(v1), v2.vec_);
        return ret;
    }

    friend force_inline float vectorcall dot(const simd_vec<float, 4> v1, const simd_vec<float, 4> v2) {
        __m128 r1, r2;
        r1 = _mm_mul_ps(v1.vec_, v2.vec_);
        r2 = _mm_shuffle_ps(r1, r1, _MM_SHUFFLE(2, 3, 0, 1));
        r1 = _mm_add_ps(r1, r2);
        r2 = _mm_shuffle_ps(r1, r1, _MM_SHUFFLE(0, 1, 2, 3));
        r1 = _mm_add_ps(r1, r2);
        return _mm_cvtss_f32(r1);
    }

    friend force_inline simd_vec<float, 4> vectorcall clamp(const simd_vec<float, 4> v1, const float min,
                                                            const float max) {
        simd_vec<float, 4> ret;
        ret.vec_ = _mm_max_ps(_mm_set1_ps(min), _mm_min_ps(v1.vec_, _mm_set1_ps(max)));
        return ret;
    }

    friend force_inline simd_vec<float, 4> vectorcall pow(const simd_vec<float, 4> v1, const simd_vec<float, 4> v2) {
        alignas(16) float comp1[4], comp2[4];
        _mm_store_ps(comp1, v1.vec_);
        _mm_store_ps(comp2, v2.vec_);
        UNROLLED_FOR(i, 4, { comp1[i] = std::pow(comp1[i], comp2[i]); })
        return simd_vec<float, 4>{comp1, simd_mem_aligned};
    }

    friend force_inline simd_vec<float, 4> vectorcall normalize(const simd_vec<float, 4> v1) {
        return v1 / v1.length();
    }

#ifndef NDEBUG
    friend void vectorcall __assert_valid_mask(const simd_vec<float, 4> mask) {
        UNROLLED_FOR(i, 4, {
            const float val = mask.get<i>();
            assert(reinterpret_cast<const uint32_t &>(val) == 0 ||
                   reinterpret_cast<const uint32_t &>(val) == 0xffffffff);
        })
    }
#endif

    friend force_inline const float *vectorcall value_ptr(const simd_vec<float, 4> &v1) {
        return reinterpret_cast<const float *>(&v1.vec_);
    }
    friend force_inline float *vectorcall value_ptr(simd_vec<float, 4> &v1) {
        return reinterpret_cast<float *>(&v1.vec_);
    }

    static int size() { return 4; }
    static bool is_native() { return true; }
};

template <> class simd_vec<int, 4> {
    __m128i vec_;

    friend class simd_vec<float, 4>;

  public:
    force_inline simd_vec() = default;
    force_inline simd_vec(const int v) { vec_ = _mm_set1_epi32(v); }
    force_inline simd_vec(const int i1, const int i2, const int i3, const int i4) {
        vec_ = _mm_setr_epi32(i1, i2, i3, i4);
    }
    force_inline explicit simd_vec(const int *f) { vec_ = _mm_loadu_si128((const __m128i *)f); }
    force_inline simd_vec(const int *f, simd_mem_aligned_tag) { vec_ = _mm_load_si128((const __m128i *)f); }

    force_inline int operator[](const int i) const {
#if defined(_MSC_VER) && !defined(__clang__)
        return vec_.m128i_i32[i];
#else  // _MSC_VER
        alignas(16) int comp[4];
        _mm_store_si128((__m128i *)comp, vec_);
        return comp[i];
#endif // _MSC_VER
    }

    template <int i> force_inline int get() const {
#if defined(USE_SSE41)
        return _mm_extract_epi32(vec_, i & 3);
#elif defined(_MSC_VER) && !defined(__clang__)
        return vec_.m128i_i32[i];
#else
        alignas(16) int comp[4];
        _mm_store_si128((__m128i *)comp, vec_);
        return comp[i & 3];
#endif
    }
    template <int i> force_inline void set(const int v) {
        const int ndx = (i & 3);
#if defined(USE_SSE41)
        vec_ = _mm_insert_epi32(vec_, v, ndx);
#elif defined(_MSC_VER) && !defined(__clang__)
        vec_.m128i_i32[i] = v;
#else
        static const int maskl[8] = {0, 0, 0, 0, -1, 0, 0, 0};
        __m128i mask = _mm_loadu_si128((const __m128i *)(maskl + 4 - ndx));
        __m128i temp1 = _mm_and_si128(mask, _mm_set1_epi32(v));
        __m128i temp2 = _mm_andnot_si128(mask, vec_);
        vec_ = _mm_or_si128(temp1, temp2);
#endif
    }
    force_inline void set(const int i, const int v) {
        static const int maskl[8] = {0, 0, 0, 0, -1, 0, 0, 0};
        __m128i mask = _mm_loadu_si128((const __m128i *)(maskl + 4 - i));
#if defined(USE_SSE41)
        vec_ = _mm_blendv_epi8(vec_, _mm_set1_epi32(v), mask);
#elif defined(_MSC_VER) && !defined(__clang__)
        vec_.m128i_i32[i] = v;
#else
        __m128i temp1 = _mm_and_si128(mask, _mm_set1_epi32(v));
        __m128i temp2 = _mm_andnot_si128(mask, vec_);
        vec_ = _mm_or_si128(temp1, temp2);
#endif
    }

    force_inline simd_vec<int, 4> &vectorcall operator+=(const simd_vec<int, 4> rhs) {
        vec_ = _mm_add_epi32(vec_, rhs.vec_);
        return *this;
    }

    force_inline simd_vec<int, 4> &vectorcall operator+=(const int rhs) {
        vec_ = _mm_add_epi32(vec_, _mm_set1_epi32(rhs));
        return *this;
    }

    force_inline simd_vec<int, 4> &vectorcall operator-=(const simd_vec<int, 4> rhs) {
        vec_ = _mm_sub_epi32(vec_, rhs.vec_);
        return *this;
    }

    force_inline simd_vec<int, 4> &operator-=(const int rhs) {
        vec_ = _mm_sub_epi32(vec_, _mm_set1_epi32(rhs));
        return *this;
    }

    force_inline simd_vec<int, 4> &vectorcall operator*=(const simd_vec<int, 4> rhs) {
#if defined(USE_SSE41)
        vec_ = _mm_mullo_epi32(vec_, rhs.vec_);
#elif defined(_MSC_VER) && !defined(__clang__)
        UNROLLED_FOR(i, 4, { vec_.m128i_i32[i] *= rhs.vec_.m128i_i32[i]; })
#else
        alignas(16) int comp[4], comp_rhs[4];
        _mm_store_si128((__m128i *)comp, vec_);
        _mm_store_si128((__m128i *)comp_rhs, rhs.vec_);
        UNROLLED_FOR(i, 4, { comp[i] = comp[i] * comp_rhs[i]; })
        vec_ = _mm_load_si128((const __m128i *)comp);
#endif
        return *this;
    }

    force_inline simd_vec<int, 4> &vectorcall operator*=(const int rhs) {
#if defined(USE_SSE41)
        vec_ = _mm_mullo_epi32(vec_, _mm_set1_epi32(rhs));
#elif defined(_MSC_VER) && !defined(__clang__)
        UNROLLED_FOR(i, 4, { vec_.m128i_i32[i] *= rhs; })
#else
        alignas(16) int comp[4];
        _mm_store_si128((__m128i *)comp, vec_);
        UNROLLED_FOR(i, 4, { comp[i] *= rhs; })
        vec_ = _mm_load_si128((const __m128i *)comp);
#endif
        return *this;
    }

    force_inline simd_vec<int, 4> &vectorcall operator/=(const simd_vec<int, 4> rhs) {
#if defined(_MSC_VER) && !defined(__clang__)
        UNROLLED_FOR(i, 4, { vec_.m128i_i32[i] /= rhs.vec_.m128i_i32[i]; })
#else
        alignas(16) int comp[4], comp_rhs[4];
        _mm_store_si128((__m128i *)comp, vec_);
        _mm_store_si128((__m128i *)comp_rhs, rhs.vec_);
        UNROLLED_FOR(i, 4, { comp[i] /= comp_rhs[i]; })
        vec_ = _mm_load_si128((const __m128i *)comp);
#endif
        return *this;
    }

    force_inline simd_vec<int, 4> &vectorcall operator/=(const int rhs) {
#if defined(_MSC_VER) && !defined(__clang__)
        UNROLLED_FOR(i, 4, { vec_.m128i_i32[i] /= rhs; })
#else
        alignas(16) int comp[4];
        _mm_store_si128((__m128i *)comp, vec_);
        UNROLLED_FOR(i, 4, { comp[i] /= rhs; })
        vec_ = _mm_load_si128((const __m128i *)comp);
#endif
        return *this;
    }

    force_inline simd_vec<int, 4> &vectorcall operator|=(const simd_vec<int, 4> rhs) {
        vec_ = _mm_or_si128(vec_, rhs.vec_);
        return *this;
    }

    force_inline simd_vec<int, 4> &vectorcall operator|=(const int rhs) {
        vec_ = _mm_or_si128(vec_, _mm_set1_epi32(rhs));
        return *this;
    }

    force_inline simd_vec<int, 4> vectorcall operator-() const {
        simd_vec<int, 4> temp;
        temp.vec_ = _mm_sub_epi32(_mm_setzero_si128(), vec_);
        return temp;
    }

    force_inline simd_vec<int, 4> vectorcall operator==(const int rhs) const {
        simd_vec<int, 4> ret;
        ret.vec_ = _mm_cmpeq_epi32(vec_, _mm_set1_epi32(rhs));
        return ret;
    }

    force_inline simd_vec<int, 4> vectorcall operator==(const simd_vec<int, 4> rhs) const {
        simd_vec<int, 4> ret;
        ret.vec_ = _mm_cmpeq_epi32(vec_, rhs.vec_);
        return ret;
    }

    force_inline simd_vec<int, 4> vectorcall operator!=(const int rhs) const {
        simd_vec<int, 4> ret;
        ret.vec_ = _mm_andnot_si128(_mm_cmpeq_epi32(vec_, _mm_set1_epi32(rhs)), _mm_set1_epi32(~0));
        return ret;
    }

    force_inline simd_vec<int, 4> vectorcall operator!=(const simd_vec<int, 4> rhs) const {
        simd_vec<int, 4> ret;
        ret.vec_ = _mm_andnot_si128(_mm_cmpeq_epi32(vec_, rhs.vec_), _mm_set1_epi32(~0));
        return ret;
    }

    force_inline simd_vec<int, 4> vectorcall operator<(const simd_vec<int, 4> rhs) const {
        simd_vec<int, 4> ret;
        ret.vec_ = _mm_cmplt_epi32(vec_, rhs.vec_);
        return ret;
    }

    force_inline simd_vec<int, 4> vectorcall operator<=(const simd_vec<int, 4> rhs) const {
        simd_vec<int, 4> ret;
        ret.vec_ = _mm_andnot_si128(_mm_cmpgt_epi32(vec_, rhs.vec_), _mm_set_epi32(~0, ~0, ~0, ~0));
        return ret;
    }

    force_inline simd_vec<int, 4> vectorcall operator>(const simd_vec<int, 4> rhs) const {
        simd_vec<int, 4> ret;
        ret.vec_ = _mm_cmpgt_epi32(vec_, rhs.vec_);
        return ret;
    }

    force_inline simd_vec<int, 4> vectorcall operator>=(const simd_vec<int, 4> rhs) const {
        simd_vec<int, 4> ret;
        ret.vec_ = _mm_andnot_si128(_mm_cmplt_epi32(vec_, rhs.vec_), _mm_set_epi32(~0, ~0, ~0, ~0));
        return ret;
    }

    force_inline simd_vec<int, 4> &vectorcall operator&=(const simd_vec<int, 4> rhs) {
        vec_ = _mm_and_si128(vec_, rhs.vec_);
        return *this;
    }

    force_inline simd_vec<int, 4> vectorcall operator<(const int rhs) const {
        simd_vec<int, 4> ret;
        ret.vec_ = _mm_cmplt_epi32(vec_, _mm_set1_epi32(rhs));
        return ret;
    }

    force_inline simd_vec<int, 4> vectorcall operator<=(const int rhs) const {
        simd_vec<int, 4> ret;
        ret.vec_ = _mm_andnot_si128(_mm_cmpgt_epi32(vec_, _mm_set1_epi32(rhs)), _mm_set_epi32(~0, ~0, ~0, ~0));
        return ret;
    }

    force_inline simd_vec<int, 4> vectorcall operator>(const int rhs) const {
        simd_vec<int, 4> ret;
        ret.vec_ = _mm_cmpgt_epi32(vec_, _mm_set1_epi32(rhs));
        return ret;
    }

    force_inline simd_vec<int, 4> vectorcall operator>=(const int rhs) const {
        simd_vec<int, 4> ret;
        ret.vec_ = _mm_andnot_si128(_mm_cmplt_epi32(vec_, _mm_set1_epi32(rhs)), _mm_set_epi32(~0, ~0, ~0, ~0));
        return ret;
    }

    force_inline simd_vec<int, 4> &vectorcall operator&=(const int rhs) {
        vec_ = _mm_and_si128(vec_, _mm_set1_epi32(rhs));
        return *this;
    }

    force_inline simd_vec<int, 4> vectorcall operator~() const {
        simd_vec<int, 4> ret;
        ret.vec_ = _mm_andnot_si128(vec_, _mm_set1_epi32(~0));
        return ret;
    }

    force_inline explicit vectorcall operator simd_vec<float, 4>() const {
        simd_vec<float, 4> ret;
        ret.vec_ = _mm_cvtepi32_ps(vec_);
        return ret;
    }

    force_inline void copy_to(int *f) const { _mm_storeu_si128((__m128i *)f, vec_); }
    force_inline void copy_to(int *f, simd_mem_aligned_tag) const { _mm_store_si128((__m128i *)f, vec_); }

    force_inline void vectorcall blend_to(const simd_vec<int, 4> mask, const simd_vec<int, 4> v1) {
        validate_mask(mask);
#if defined(USE_SSE41)
        vec_ = _mm_blendv_epi8(vec_, v1.vec_, mask.vec_);
#else
        __m128i temp1 = _mm_and_si128(mask.vec_, v1.vec_);
        __m128i temp2 = _mm_andnot_si128(mask.vec_, vec_);
        vec_ = _mm_or_si128(temp1, temp2);
#endif
    }

    force_inline void vectorcall blend_inv_to(const simd_vec<int, 4> mask, const simd_vec<int, 4> v1) {
        validate_mask(mask);
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

    force_inline bool vectorcall all_zeros(const simd_vec<int, 4> mask) const {
#if defined(USE_SSE41)
        return _mm_test_all_zeros(vec_, mask.vec_);
#else
        return _mm_movemask_epi8(_mm_cmpeq_epi32(_mm_and_si128(vec_, mask.vec_), _mm_setzero_si128())) == 0xFFFF;
#endif
    }

    force_inline bool not_all_zeros() const { return !all_zeros(); }

    force_inline static simd_vec<int, 4> vectorcall min(const simd_vec<int, 4> v1, const simd_vec<int, 4> v2) {
        simd_vec<int, 4> temp;
#if defined(USE_SSE41)
        temp.vec_ = _mm_min_epi32(v1.vec_, v2.vec_);
#elif defined(_MSC_VER) && !defined(__clang__)
        UNROLLED_FOR(i, 4, {
            temp.vec_.m128i_i32[i] =
                (v1.vec_.m128i_i32[i] < v2.vec_.m128i_i32[i]) ? v1.vec_.m128i_i32[i] : v2.vec_.m128i_i32[i];
        })
#else
        alignas(16) int comp1[4], comp2[4];
        _mm_store_si128((__m128i *)comp1, v1.vec_);
        _mm_store_si128((__m128i *)comp2, v2.vec_);
        UNROLLED_FOR(i, 4, { comp1[i] = (comp1[i] < comp2[i]) ? comp1[i] : comp2[i]; })
        temp.vec_ = _mm_load_si128((const __m128i *)comp1);
#endif
        return temp;
    }

    force_inline static simd_vec<int, 4> vectorcall min(const simd_vec<int, 4> v1, const int v2) {
        simd_vec<int, 4> temp;
#if defined(USE_SSE41)
        temp.vec_ = _mm_min_epi32(v1.vec_, _mm_set1_epi32(v2));
#elif defined(_MSC_VER) && !defined(__clang__)
        UNROLLED_FOR(i, 4, { temp.vec_.m128i_i32[i] = (v1.vec_.m128i_i32[i] < v2) ? v1.vec_.m128i_i32[i] : v2; })
#else
        alignas(16) int comp[4];
        _mm_store_si128((__m128i *)comp, v1.vec_);
        UNROLLED_FOR(i, 4, { comp[i] = (comp[i] < v2) ? comp[i] : v2; })
        temp.vec_ = _mm_load_si128((const __m128i *)comp);
#endif
        return temp;
    }

    force_inline static simd_vec<int, 4> vectorcall min(const int v1, const simd_vec<int, 4> v2) {
        simd_vec<int, 4> temp;
#if defined(USE_SSE41)
        temp.vec_ = _mm_min_epi32(_mm_set1_epi32(v1), v2.vec_);
#elif defined(_MSC_VER) && !defined(__clang__)
        UNROLLED_FOR(i, 4, { temp.vec_.m128i_i32[i] = (v1 < v2.vec_.m128i_i32[i]) ? v1 : v2.vec_.m128i_i32[i]; })
#else
        alignas(16) int comp[4];
        _mm_store_si128((__m128i *)comp, v2.vec_);
        UNROLLED_FOR(i, 4, { comp[i] = (comp[i] < v1) ? v1 : comp[i]; })
        temp.vec_ = _mm_load_si128((const __m128i *)comp);
#endif
        return temp;
    }

    force_inline static simd_vec<int, 4> vectorcall max(const simd_vec<int, 4> v1, const simd_vec<int, 4> v2) {
        simd_vec<int, 4> temp;
#if defined(USE_SSE41)
        temp.vec_ = _mm_max_epi32(v1.vec_, v2.vec_);
#elif defined(_MSC_VER) && !defined(__clang__)
        UNROLLED_FOR(i, 4, {
            temp.vec_.m128i_i32[i] =
                (v1.vec_.m128i_i32[i] < v2.vec_.m128i_i32[i]) ? v1.vec_.m128i_i32[i] : v2.vec_.m128i_i32[i];
        })
#else
        alignas(16) int comp1[4], comp2[4];
        _mm_store_si128((__m128i *)comp1, v1.vec_);
        _mm_store_si128((__m128i *)comp2, v2.vec_);
        UNROLLED_FOR(i, 4, { comp1[i] = (comp1[i] > comp2[i]) ? comp1[i] : comp2[i]; })
        temp.vec_ = _mm_load_si128((const __m128i *)comp1);
#endif
        return temp;
    }

    force_inline static simd_vec<int, 4> vectorcall max(const simd_vec<int, 4> v1, const int v2) {
        simd_vec<int, 4> temp;
        temp.vec_ = _mm_max_epi32(v1.vec_, _mm_set1_epi32(v2));
        return temp;
    }

    force_inline static simd_vec<int, 4> vectorcall max(const int v1, const simd_vec<int, 4> v2) {
        simd_vec<int, 4> temp;
        temp.vec_ = _mm_max_epi32(_mm_set1_epi32(v1), v2.vec_);
        return temp;
    }

    force_inline static simd_vec<int, 4> vectorcall and_not(const simd_vec<int, 4> v1, const simd_vec<int, 4> v2) {
        simd_vec<int, 4> temp;
        temp.vec_ = _mm_andnot_si128(v1.vec_, v2.vec_);
        return temp;
    }

    friend force_inline simd_vec<int, 4> vectorcall operator&(const simd_vec<int, 4> v1, const simd_vec<int, 4> v2) {
        simd_vec<int, 4> temp;
        temp.vec_ = _mm_and_si128(v1.vec_, v2.vec_);
        return temp;
    }

    friend force_inline simd_vec<int, 4> vectorcall operator|(const simd_vec<int, 4> v1, const simd_vec<int, 4> v2) {
        simd_vec<int, 4> temp;
        temp.vec_ = _mm_or_si128(v1.vec_, v2.vec_);
        return temp;
    }

    friend force_inline simd_vec<int, 4> vectorcall operator^(const simd_vec<int, 4> v1, const simd_vec<int, 4> v2) {
        simd_vec<int, 4> temp;
        temp.vec_ = _mm_xor_si128(v1.vec_, v2.vec_);
        ;
        return temp;
    }

    friend force_inline simd_vec<int, 4> vectorcall operator+(const simd_vec<int, 4> v1, const simd_vec<int, 4> v2) {
        simd_vec<int, 4> ret;
        ret.vec_ = _mm_add_epi32(v1.vec_, v2.vec_);
        return ret;
    }

    friend force_inline simd_vec<int, 4> vectorcall operator-(const simd_vec<int, 4> v1, const simd_vec<int, 4> v2) {
        simd_vec<int, 4> ret;
        ret.vec_ = _mm_sub_epi32(v1.vec_, v2.vec_);
        return ret;
    }

    friend force_inline simd_vec<int, 4> vectorcall operator*(const simd_vec<int, 4> v1, const simd_vec<int, 4> v2) {
        simd_vec<int, 4> ret;
#if defined(USE_SSE41)
        ret.vec_ = _mm_mullo_epi32(v1.vec_, v2.vec_);
#elif defined(_MSC_VER) && !defined(__clang__)
        UNROLLED_FOR(i, 4, { ret.vec_.m128i_i32[i] = v1.vec_.m128i_i32[i] * v2.vec_.m128i_i32[i]; })
#else
        alignas(16) int comp1[4], comp2[4];
        _mm_store_si128((__m128i *)comp1, v1.vec_);
        _mm_store_si128((__m128i *)comp2, v2.vec_);
        UNROLLED_FOR(i, 4, { comp1[i] *= comp2[i]; })
        ret.vec_ = _mm_load_si128((const __m128i *)comp1);
#endif
        return ret;
    }

    friend force_inline simd_vec<int, 4> vectorcall operator/(const simd_vec<int, 4> v1, const simd_vec<int, 4> v2) {
        simd_vec<int, 4> ret;
#if defined(_MSC_VER) && !defined(__clang__)
        UNROLLED_FOR(i, 4, { ret.vec_.m128i_i32[i] = v1.vec_.m128i_i32[i] / v2.vec_.m128i_i32[i]; })
#else
        alignas(16) int comp1[4], comp2[4];
        _mm_store_si128((__m128i *)comp1, v1.vec_);
        _mm_store_si128((__m128i *)comp2, v2.vec_);
        UNROLLED_FOR(i, 4, { comp1[i] /= comp2[i]; })
        ret.vec_ = _mm_load_si128((const __m128i *)comp1);
#endif
        return ret;
    }

    friend force_inline simd_vec<int, 4> vectorcall operator+(const simd_vec<int, 4> v1, const int v2) {
        simd_vec<int, 4> ret;
        ret.vec_ = _mm_add_epi32(v1.vec_, _mm_set1_epi32(v2));
        return ret;
    }

    friend force_inline simd_vec<int, 4> vectorcall operator-(const simd_vec<int, 4> v1, const int v2) {
        simd_vec<int, 4> ret;
        ret.vec_ = _mm_sub_epi32(v1.vec_, _mm_set1_epi32(v2));
        return ret;
    }

    friend force_inline simd_vec<int, 4> vectorcall operator*(const simd_vec<int, 4> v1, const int v2) {
        simd_vec<int, 4> ret;
#if defined(USE_SSE41)
        ret.vec_ = _mm_mullo_epi32(v1.vec_, _mm_set1_epi32(v2));
#elif defined(_MSC_VER) && !defined(__clang__)
        UNROLLED_FOR(i, 4, { ret.vec_.m128i_i32[i] = v1.vec_.m128i_i32[i] * v2; })
#else
        alignas(16) int comp[4];
        _mm_store_si128((__m128i *)comp, v1.vec_);
        UNROLLED_FOR(i, 4, { comp[i] *= v2; })
        ret.vec_ = _mm_load_si128((const __m128i *)comp);
#endif
        return ret;
    }

    friend force_inline simd_vec<int, 4> vectorcall operator/(const simd_vec<int, 4> v1, const int v2) {
        simd_vec<int, 4> ret;
#if defined(_MSC_VER) && !defined(__clang__)
        UNROLLED_FOR(i, 4, { ret.vec_.m128i_i32[i] = v1.vec_.m128i_i32[i] / v2; })
#else
        alignas(16) int comp[4];
        _mm_store_si128((__m128i *)comp, v1.vec_);
        UNROLLED_FOR(i, 4, { comp[i] /= v2; })
        ret.vec_ = _mm_load_si128((const __m128i *)comp);
#endif
        return ret;
    }

    friend force_inline simd_vec<int, 4> vectorcall operator+(const int v1, const simd_vec<int, 4> v2) {
        return operator+(v2, v1);
    }

    friend force_inline simd_vec<int, 4> vectorcall operator-(const int v1, const simd_vec<int, 4> v2) {
        simd_vec<int, 4> ret;
        ret.vec_ = _mm_sub_epi32(_mm_set1_epi32(v1), v2.vec_);
        return ret;
    }

    friend force_inline simd_vec<int, 4> vectorcall operator*(const int v1, const simd_vec<int, 4> v2) {
        simd_vec<int, 4> ret;
#if defined(USE_SSE41)
        ret.vec_ = _mm_mullo_epi32(_mm_set1_epi32(v1), v2.vec_);
#elif defined(_MSC_VER) && !defined(__clang__)
        UNROLLED_FOR(i, 4, { ret.vec_.m128i_i32[i] = v1 * v2.vec_.m128i_i32[i]; })
#else
        alignas(16) int comp[4];
        _mm_store_si128((__m128i *)comp, v2.vec_);
        UNROLLED_FOR(i, 4, { comp[i] *= v1; })
        ret.vec_ = _mm_load_si128((const __m128i *)comp);
#endif
        return ret;
    }

    friend force_inline simd_vec<int, 4> vectorcall operator/(const int v1, const simd_vec<int, 4> v2) {
        simd_vec<int, 4> ret;
#if defined(_MSC_VER) && !defined(__clang__)
        UNROLLED_FOR(i, 4, { ret.vec_.m128i_i32[i] = v1 / v2.vec_.m128i_i32[i]; })
#else
        alignas(16) int comp[4];
        _mm_store_si128((__m128i *)comp, v2.vec_);
        UNROLLED_FOR(i, 4, { comp[i] = v1 / comp[i]; })
        ret.vec_ = _mm_load_si128((const __m128i *)comp);
#endif
        return ret;
    }

    friend force_inline simd_vec<int, 4> vectorcall operator>>(const simd_vec<int, 4> v1, const simd_vec<int, 4> v2) {
        simd_vec<int, 4> ret;
#if defined(_MSC_VER) && !defined(__clang__)
        UNROLLED_FOR(i, 4, { ret.vec_.m128i_i32[i] = v1.vec_.m128i_i32[i] >> v2.vec_.m128i_i32[i]; })
#else
        alignas(16) int comp1[4], comp2[4];
        _mm_store_si128((__m128i *)comp1, v1.vec_);
        _mm_store_si128((__m128i *)comp2, v2.vec_);
        UNROLLED_FOR(i, 4, { comp1[i] = reinterpret_cast<const unsigned &>(comp1[i]) >> comp2[i]; })
        ret.vec_ = _mm_load_si128((const __m128i *)comp1);
#endif
        return ret;
    }

    friend force_inline simd_vec<int, 4> vectorcall operator>>(const simd_vec<int, 4> v1, const int v2) {
        simd_vec<int, 4> ret;
        ret.vec_ = _mm_srli_epi32(v1.vec_, v2);
        return ret;
    }

    friend force_inline simd_vec<int, 4> vectorcall operator<<(const simd_vec<int, 4> v1, const simd_vec<int, 4> v2) {
        simd_vec<int, 4> ret;
#if defined(_MSC_VER) && !defined(__clang__)
        UNROLLED_FOR(i, 4, { ret.vec_.m128i_i32[i] = v1.vec_.m128i_i32[i] << v2.vec_.m128i_i32[i]; })
#else
        alignas(16) int comp1[4], comp2[4];
        _mm_store_si128((__m128i *)comp1, v1.vec_);
        _mm_store_si128((__m128i *)comp2, v2.vec_);
        UNROLLED_FOR(i, 4, { comp1[i] <<= comp2[i]; })
        ret.vec_ = _mm_load_si128((const __m128i *)comp1);
#endif
        return ret;
    }

    friend force_inline simd_vec<int, 4> vectorcall operator<<(const simd_vec<int, 4> v1, const int v2) {
        simd_vec<int, 4> ret;
        ret.vec_ = _mm_slli_epi32(v1.vec_, v2);
        return ret;
    }

    friend force_inline simd_vec<int, 4> vectorcall srai(const simd_vec<int, 4> v1, const int v2) {
        simd_vec<int, 4> ret;
        ret.vec_ = _mm_srai_epi32(v1.vec_, v2);
        return ret;
    }

    friend force_inline bool vectorcall is_equal(const simd_vec<int, 4> v1, const simd_vec<int, 4> v2) {
        __m128i vcmp = _mm_cmpeq_epi32(v1.vec_, v2.vec_);
        return (_mm_movemask_epi8(vcmp) == 0xffff);
    }

#ifndef NDEBUG
    friend void vectorcall __assert_valid_mask(const simd_vec<int, 4> mask) {
        UNROLLED_FOR(i, 4, {
            const int val = mask.get<i>();
            assert(val == 0 || val == -1);
        })
    }
#endif

    friend force_inline const int *value_ptr(const simd_vec<int, 4> &v1) {
        return reinterpret_cast<const int *>(&v1.vec_);
    }
    friend force_inline int *value_ptr(simd_vec<int, 4> &v1) { return reinterpret_cast<int *>(&v1.vec_); }

    static int size() { return 4; }
    static bool is_native() { return true; }
};

force_inline vectorcall simd_vec<float, 4>::operator simd_vec<int, 4>() const {
    simd_vec<int, 4> ret;
    ret.vec_ = _mm_cvttps_epi32(vec_);
    return ret;
}

} // namespace NS
} // namespace Ray

#undef validate_mask

#ifdef __GNUC__
#pragma GCC pop_options
#pragma clang attribute pop
#endif