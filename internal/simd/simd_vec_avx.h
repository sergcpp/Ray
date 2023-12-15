// #pragma once

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

#if defined(__GNUC__) || defined(__clang__)
#define _mm256_test_all_zeros(mask, val) _mm256_testz_si256((mask), (val))
#endif

#ifndef NDEBUG
#define validate_mask(m) __assert_valid_mask(m)
#else
#define validate_mask(m) ((void)m)
#endif

#if defined(USE_AVX2) || defined(USE_AVX512)
#define USE_FMA
#endif

#pragma warning(push)
#pragma warning(disable : 4752)

namespace Ray {
namespace NS {

template <> force_inline __m256 _mm_cast(__m256i x) { return _mm256_castsi256_ps(x); }
template <> force_inline __m256i _mm_cast(__m256 x) { return _mm256_castps_si256(x); }

template <> class simd_vec<int, 8>;
template <> class simd_vec<unsigned, 8>;

template <> class simd_vec<float, 8> {
    __m256 vec_;

    friend class simd_vec<int, 8>;
    friend class simd_vec<unsigned, 8>;

  public:
    force_inline simd_vec() = default;
    force_inline simd_vec(const float f) { vec_ = _mm256_set1_ps(f); }
    force_inline simd_vec(const float f1, const float f2, const float f3, const float f4, const float f5,
                          const float f6, const float f7, const float f8) {
        vec_ = _mm256_setr_ps(f1, f2, f3, f4, f5, f6, f7, f8);
    }
    force_inline explicit simd_vec(const float *f) { vec_ = _mm256_loadu_ps(f); }
    force_inline simd_vec(const float *f, simd_mem_aligned_tag) { vec_ = _mm256_load_ps(f); }

    force_inline float operator[](const int i) const {
#if defined(_MSC_VER) && !defined(__clang__)
        return vec_.m256_f32[i];
#else
        alignas(32) float comp[8];
        _mm256_store_ps(comp, vec_);
        return comp[i];
#endif
    }

    force_inline float operator[](const long i) const { return operator[](int(i)); }

    template <int _i> force_inline float get() const {
#if defined(_MSC_VER) && !defined(__clang__)
        return vec_.m256_f32[_i & 7];
#else
        __m128 temp = _mm256_extractf128_ps(vec_, (_i & 7) / 4);
        const int ndx = (_i & 7) % 4;
        return _mm_cvtss_f32(_mm_shuffle_ps(temp, temp, _MM_SHUFFLE(ndx, ndx, ndx, ndx)));
#endif
    }
    template <int i> force_inline void set(const float v) {
#if defined(_MSC_VER) && !defined(__clang__)
        vec_.m256_f32[i & 7] = v;
#else
        __m256 temp = _mm256_set1_ps(v);
        vec_ = _mm256_blend_ps(vec_, temp, 1u << (i & 7));
#endif
    }
    force_inline void set(const int i, const float v) {
#if defined(_MSC_VER) && !defined(__clang__)
        vec_.m256_f32[i] = v;
#else
        __m256 broad = _mm256_set1_ps(v);
        static const int maskl[16] = {0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0};
        __m256 mask = _mm256_castsi256_ps(_mm256_loadu_si256((const __m256i *)(maskl + 8 - i)));
        vec_ = _mm256_blendv_ps(vec_, broad, mask);
#endif
    }

    force_inline simd_vec<float, 8> &vectorcall operator+=(const simd_vec<float, 8> rhs) {
        vec_ = _mm256_add_ps(vec_, rhs.vec_);
        return *this;
    }

    force_inline simd_vec<float, 8> &vectorcall operator-=(const simd_vec<float, 8> rhs) {
        vec_ = _mm256_sub_ps(vec_, rhs.vec_);
        return *this;
    }

    force_inline simd_vec<float, 8> &vectorcall operator*=(const simd_vec<float, 8> rhs) {
        vec_ = _mm256_mul_ps(vec_, rhs.vec_);
        return *this;
    }

    force_inline simd_vec<float, 8> &vectorcall operator/=(const simd_vec<float, 8> rhs) {
        vec_ = _mm256_div_ps(vec_, rhs.vec_);
        return *this;
    }

    force_inline simd_vec<float, 8> &vectorcall operator|=(const simd_vec<float, 8> rhs) {
        vec_ = _mm256_or_ps(vec_, rhs.vec_);
        return *this;
    }

    force_inline simd_vec<float, 8> &vectorcall operator&=(const simd_vec<float, 8> rhs) {
        vec_ = _mm256_and_ps(vec_, rhs.vec_);
        return *this;
    }

    force_inline simd_vec<float, 8> operator~() const;
    force_inline simd_vec<float, 8> operator-() const;
    force_inline explicit vectorcall operator simd_vec<int, 8>() const;
    force_inline explicit vectorcall operator simd_vec<unsigned, 8>() const;

    force_inline simd_vec<float, 8> sqrt() const;
    force_inline simd_vec<float, 8> log() const;

    force_inline float length() const { return sqrtf(length2()); }

    force_inline float length2() const {
        float ret = 0;
#if defined(_MSC_VER) && !defined(__clang__)
        UNROLLED_FOR(i, 8, { ret += vec_.m256_f32[i] * vec_.m256_f32[i]; })
#else
        alignas(32) float comp[8];
        _mm256_store_ps(comp, vec_);

        UNROLLED_FOR(i, 8, { ret += comp[i] * comp[i]; })
#endif
        return ret;
    }

    force_inline float hsum() const {
#if 1
        __m256 temp = _mm256_hadd_ps(vec_, vec_);
        temp = _mm256_hadd_ps(temp, temp);

        __m256 ret = _mm256_permute2f128_ps(temp, temp, 1);
        ret = _mm256_add_ps(ret, temp);

        return _mm256_cvtss_f32(ret);
#else
        // ( x3+x7, x2+x6, x1+x5, x0+x4 )
        const __m128 x128 = _mm_add_ps(_mm256_extractf128_ps(vec_, 1), _mm256_castps256_ps128(vec_));
        // ( -, -, x1+x3+x5+x7, x0+x2+x4+x6 )
        const __m128 x64 = _mm_add_ps(x128, _mm_movehl_ps(x128, x128));
        // ( -, -, -, x0+x1+x2+x3+x4+x5+x6+x7 )
        const __m128 x32 = _mm_add_ss(x64, _mm_shuffle_ps(x64, x64, 0x55));
        // Conversion to float is a no-op on x86-64
        return _mm_cvtss_f32(x32);
#endif
    }

#if defined(USE_AVX2) || defined(USE_AVX512)
    friend force_inline simd_vec<float, 8> vectorcall inclusive_scan(simd_vec<float, 8> v1) {
        v1.vec_ = _mm256_add_ps(v1.vec_, _mm256_castsi256_ps(_mm256_slli_si256(_mm256_castps_si256(v1.vec_), 4)));
        v1.vec_ = _mm256_add_ps(v1.vec_, _mm256_castsi256_ps(_mm256_slli_si256(_mm256_castps_si256(v1.vec_), 8)));

        __m256 temp = _mm256_shuffle_ps(v1.vec_, v1.vec_, _MM_SHUFFLE(3, 3, 3, 3));
        temp = _mm256_permute2f128_ps(_mm256_setzero_ps(), temp, 0x20);

        v1.vec_ = _mm256_add_ps(v1.vec_, temp);

        return v1;
    }
#endif

    force_inline void store_to(float *f) const { _mm256_storeu_ps(f, vec_); }
    force_inline void store_to(float *f, simd_mem_aligned_tag) const { _mm256_store_ps(f, vec_); }

    force_inline void vectorcall blend_to(const simd_vec<float, 8> mask, const simd_vec<float, 8> v1) {
        validate_mask(mask);
        vec_ = _mm256_blendv_ps(vec_, v1.vec_, mask.vec_);
    }

    force_inline void vectorcall blend_inv_to(const simd_vec<float, 8> mask, const simd_vec<float, 8> v1) {
        validate_mask(mask);
        vec_ = _mm256_blendv_ps(v1.vec_, vec_, mask.vec_);
    }

    friend force_inline simd_vec<float, 8> vectorcall min(simd_vec<float, 8> v1, simd_vec<float, 8> v2);
    friend force_inline simd_vec<float, 8> vectorcall max(simd_vec<float, 8> v1, simd_vec<float, 8> v2);

    friend force_inline simd_vec<float, 8> vectorcall and_not(simd_vec<float, 8> v1, simd_vec<float, 8> v2);
    friend force_inline simd_vec<float, 8> vectorcall floor(simd_vec<float, 8> v1);
    friend force_inline simd_vec<float, 8> vectorcall ceil(simd_vec<float, 8> v1);

    friend force_inline simd_vec<float, 8> vectorcall operator&(simd_vec<float, 8> v1, simd_vec<float, 8> v2);
    friend force_inline simd_vec<float, 8> vectorcall operator|(simd_vec<float, 8> v1, simd_vec<float, 8> v2);
    friend force_inline simd_vec<float, 8> vectorcall operator^(simd_vec<float, 8> v1, simd_vec<float, 8> v2);
    friend force_inline simd_vec<float, 8> vectorcall operator+(simd_vec<float, 8> v1, simd_vec<float, 8> v2);
    friend force_inline simd_vec<float, 8> vectorcall operator-(simd_vec<float, 8> v1, simd_vec<float, 8> v2);
    friend force_inline simd_vec<float, 8> vectorcall operator*(simd_vec<float, 8> v1, simd_vec<float, 8> v2);
    friend force_inline simd_vec<float, 8> vectorcall operator/(simd_vec<float, 8> v1, simd_vec<float, 8> v2);

    friend force_inline simd_vec<float, 8> vectorcall operator<(simd_vec<float, 8> v1, simd_vec<float, 8> v2);
    friend force_inline simd_vec<float, 8> vectorcall operator<=(simd_vec<float, 8> v1, simd_vec<float, 8> v2);
    friend force_inline simd_vec<float, 8> vectorcall operator>(simd_vec<float, 8> v1, simd_vec<float, 8> v2);
    friend force_inline simd_vec<float, 8> vectorcall operator>=(simd_vec<float, 8> v1, simd_vec<float, 8> v2);
    friend force_inline simd_vec<float, 8> vectorcall operator==(simd_vec<float, 8> v1, simd_vec<float, 8> v2);
    friend force_inline simd_vec<float, 8> vectorcall operator!=(simd_vec<float, 8> v1, simd_vec<float, 8> v2);

    friend force_inline simd_vec<float, 8> vectorcall clamp(simd_vec<float, 8> v1, simd_vec<float, 8> min,
                                                            simd_vec<float, 8> max);
    // friend force_inline simd_vec<float, 8> vectorcall clamp(simd_vec<float, 8> v1, float min, float max);
    friend force_inline simd_vec<float, 8> vectorcall saturate(simd_vec<float, 8> v1) { return clamp(v1, 0.0f, 1.0f); }
    friend force_inline simd_vec<float, 8> vectorcall pow(simd_vec<float, 8> v1, simd_vec<float, 8> v2);

    friend force_inline simd_vec<float, 8> vectorcall normalize(simd_vec<float, 8> v1);
    friend force_inline simd_vec<float, 8> vectorcall normalize_len(simd_vec<float, 8> v1, float &out_len);

#ifdef USE_FMA
    friend force_inline simd_vec<float, 8> vectorcall fmadd(simd_vec<float, 8> a, simd_vec<float, 8> b,
                                                            simd_vec<float, 8> c);
    friend force_inline simd_vec<float, 8> vectorcall fmsub(simd_vec<float, 8> a, simd_vec<float, 8> b,
                                                            simd_vec<float, 8> c);
#endif // USE_FMA

#if defined(USE_AVX2) || defined(USE_AVX512)
    friend force_inline simd_vec<float, 8> vectorcall gather(const float *base_addr, simd_vec<int, 8> vindex);
    friend force_inline simd_vec<float, 8> vectorcall gather(simd_vec<float, 8> src, const float *base_addr,
                                                             simd_vec<int, 8> mask, simd_vec<int, 8> vindex);
#endif

    template <typename U>
    friend force_inline simd_vec<float, 8> vectorcall select(const simd_vec<U, 8> mask, const simd_vec<float, 8> vec1,
                                                             const simd_vec<float, 8> vec2);
    template <typename U>
    friend force_inline simd_vec<int, 8> vectorcall select(const simd_vec<U, 8> mask, const simd_vec<int, 8> vec1,
                                                           const simd_vec<int, 8> vec2);
    template <typename U>
    friend force_inline simd_vec<unsigned, 8> vectorcall select(const simd_vec<U, 8> mask,
                                                                const simd_vec<unsigned, 8> vec1,
                                                                const simd_vec<unsigned, 8> vec2);

#ifndef NDEBUG
    friend void vectorcall __assert_valid_mask(const simd_vec<float, 8> mask) {
        UNROLLED_FOR(i, 8, {
            const float val = mask.get<i>();
            assert(reinterpret_cast<const uint32_t &>(val) == 0 ||
                   reinterpret_cast<const uint32_t &>(val) == 0xffffffff);
        })
    }
#endif

    friend force_inline const float *value_ptr(const simd_vec<float, 8> &v1) {
        return reinterpret_cast<const float *>(&v1.vec_);
    }
    friend force_inline float *value_ptr(simd_vec<float, 8> &v1) { return reinterpret_cast<float *>(&v1.vec_); }

    static int size() { return 8; }
    static bool is_native() { return true; }
};

template <> class simd_vec<int, 8> {
    __m256i vec_;

    friend class simd_vec<float, 8>;
    friend class simd_vec<unsigned, 8>;

  public:
    force_inline simd_vec() = default;
    force_inline simd_vec(const int f) { vec_ = _mm256_set1_epi32(f); }
    force_inline simd_vec(const int i1, const int i2, const int i3, const int i4, const int i5, const int i6,
                          const int i7, const int i8) {
        vec_ = _mm256_setr_epi32(i1, i2, i3, i4, i5, i6, i7, i8);
    }
    force_inline explicit simd_vec(const int *f) { vec_ = _mm256_loadu_si256((const __m256i *)f); }
    force_inline simd_vec(const int *f, simd_mem_aligned_tag) { vec_ = _mm256_load_si256((const __m256i *)f); }

    force_inline int operator[](const int i) const {
#if defined(_MSC_VER) && !defined(__clang__)
        return vec_.m256i_i32[i];
#else
        alignas(32) int comp[8];
        _mm256_store_si256((__m256i *)comp, vec_);
        return comp[i];
#endif
    }

    force_inline int operator[](const long i) const { return operator[](int(i)); }

    template <int i> force_inline int get() const { return _mm256_extract_epi32(vec_, i & 7); }
    template <int i> force_inline void set(const int v) { vec_ = _mm256_insert_epi32(vec_, v, i & 7); }
    force_inline void set(const int i, const int v) {
#if defined(_MSC_VER) && !defined(__clang__)
        vec_.m256i_i32[i] = v;
#else
        static const int maskl[16] = {0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0};
        __m256i mask = _mm256_loadu_si256((const __m256i *)(maskl + 8 - i));
#if defined(USE_AVX2) || defined(USE_AVX512)
        vec_ = _mm256_blendv_epi8(vec_, _mm256_set1_epi32(v), mask);
#else
        __m256 temp1 = _mm256_and_ps(_mm256_castsi256_ps(mask), _mm256_castsi256_ps(_mm256_set1_epi32(v)));
        __m256 temp2 = _mm256_andnot_ps(_mm256_castsi256_ps(mask), _mm256_castsi256_ps(vec_));
        vec_ = _mm256_castps_si256(_mm256_or_ps(temp1, temp2));
#endif
#endif
    }

    force_inline simd_vec<int, 8> &vectorcall operator+=(const simd_vec<int, 8> rhs) {
#if defined(USE_AVX2) || defined(USE_AVX512)
        vec_ = _mm256_add_epi32(vec_, rhs.vec_);
#elif defined(_MSC_VER) && !defined(__clang__)
        UNROLLED_FOR(i, 8, { vec_.m256i_i32[i] += rhs.vec_.m256i_i32[i]; })
#else
        alignas(32) int comp1[8], comp2[8];
        _mm256_store_si256((__m256i *)comp1, vec_);
        _mm256_store_si256((__m256i *)comp2, rhs.vec_);
        UNROLLED_FOR(i, 8, { comp1[i] += comp2[i]; })
        vec_ = _mm256_load_si256((const __m256i *)comp1);
#endif
        return *this;
    }

    force_inline simd_vec<int, 8> &vectorcall operator+=(const int rhs) { return operator+=(simd_vec<int, 8>{rhs}); }

    force_inline simd_vec<int, 8> &vectorcall operator-=(const simd_vec<int, 8> rhs) {
#if defined(USE_AVX2) || defined(USE_AVX512)
        vec_ = _mm256_sub_epi32(vec_, rhs.vec_);
#elif defined(_MSC_VER) && !defined(__clang__)
        UNROLLED_FOR(i, 8, { vec_.m256i_i32[i] -= rhs.vec_.m256i_i32[i]; })
#else
        alignas(32) int comp1[8], comp2[8];
        _mm256_store_si256((__m256i *)comp1, vec_);
        _mm256_store_si256((__m256i *)comp2, rhs.vec_);
        UNROLLED_FOR(i, 8, { comp1[i] -= comp2[i]; })
        vec_ = _mm256_load_si256((const __m256i *)comp1);
#endif
        return *this;
    }

    force_inline simd_vec<int, 8> &vectorcall operator*=(const simd_vec<int, 8> rhs) {
#if defined(USE_AVX2) || defined(USE_AVX512)
        vec_ = _mm256_mullo_epi32(vec_, rhs.vec_);
#elif defined(_MSC_VER) && !defined(__clang__)
        UNROLLED_FOR(i, 8, { vec_.m256i_i32[i] *= rhs.vec_.m256i_i32[i]; })
#else
        alignas(32) int comp1[8], comp2[8];
        _mm256_store_si256((__m256i *)comp1, vec_);
        _mm256_store_si256((__m256i *)comp2, rhs.vec_);
        UNROLLED_FOR(i, 8, { comp1[i] *= comp2[i]; })
        vec_ = _mm256_load_si256((const __m256i *)comp1);
#endif
        return *this;
    }

    force_inline simd_vec<int, 8> &vectorcall operator/=(const simd_vec<int, 8> rhs) {
#if defined(_MSC_VER) && !defined(__clang__)
        UNROLLED_FOR(i, 8, { vec_.m256i_i32[i] /= rhs.vec_.m256i_i32[i]; })
#else
        alignas(32) int comp1[8], comp2[8];
        _mm256_store_si256((__m256i *)comp1, vec_);
        _mm256_store_si256((__m256i *)comp2, rhs.vec_);
        UNROLLED_FOR(i, 8, { comp1[i] /= comp2[i]; })
        vec_ = _mm256_load_si256((const __m256i *)comp1);
#endif
        return *this;
    }

    force_inline simd_vec<int, 8> &vectorcall operator|=(const simd_vec<int, 8> rhs) {
#if defined(USE_AVX2) || defined(USE_AVX512)
        vec_ = _mm256_or_si256(vec_, rhs.vec_);
#elif defined(_MSC_VER) && !defined(__clang__)
        UNROLLED_FOR(i, 8, { vec_.m256i_i32[i] |= rhs.vec_.m256i_i32[i]; })
#else
        alignas(32) int comp1[8], comp2[8];
        _mm256_store_si256((__m256i *)comp1, vec_);
        _mm256_store_si256((__m256i *)comp2, rhs.vec_);
        UNROLLED_FOR(i, 8, { comp1[i] |= comp2[i]; })
        vec_ = _mm256_load_si256((const __m256i *)comp1);
#endif
        return *this;
    }

    force_inline simd_vec<int, 8> &vectorcall operator^=(const simd_vec<int, 8> rhs) {
#if defined(USE_AVX2) || defined(USE_AVX512)
        vec_ = _mm256_xor_si256(vec_, rhs.vec_);
#elif defined(_MSC_VER) && !defined(__clang__)
        UNROLLED_FOR(i, 8, { vec_.m256i_i32[i] ^= rhs.vec_.m256i_i32[i]; })
#else
        alignas(32) int comp1[8], comp2[8];
        _mm256_store_si256((__m256i *)comp1, vec_);
        _mm256_store_si256((__m256i *)comp2, rhs.vec_);
        UNROLLED_FOR(i, 8, { comp1[i] ^= comp2[i]; })
        vec_ = _mm256_load_si256((const __m256i *)comp1);
#endif
        return *this;
    }

    force_inline simd_vec<int, 8> vectorcall operator-() const {
        simd_vec<int, 8> ret;
#if defined(USE_AVX2) || defined(USE_AVX512)
        ret.vec_ = _mm256_sub_epi32(_mm256_setzero_si256(), vec_);
#elif defined(_MSC_VER) && !defined(__clang__)
        UNROLLED_FOR(i, 8, { ret.vec_.m256i_i32[i] = -vec_.m256i_i32[i]; })
#else
        alignas(32) int comp[8];
        _mm256_store_si256((__m256i *)comp, vec_);
        UNROLLED_FOR(i, 8, { comp[i] = -comp[i]; })
        ret.vec_ = _mm256_load_si256((const __m256i *)comp);
#endif
        return ret;
    }

    force_inline simd_vec<int, 8> vectorcall operator==(const simd_vec<int, 8> rhs) const {
        simd_vec<int, 8> ret;
#if defined(USE_AVX2) || defined(USE_AVX512)
        ret.vec_ = _mm256_cmpeq_epi32(vec_, rhs.vec_);
#elif defined(_MSC_VER) && !defined(__clang__)
        UNROLLED_FOR(i, 8, { ret.vec_.m256i_i32[i] = (vec_.m256i_i32[i] == rhs.vec_.m256i_i32[i]) ? -1 : 0; })
#else
        alignas(32) int comp1[8], comp2[8];
        _mm256_store_si256((__m256i *)comp1, vec_);
        _mm256_store_si256((__m256i *)comp2, rhs.vec_);
        UNROLLED_FOR(i, 8, { comp1[i] = (comp1[i] == comp2[i]) ? -1 : 0; })
        ret.vec_ = _mm256_load_si256((const __m256i *)comp1);
#endif
        return ret;
    }

    force_inline simd_vec<int, 8> vectorcall operator!=(const simd_vec<int, 8> rhs) const {
        simd_vec<int, 8> ret;
#if defined(USE_AVX2) || defined(USE_AVX512)
        ret.vec_ = _mm256_andnot_si256(_mm256_cmpeq_epi32(vec_, rhs.vec_), _mm256_set1_epi32(~0));
#elif defined(_MSC_VER) && !defined(__clang__)
        UNROLLED_FOR(i, 8, { ret.vec_.m256i_i32[i] = (vec_.m256i_i32[i] != rhs.vec_.m256i_i32[i]) ? -1 : 0; })
#else
        alignas(32) int comp1[8], comp2[8];
        _mm256_store_si256((__m256i *)comp1, vec_);
        _mm256_store_si256((__m256i *)comp2, rhs.vec_);
        UNROLLED_FOR(i, 8, { comp1[i] = (comp1[i] != comp2[i]) ? -1 : 0; })
        ret.vec_ = _mm256_load_si256((const __m256i *)comp1);
#endif
        return ret;
    }

    force_inline simd_vec<int, 8> &vectorcall operator&=(const simd_vec<int, 8> rhs) {
#if defined(USE_AVX2) || defined(USE_AVX512)
        vec_ = _mm256_and_si256(vec_, rhs.vec_);
#elif defined(_MSC_VER) && !defined(__clang__)
        UNROLLED_FOR(i, 8, { vec_.m256i_i32[i] &= rhs.vec_.m256i_i32[i]; })
#else
        alignas(32) int comp1[8], comp2[8];
        _mm256_store_si256((__m256i *)comp1, vec_);
        _mm256_store_si256((__m256i *)comp2, rhs.vec_);
        UNROLLED_FOR(i, 8, { comp1[i] &= comp2[i]; })
        vec_ = _mm256_load_si256((const __m256i *)comp1);
#endif
        return *this;
    }

    force_inline explicit vectorcall operator simd_vec<float, 8>() const {
        simd_vec<float, 8> ret;
        ret.vec_ = _mm256_cvtepi32_ps(vec_);
        return ret;
    }

    force_inline explicit vectorcall operator simd_vec<unsigned, 8>() const;

    force_inline int hsum() const {
#if defined(USE_AVX2) || defined(USE_AVX512)
        __m256i temp = _mm256_hadd_epi32(vec_, vec_);
        temp = _mm256_hadd_epi32(temp, temp);

        __m256i ret = _mm256_permute2f128_si256(temp, temp, 1);
        ret = _mm256_add_epi32(ret, temp);

        return _mm256_cvtsi256_si32(ret);
#elif defined(_MSC_VER) && !defined(__clang__)
        int ret = vec_.m256i_i32[0];
        UNROLLED_FOR(i, 7, { ret += vec_.m256i_i32[i + 1]; })
        return ret;
#else
        alignas(32) int comp1[8];
        _mm256_store_si256((__m256i *)comp1, vec_);
        UNROLLED_FOR(i, 7, { comp1[0] += comp1[i + 1]; })
        return comp1[0];
#endif
    }

    force_inline void store_to(int *f) const { _mm256_storeu_si256((__m256i *)f, vec_); }
    force_inline void store_to(int *f, simd_mem_aligned_tag) const { _mm256_store_si256((__m256i *)f, vec_); }

    force_inline void vectorcall blend_to(const simd_vec<int, 8> mask, const simd_vec<int, 8> v1) {
        validate_mask(mask);
        vec_ = _mm256_castps_si256(
            _mm256_blendv_ps(_mm256_castsi256_ps(vec_), _mm256_castsi256_ps(v1.vec_), _mm256_castsi256_ps(mask.vec_)));
    }

    force_inline void vectorcall blend_inv_to(const simd_vec<int, 8> mask, const simd_vec<int, 8> v1) {
        validate_mask(mask);
        vec_ = _mm256_castps_si256(
            _mm256_blendv_ps(_mm256_castsi256_ps(v1.vec_), _mm256_castsi256_ps(vec_), _mm256_castsi256_ps(mask.vec_)));
    }

    force_inline int movemask() const { return _mm256_movemask_ps(_mm256_castsi256_ps(vec_)); }

    force_inline bool vectorcall all_zeros() const { return _mm256_test_all_zeros(vec_, vec_) != 0; }
    force_inline bool vectorcall all_zeros(const simd_vec<int, 8> mask) const {
        return _mm256_test_all_zeros(vec_, mask.vec_) != 0;
    }

    force_inline bool vectorcall not_all_zeros() const {
        int res = _mm256_test_all_zeros(vec_, vec_);
        return res == 0;
    }

    friend force_inline simd_vec<int, 8> vectorcall min(const simd_vec<int, 8> v1, const simd_vec<int, 8> v2) {
        simd_vec<int, 8> ret;
#if defined(USE_AVX2) || defined(USE_AVX512)
        ret.vec_ = _mm256_min_epi32(v1.vec_, v2.vec_);
#elif defined(_MSC_VER) && !defined(__clang__)
        UNROLLED_FOR(i, 8, { ret.vec_.m256i_i32[i] = std::min(v1.vec_.m256i_i32[i], v2.vec_.m256i_i32[i]); })
#else
        alignas(32) int comp1[8], comp2[8];
        _mm256_store_si256((__m256i *)comp1, v1.vec_);
        _mm256_store_si256((__m256i *)comp2, v2.vec_);
        UNROLLED_FOR(i, 8, { comp1[i] = std::min(comp1[i], comp2[i]); })
        ret.vec_ = _mm256_load_si256((const __m256i *)comp1);
#endif
        return ret;
    }

    force_inline static simd_vec<int, 8> vectorcall max(const simd_vec<int, 8> v1, const simd_vec<int, 8> v2) {
        simd_vec<int, 8> ret;
#if defined(USE_AVX2) || defined(USE_AVX512)
        ret.vec_ = _mm256_max_epi32(v1.vec_, v2.vec_);
#elif defined(_MSC_VER) && !defined(__clang__)
        UNROLLED_FOR(i, 8, { ret.vec_.m256i_i32[i] = std::max(v1.vec_.m256i_i32[i], v2.vec_.m256i_i32[i]); })
#else
        alignas(32) int comp1[8], comp2[8];
        _mm256_store_si256((__m256i *)comp1, v1.vec_);
        _mm256_store_si256((__m256i *)comp2, v2.vec_);
        UNROLLED_FOR(i, 8, { comp1[i] = std::max(comp1[i], comp2[i]); })
        ret.vec_ = _mm256_load_si256((const __m256i *)comp1);
#endif
        return ret;
    }

    friend force_inline simd_vec<int, 8> vectorcall clamp(const simd_vec<int, 8> v1, const simd_vec<int, 8> _min,
                                                          const simd_vec<int, 8> _max) {
        return max(_min, min(v1, _max));
    }

    force_inline static simd_vec<int, 8> vectorcall and_not(const simd_vec<int, 8> v1, const simd_vec<int, 8> v2) {
        simd_vec<int, 8> temp;
        temp.vec_ = _mm256_castps_si256(_mm256_andnot_ps(_mm256_castsi256_ps(v1.vec_), _mm256_castsi256_ps(v2.vec_)));
        return temp;
    }

    friend force_inline simd_vec<int, 8> vectorcall operator&(const simd_vec<int, 8> v1, const simd_vec<int, 8> v2) {
        simd_vec<int, 8> temp;
        temp.vec_ = _mm256_castps_si256(_mm256_and_ps(_mm256_castsi256_ps(v1.vec_), _mm256_castsi256_ps(v2.vec_)));
        return temp;
    }

    friend force_inline simd_vec<int, 8> vectorcall operator|(const simd_vec<int, 8> v1, const simd_vec<int, 8> v2) {
        simd_vec<int, 8> temp;
        temp.vec_ = _mm256_castps_si256(_mm256_or_ps(_mm256_castsi256_ps(v1.vec_), _mm256_castsi256_ps(v2.vec_)));
        return temp;
    }

    friend force_inline simd_vec<int, 8> vectorcall operator^(const simd_vec<int, 8> v1, const simd_vec<int, 8> v2) {
        simd_vec<int, 8> temp;
        temp.vec_ = _mm256_castps_si256(_mm256_xor_ps(_mm256_castsi256_ps(v1.vec_), _mm256_castsi256_ps(v2.vec_)));
        return temp;
    }

    friend force_inline simd_vec<int, 8> vectorcall operator+(const simd_vec<int, 8> v1, const simd_vec<int, 8> v2) {
        simd_vec<int, 8> ret;
#if defined(USE_AVX2) || defined(USE_AVX512)
        ret.vec_ = _mm256_add_epi32(v1.vec_, v2.vec_);
#elif defined(_MSC_VER) && !defined(__clang__)
        UNROLLED_FOR(i, 8, { ret.vec_.m256i_i32[i] = v1.vec_.m256i_i32[i] + v2.vec_.m256i_i32[i]; })
#else
        alignas(32) int comp1[8], comp2[8];
        _mm256_store_si256((__m256i *)comp1, v1.vec_);
        _mm256_store_si256((__m256i *)comp2, v2.vec_);
        UNROLLED_FOR(i, 8, { comp1[i] += comp2[i]; })
        ret.vec_ = _mm256_load_si256((const __m256i *)comp1);
#endif
        return ret;
    }

    friend force_inline simd_vec<int, 8> vectorcall operator-(const simd_vec<int, 8> v1, const simd_vec<int, 8> v2) {
        simd_vec<int, 8> ret;
#if defined(USE_AVX2) || defined(USE_AVX512)
        ret.vec_ = _mm256_sub_epi32(v1.vec_, v2.vec_);
#elif defined(_MSC_VER) && !defined(__clang__)
        UNROLLED_FOR(i, 8, { ret.vec_.m256i_i32[i] = v1.vec_.m256i_i32[i] - v2.vec_.m256i_i32[i]; })
#else
        alignas(32) int comp1[8], comp2[8];
        _mm256_store_si256((__m256i *)comp1, v1.vec_);
        _mm256_store_si256((__m256i *)comp2, v2.vec_);
        UNROLLED_FOR(i, 8, { comp1[i] -= comp2[i]; })
        ret.vec_ = _mm256_load_si256((const __m256i *)comp1);
#endif
        return ret;
    }

    friend force_inline simd_vec<int, 8> vectorcall operator*(const simd_vec<int, 8> v1, const simd_vec<int, 8> v2) {
        simd_vec<int, 8> ret;
#if defined(USE_AVX2) || defined(USE_AVX512)
        ret.vec_ = _mm256_mullo_epi32(v1.vec_, v2.vec_);
#elif defined(_MSC_VER) && !defined(__clang__)
        UNROLLED_FOR(i, 8, { ret.vec_.m256i_i32[i] = v1.vec_.m256i_i32[i] * v2.vec_.m256i_i32[i]; })
#else
        alignas(32) int comp1[8], comp2[8];
        _mm256_store_si256((__m256i *)comp1, v1.vec_);
        _mm256_store_si256((__m256i *)comp2, v2.vec_);
        UNROLLED_FOR(i, 8, { comp1[i] *= comp2[i]; })
        ret.vec_ = _mm256_load_si256((const __m256i *)comp1);
#endif
        return ret;
    }

    friend force_inline simd_vec<int, 8> vectorcall operator/(const simd_vec<int, 8> v1, const simd_vec<int, 8> v2) {
        simd_vec<int, 8> ret;
#if defined(_MSC_VER) && !defined(__clang__)
        UNROLLED_FOR(i, 8, { ret.vec_.m256i_i32[i] = (v1.vec_.m256i_i32[i] / v2.vec_.m256i_i32[i]); })
#else
        alignas(32) int comp1[8], comp2[8];
        _mm256_store_si256((__m256i *)comp1, v1.vec_);
        _mm256_store_si256((__m256i *)comp2, v2.vec_);
        UNROLLED_FOR(i, 8, { comp1[i] /= comp2[i]; })
        ret.vec_ = _mm256_load_si256((const __m256i *)comp1);
#endif
        return ret;
    }

    friend force_inline simd_vec<int, 8> vectorcall operator<(const simd_vec<int, 8> v1, const simd_vec<int, 8> v2) {
        simd_vec<int, 8> ret;
#if defined(USE_AVX2) || defined(USE_AVX512)
        ret.vec_ = _mm256_cmpgt_epi32(v2.vec_, v1.vec_);
#elif defined(_MSC_VER) && !defined(__clang__)
        UNROLLED_FOR(i, 8, { ret.vec_.m256i_i32[i] = (v1.vec_.m256i_i32[i] < v2.vec_.m256i_i32[i]) ? -1 : 0; })
#else
        alignas(32) int comp1[8], comp2[8];
        _mm256_store_si256((__m256i *)comp1, v1.vec_);
        _mm256_store_si256((__m256i *)comp2, v2.vec_);
        UNROLLED_FOR(i, 8, { comp1[i] = comp1[i] < comp2[i] ? -1 : 0; })
        ret.vec_ = _mm256_load_si256((const __m256i *)comp1);
#endif
        return ret;
    }

    friend force_inline simd_vec<int, 8> vectorcall operator>(const simd_vec<int, 8> v1, const simd_vec<int, 8> v2) {
        simd_vec<int, 8> ret;
#if defined(USE_AVX2) || defined(USE_AVX512)
        ret.vec_ = _mm256_cmpgt_epi32(v1.vec_, v2.vec_);
#elif defined(_MSC_VER) && !defined(__clang__)
        UNROLLED_FOR(i, 8, { ret.vec_.m256i_i32[i] = (v1.vec_.m256i_i32[i] > v2.vec_.m256i_i32[i]) ? -1 : 0; })
#else
        alignas(32) int comp1[8], comp2[8];
        _mm256_store_si256((__m256i *)comp1, v1.vec_);
        _mm256_store_si256((__m256i *)comp2, v2.vec_);
        UNROLLED_FOR(i, 8, { comp1[i] = comp1[i] > comp2[i] ? -1 : 0; })
        ret.vec_ = _mm256_load_si256((const __m256i *)comp1);
#endif
        return ret;
    }

    friend force_inline simd_vec<int, 8> vectorcall operator>=(const simd_vec<int, 8> v1, const simd_vec<int, 8> v2) {
        simd_vec<int, 8> ret;
#if defined(USE_AVX2) || defined(USE_AVX512)
        ret.vec_ = _mm256_andnot_si256(_mm256_cmpgt_epi32(v2.vec_, v1.vec_), _mm256_set1_epi32(-1));
#elif defined(_MSC_VER) && !defined(__clang__)
        UNROLLED_FOR(i, 8, { ret.vec_.m256i_i32[i] = (v1.vec_.m256i_i32[i] >= v2.vec_.m256i_i32[i]) ? -1 : 0; })
#else
        alignas(32) int comp1[8], comp2[8];
        _mm256_store_si256((__m256i *)comp1, v1.vec_);
        _mm256_store_si256((__m256i *)comp2, v2.vec_);
        UNROLLED_FOR(i, 8, { comp1[i] = comp1[i] >= comp2[i] ? -1 : 0; })
        ret.vec_ = _mm256_load_si256((const __m256i *)comp1);
#endif
        return ret;
    }

    friend force_inline simd_vec<int, 8> vectorcall operator>>(const simd_vec<int, 8> v1, const simd_vec<int, 8> v2) {
        simd_vec<int, 8> ret;
#if defined(USE_AVX2) || defined(USE_AVX512)
        ret.vec_ = _mm256_srlv_epi32(v1.vec_, v2.vec_);
#elif defined(_MSC_VER) && !defined(__clang__)
        UNROLLED_FOR(i, 8, { ret.vec_.m256i_u32[i] = (v1.vec_.m256i_u32[i] >> v2.vec_.m256i_u32[i]); })
#else
        alignas(32) int comp1[8], comp2[8];
        _mm256_store_si256((__m256i *)comp1, v1.vec_);
        _mm256_store_si256((__m256i *)comp2, v2.vec_);
        UNROLLED_FOR(i, 8, { comp1[i] = reinterpret_cast<const unsigned &>(comp1[i]) >> comp2[i]; })
        ret.vec_ = _mm256_load_si256((const __m256i *)comp1);
#endif
        return ret;
    }

    friend force_inline simd_vec<int, 8> vectorcall operator>>(const simd_vec<int, 8> v1, const int v2) {
        simd_vec<int, 8> ret;
#if defined(USE_AVX2) || defined(USE_AVX512)
        ret.vec_ = _mm256_srli_epi32(v1.vec_, v2);
#elif defined(_MSC_VER) && !defined(__clang__)
        UNROLLED_FOR(i, 8, { ret.vec_.m256i_u32[i] = (v1.vec_.m256i_u32[i] >> v2); })
#else
        alignas(32) int comp1[8];
        _mm256_store_si256((__m256i *)comp1, v1.vec_);
        UNROLLED_FOR(i, 8, { comp1[i] = reinterpret_cast<const unsigned &>(comp1[i]) >> v2; })
        ret.vec_ = _mm256_load_si256((const __m256i *)comp1);
#endif
        return ret;
    }

    friend force_inline simd_vec<int, 8> vectorcall operator<<(const simd_vec<int, 8> v1, const simd_vec<int, 8> v2) {
        simd_vec<int, 8> ret;
#if defined(USE_AVX2) || defined(USE_AVX512)
        ret.vec_ = _mm256_sllv_epi32(v1.vec_, v2.vec_);
#elif defined(_MSC_VER) && !defined(__clang__)
        UNROLLED_FOR(i, 8, { ret.vec_.m256i_u32[i] = (v1.vec_.m256i_u32[i] << v2.vec_.m256i_u32[i]); })
#else
        alignas(32) int comp1[8], comp2[8];
        _mm256_store_si256((__m256i *)comp1, v1.vec_);
        _mm256_store_si256((__m256i *)comp2, v2.vec_);
        UNROLLED_FOR(i, 8, { comp1[i] = comp1[i] << comp2[i]; })
        ret.vec_ = _mm256_load_si256((const __m256i *)comp1);
#endif
        return ret;
    }

    friend force_inline simd_vec<int, 8> vectorcall operator<<(const simd_vec<int, 8> v1, const int v2) {
        simd_vec<int, 8> ret;
#if defined(USE_AVX2) || defined(USE_AVX512)
        ret.vec_ = _mm256_slli_epi32(v1.vec_, v2);
#elif defined(_MSC_VER) && !defined(__clang__)
        UNROLLED_FOR(i, 8, { ret.vec_.m256i_u32[i] = (v1.vec_.m256i_u32[i] << v2); })
#else
        alignas(32) int comp1[8];
        _mm256_store_si256((__m256i *)comp1, v1.vec_);
        UNROLLED_FOR(i, 8, { comp1[i] = comp1[i] << v2; })
        ret.vec_ = _mm256_load_si256((const __m256i *)comp1);
#endif
        return ret;
    }

    force_inline simd_vec<int, 8> operator~() const {
        simd_vec<int, 8> ret;
#if defined(USE_AVX2) || defined(USE_AVX512)
        ret.vec_ = _mm256_andnot_si256(vec_, _mm256_set1_epi32(~0));
#elif defined(_MSC_VER) && !defined(__clang__)
        UNROLLED_FOR(i, 8, { ret.vec_.m256i_u32[i] = ~vec_.m256i_u32[i]; })
#else
        alignas(32) int comp[8];
        _mm256_store_si256((__m256i *)comp, vec_);
        UNROLLED_FOR(i, 8, { comp[i] = ~comp[i]; })
        ret.vec_ = _mm256_load_si256((const __m256i *)comp);
#endif
        return ret;
    }

    friend force_inline simd_vec<int, 8> vectorcall srai(const simd_vec<int, 8> v1, const int v2) {
        simd_vec<int, 8> ret;
#if defined(USE_AVX2) || defined(USE_AVX512)
        ret.vec_ = _mm256_srai_epi32(v1.vec_, v2);
#elif defined(_MSC_VER) && !defined(__clang__)
        UNROLLED_FOR(i, 8, { ret.vec_.m256i_i32[i] = (v1.vec_.m256i_i32[i] >> v2); })
#else
        alignas(32) int comp[8];
        _mm256_store_si256((__m256i *)comp, v1.vec_);
        UNROLLED_FOR(i, 8, { comp[i] >>= v2; })
        ret.vec_ = _mm256_load_si256((const __m256i *)comp);
#endif
        return ret;
    }

    friend force_inline bool vectorcall is_equal(const simd_vec<int, 8> v1, const simd_vec<int, 8> v2) {
#if defined(USE_AVX2) || defined(USE_AVX512)
        __m256i vcmp = _mm256_cmpeq_epi32(v1.vec_, v2.vec_);
        return (_mm256_movemask_epi8(vcmp) == 0xffffffff);
#elif defined(_MSC_VER) && !defined(__clang__)
        bool ret = true;

        UNROLLED_FOR(i, 8, { ret &= (v1.vec_.m256i_i32[i] == v2.vec_.m256i_i32[i]); })

        return ret;
#else
        bool ret = true;

        alignas(32) int comp1[8], comp2[8];
        _mm256_store_si256((__m256i *)comp1, v1.vec_);
        _mm256_store_si256((__m256i *)comp2, v2.vec_);
        UNROLLED_FOR(i, 8, { ret &= (comp1[i] == comp2[i]); })

        return ret;
#endif
    }

#if defined(USE_AVX2) || defined(USE_AVX512)
    friend force_inline simd_vec<int, 8> vectorcall inclusive_scan(simd_vec<int, 8> v1) {
        v1.vec_ = _mm256_add_epi32(v1.vec_, _mm256_slli_si256(v1.vec_, 4));
        v1.vec_ = _mm256_add_epi32(v1.vec_, _mm256_slli_si256(v1.vec_, 8));

        __m256i temp = _mm256_shuffle_epi32(v1.vec_, _MM_SHUFFLE(3, 3, 3, 3));
        temp = _mm256_permute2x128_si256(_mm256_setzero_si256(), temp, 0x20);

        v1.vec_ = _mm256_add_epi32(v1.vec_, temp);

        return v1;
    }
#endif

#if defined(USE_AVX2) || defined(USE_AVX512)
    friend force_inline simd_vec<float, 8> vectorcall gather(const float *base_addr, simd_vec<int, 8> vindex);
    friend force_inline simd_vec<float, 8> vectorcall gather(simd_vec<float, 8> src, const float *base_addr,
                                                             simd_vec<int, 8> mask, simd_vec<int, 8> vindex);
    friend force_inline simd_vec<int, 8> vectorcall gather(const int *base_addr, simd_vec<int, 8> vindex);
    friend force_inline simd_vec<int, 8> vectorcall gather(simd_vec<int, 8> src, const int *base_addr,
                                                           simd_vec<int, 8> mask, simd_vec<int, 8> vindex);
    friend force_inline simd_vec<unsigned, 8> vectorcall gather(const unsigned *base_addr, simd_vec<int, 8> vindex);
    friend force_inline simd_vec<unsigned, 8> vectorcall gather(simd_vec<unsigned, 8> src, const unsigned *base_addr,
                                                                simd_vec<unsigned, 8> mask, simd_vec<int, 8> vindex);
#endif

    template <typename U>
    friend force_inline simd_vec<float, 8> vectorcall select(const simd_vec<U, 8> mask, const simd_vec<float, 8> vec1,
                                                             const simd_vec<float, 8> vec2);
    template <typename U>
    friend force_inline simd_vec<int, 8> vectorcall select(const simd_vec<U, 8> mask, const simd_vec<int, 8> vec1,
                                                           const simd_vec<int, 8> vec2);
    template <typename U>
    friend force_inline simd_vec<unsigned, 8> vectorcall select(const simd_vec<U, 8> mask,
                                                                const simd_vec<unsigned, 8> vec1,
                                                                const simd_vec<unsigned, 8> vec2);

#ifndef NDEBUG
    friend void vectorcall __assert_valid_mask(const simd_vec<int, 8> mask) {
        UNROLLED_FOR(i, 8, {
            const int val = mask.get<i>();
            assert(val == 0 || val == -1);
        })
    }
#endif

    friend force_inline const int *value_ptr(const simd_vec<int, 8> &v1) {
        return reinterpret_cast<const int *>(&v1.vec_);
    }
    friend force_inline int *value_ptr(simd_vec<int, 8> &v1) { return reinterpret_cast<int *>(&v1.vec_); }

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

template <> class simd_vec<unsigned, 8> {
    __m256i vec_;

    friend class simd_vec<float, 8>;
    friend class simd_vec<int, 8>;

  public:
    force_inline simd_vec() = default;
    force_inline simd_vec(const unsigned f) { vec_ = _mm256_set1_epi32(f); }
    force_inline simd_vec(const unsigned i1, const unsigned i2, const unsigned i3, const unsigned i4, const unsigned i5,
                          const unsigned i6, const unsigned i7, const unsigned i8) {
        vec_ = _mm256_setr_epi32(i1, i2, i3, i4, i5, i6, i7, i8);
    }
    force_inline explicit simd_vec(const unsigned *f) { vec_ = _mm256_loadu_si256((const __m256i *)f); }
    force_inline simd_vec(const unsigned *f, simd_mem_aligned_tag) { vec_ = _mm256_load_si256((const __m256i *)f); }

    force_inline unsigned operator[](const int i) const {
#if defined(_MSC_VER) && !defined(__clang__)
        return vec_.m256i_u32[i];
#else
        alignas(32) unsigned comp[8];
        _mm256_store_si256((__m256i *)comp, vec_);
        return comp[i];
#endif
    }

    force_inline unsigned operator[](const long i) const { return operator[](int(i)); }

    template <int i> force_inline unsigned get() const { return _mm256_extract_epi32(vec_, i & 7); }
    template <int i> force_inline void set(const unsigned v) { vec_ = _mm256_insert_epi32(vec_, v, i & 7); }
    force_inline void set(const int i, const unsigned v) {
#if defined(_MSC_VER) && !defined(__clang__)
        vec_.m256i_u32[i] = v;
#else
        static const int maskl[16] = {0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0};
        __m256i mask = _mm256_loadu_si256((const __m256i *)(maskl + 8 - i));
#if defined(USE_AVX2) || defined(USE_AVX512)
        vec_ = _mm256_blendv_epi8(vec_, _mm256_set1_epi32(v), mask);
#else
        __m256 temp1 = _mm256_and_ps(_mm256_castsi256_ps(mask), _mm256_castsi256_ps(_mm256_set1_epi32(v)));
        __m256 temp2 = _mm256_andnot_ps(_mm256_castsi256_ps(mask), _mm256_castsi256_ps(vec_));
        vec_ = _mm256_castps_si256(_mm256_or_ps(temp1, temp2));
#endif
#endif
    }

    force_inline simd_vec<unsigned, 8> &vectorcall operator+=(const simd_vec<unsigned, 8> rhs) {
#if defined(USE_AVX2) || defined(USE_AVX512)
        vec_ = _mm256_add_epi32(vec_, rhs.vec_);
#elif defined(_MSC_VER) && !defined(__clang__)
        UNROLLED_FOR(i, 8, { vec_.m256i_u32[i] += rhs.vec_.m256i_u32[i]; })
#else
        alignas(32) unsigned comp1[8], comp2[8];
        _mm256_store_si256((__m256i *)comp1, vec_);
        _mm256_store_si256((__m256i *)comp2, rhs.vec_);
        UNROLLED_FOR(i, 8, { comp1[i] += comp2[i]; })
        vec_ = _mm256_load_si256((const __m256i *)comp1);
#endif
        return *this;
    }

    force_inline simd_vec<unsigned, 8> &vectorcall operator+=(const unsigned rhs) {
        return operator+=(simd_vec<unsigned, 8>{rhs});
    }

    force_inline simd_vec<unsigned, 8> &vectorcall operator-=(const simd_vec<unsigned, 8> rhs) {
#if defined(USE_AVX2) || defined(USE_AVX512)
        vec_ = _mm256_sub_epi32(vec_, rhs.vec_);
#elif defined(_MSC_VER) && !defined(__clang__)
        UNROLLED_FOR(i, 8, { vec_.m256i_u32[i] -= rhs.vec_.m256i_u32[i]; })
#else
        alignas(32) unsigned comp1[8], comp2[8];
        _mm256_store_si256((__m256i *)comp1, vec_);
        _mm256_store_si256((__m256i *)comp2, rhs.vec_);
        UNROLLED_FOR(i, 8, { comp1[i] -= comp2[i]; })
        vec_ = _mm256_load_si256((const __m256i *)comp1);
#endif
        return *this;
    }

    force_inline simd_vec<unsigned, 8> &vectorcall operator-=(const unsigned rhs) {
        return operator-=(simd_vec<unsigned, 8>{rhs});
    }

    force_inline simd_vec<unsigned, 8> &vectorcall operator*=(const simd_vec<unsigned, 8> rhs) {
#if 0 // defined(USE_AVX2) || defined(USE_AVX512)
        vec_ = _mm256_mullo_epi32(vec_, rhs.vec_);
#elif defined(_MSC_VER) && !defined(__clang__)
        UNROLLED_FOR(i, 8, { vec_.m256i_u32[i] *= rhs.vec_.m256i_u32[i]; })
#else
        alignas(32) unsigned comp1[8], comp2[8];
        _mm256_store_si256((__m256i *)comp1, vec_);
        _mm256_store_si256((__m256i *)comp2, rhs.vec_);
        UNROLLED_FOR(i, 8, { comp1[i] *= comp2[i]; })
        vec_ = _mm256_load_si256((const __m256i *)comp1);
#endif
        return *this;
    }

    force_inline simd_vec<unsigned, 8> &vectorcall operator*=(const unsigned rhs) {
        return operator*=(simd_vec<unsigned, 8>{rhs});
    }

    force_inline simd_vec<unsigned, 8> &vectorcall operator/=(const simd_vec<unsigned, 8> rhs) {
#if defined(_MSC_VER) && !defined(__clang__)
        UNROLLED_FOR(i, 8, { vec_.m256i_u32[i] /= rhs.vec_.m256i_u32[i]; })
#else
        alignas(32) unsigned comp1[8], comp2[8];
        _mm256_store_si256((__m256i *)comp1, vec_);
        _mm256_store_si256((__m256i *)comp2, rhs.vec_);
        UNROLLED_FOR(i, 8, { comp1[i] /= comp2[i]; })
        vec_ = _mm256_load_si256((const __m256i *)comp1);
#endif
        return *this;
    }

    force_inline simd_vec<unsigned, 8> &vectorcall operator/=(const unsigned rhs) {
        return operator/=(simd_vec<unsigned, 8>{rhs});
    }

    force_inline simd_vec<unsigned, 8> &vectorcall operator|=(const simd_vec<unsigned, 8> rhs) {
#if defined(USE_AVX2) || defined(USE_AVX512)
        vec_ = _mm256_or_si256(vec_, rhs.vec_);
#elif defined(_MSC_VER) && !defined(__clang__)
        UNROLLED_FOR(i, 8, { vec_.m256i_u32[i] |= rhs.vec_.m256i_u32[i]; })
#else
        alignas(32) unsigned comp1[8], comp2[8];
        _mm256_store_si256((__m256i *)comp1, vec_);
        _mm256_store_si256((__m256i *)comp2, rhs.vec_);
        UNROLLED_FOR(i, 8, { comp1[i] |= comp2[i]; })
        vec_ = _mm256_load_si256((const __m256i *)comp1);
#endif
        return *this;
    }

    force_inline simd_vec<unsigned, 8> &vectorcall operator^=(const simd_vec<unsigned, 8> rhs) {
#if defined(USE_AVX2) || defined(USE_AVX512)
        vec_ = _mm256_xor_si256(vec_, rhs.vec_);
#elif defined(_MSC_VER) && !defined(__clang__)
        UNROLLED_FOR(i, 8, { vec_.m256i_u32[i] ^= rhs.vec_.m256i_u32[i]; })
#else
        alignas(32) unsigned comp1[8], comp2[8];
        _mm256_store_si256((__m256i *)comp1, vec_);
        _mm256_store_si256((__m256i *)comp2, rhs.vec_);
        UNROLLED_FOR(i, 8, { comp1[i] ^= comp2[i]; })
        vec_ = _mm256_load_si256((const __m256i *)comp1);
#endif
        return *this;
    }

    force_inline simd_vec<unsigned, 8> vectorcall operator==(const simd_vec<unsigned, 8> rhs) const {
        simd_vec<unsigned, 8> ret;
#if defined(USE_AVX2) || defined(USE_AVX512)
        ret.vec_ = _mm256_cmpeq_epi32(vec_, rhs.vec_);
#elif defined(_MSC_VER) && !defined(__clang__)
        UNROLLED_FOR(i, 8, { ret.vec_.m256i_u32[i] = (vec_.m256i_u32[i] == rhs.vec_.m256i_u32[i]) ? 0xffffffff : 0; })
#else
        alignas(32) unsigned comp1[8], comp2[8];
        _mm256_store_si256((__m256i *)comp1, vec_);
        _mm256_store_si256((__m256i *)comp2, rhs.vec_);
        UNROLLED_FOR(i, 8, { comp1[i] = (comp1[i] == comp2[i]) ? -1 : 0; })
        ret.vec_ = _mm256_load_si256((const __m256i *)comp1);
#endif
        return ret;
    }

    force_inline simd_vec<unsigned, 8> vectorcall operator!=(const simd_vec<unsigned, 8> rhs) const {
        simd_vec<unsigned, 8> ret;
#if defined(USE_AVX2) || defined(USE_AVX512)
        ret.vec_ = _mm256_andnot_si256(_mm256_cmpeq_epi32(vec_, rhs.vec_), _mm256_set1_epi32(~0));
#elif defined(_MSC_VER) && !defined(__clang__)
        UNROLLED_FOR(i, 8, { ret.vec_.m256i_u32[i] = (vec_.m256i_u32[i] != rhs.vec_.m256i_u32[i]) ? 0xffffffff : 0; })
#else
        alignas(32) unsigned comp1[8], comp2[8];
        _mm256_store_si256((__m256i *)comp1, vec_);
        _mm256_store_si256((__m256i *)comp2, rhs.vec_);
        UNROLLED_FOR(i, 8, { comp1[i] = (comp1[i] != comp2[i]) ? -1 : 0; })
        ret.vec_ = _mm256_load_si256((const __m256i *)comp1);
#endif
        return ret;
    }

    force_inline simd_vec<unsigned, 8> &vectorcall operator&=(const simd_vec<unsigned, 8> rhs) {
#if defined(USE_AVX2) || defined(USE_AVX512)
        vec_ = _mm256_and_si256(vec_, rhs.vec_);
#elif defined(_MSC_VER) && !defined(__clang__)
        UNROLLED_FOR(i, 8, { vec_.m256i_u32[i] &= rhs.vec_.m256i_u32[i]; })
#else
        alignas(32) unsigned comp1[8], comp2[8];
        _mm256_store_si256((__m256i *)comp1, vec_);
        _mm256_store_si256((__m256i *)comp2, rhs.vec_);
        UNROLLED_FOR(i, 8, { comp1[i] &= comp2[i]; })
        vec_ = _mm256_load_si256((const __m256i *)comp1);
#endif
        return *this;
    }

    force_inline explicit vectorcall operator simd_vec<float, 8>() const {
        simd_vec<float, 8> ret;
        ret.vec_ = _mm256_cvtepi32_ps(vec_);
        return ret;
    }

    force_inline explicit vectorcall operator simd_vec<int, 8>() const {
        simd_vec<int, 8> ret;
        ret.vec_ = vec_;
        return ret;
    }

    force_inline unsigned hsum() const {
#if defined(USE_AVX2) || defined(USE_AVX512)
        __m256i temp = _mm256_hadd_epi32(vec_, vec_);
        temp = _mm256_hadd_epi32(temp, temp);

        __m256i ret = _mm256_permute2f128_si256(temp, temp, 1);
        ret = _mm256_add_epi32(ret, temp);

        return _mm256_cvtsi256_si32(ret);
#elif defined(_MSC_VER) && !defined(__clang__)
        unsigned ret = vec_.m256i_u32[0];
        UNROLLED_FOR(i, 7, { ret += vec_.m256i_u32[i + 1]; })
        return ret;
#else
        alignas(32) unsigned comp1[8];
        _mm256_store_si256((__m256i *)comp1, vec_);
        UNROLLED_FOR(i, 7, { comp1[0] += comp1[i + 1]; })
        return comp1[0];
#endif
    }

    force_inline void store_to(unsigned *f) const { _mm256_storeu_si256((__m256i *)f, vec_); }
    force_inline void store_to(unsigned *f, simd_mem_aligned_tag) const { _mm256_store_si256((__m256i *)f, vec_); }

    force_inline void vectorcall blend_to(const simd_vec<unsigned, 8> mask, const simd_vec<unsigned, 8> v1) {
        validate_mask(mask);
        vec_ = _mm256_castps_si256(
            _mm256_blendv_ps(_mm256_castsi256_ps(vec_), _mm256_castsi256_ps(v1.vec_), _mm256_castsi256_ps(mask.vec_)));
    }

    force_inline void vectorcall blend_inv_to(const simd_vec<unsigned, 8> mask, const simd_vec<unsigned, 8> v1) {
        validate_mask(mask);
        vec_ = _mm256_castps_si256(
            _mm256_blendv_ps(_mm256_castsi256_ps(v1.vec_), _mm256_castsi256_ps(vec_), _mm256_castsi256_ps(mask.vec_)));
    }

    force_inline int movemask() const { return _mm256_movemask_ps(_mm256_castsi256_ps(vec_)); }

    force_inline bool vectorcall all_zeros() const { return _mm256_test_all_zeros(vec_, vec_) != 0; }
    force_inline bool vectorcall all_zeros(const simd_vec<unsigned, 8> mask) const {
        return _mm256_test_all_zeros(vec_, mask.vec_) != 0;
    }

    force_inline bool vectorcall not_all_zeros() const {
        int res = _mm256_test_all_zeros(vec_, vec_);
        return res == 0;
    }

    friend force_inline simd_vec<unsigned, 8> vectorcall min(const simd_vec<unsigned, 8> v1,
                                                             const simd_vec<unsigned, 8> v2) {
        simd_vec<unsigned, 8> ret;
#if defined(USE_AVX2) || defined(USE_AVX512)
        ret.vec_ = _mm256_min_epu32(v1.vec_, v2.vec_);
#elif defined(_MSC_VER) && !defined(__clang__)
        UNROLLED_FOR(i, 8, { ret.vec_.m256i_u32[i] = std::min(v1.vec_.m256i_u32[i], v2.vec_.m256i_u32[i]); })
#else
        alignas(32) unsigned comp1[8], comp2[8];
        _mm256_store_si256((__m256i *)comp1, v1.vec_);
        _mm256_store_si256((__m256i *)comp2, v2.vec_);
        UNROLLED_FOR(i, 8, { comp1[i] = std::min(comp1[i], comp2[i]); })
        ret.vec_ = _mm256_load_si256((const __m256i *)comp1);
#endif
        return ret;
    }

    force_inline static simd_vec<unsigned, 8> vectorcall max(const simd_vec<unsigned, 8> v1,
                                                             const simd_vec<unsigned, 8> v2) {
        simd_vec<unsigned, 8> ret;
#if defined(USE_AVX2) || defined(USE_AVX512)
        ret.vec_ = _mm256_max_epu32(v1.vec_, v2.vec_);
#elif defined(_MSC_VER) && !defined(__clang__)
        UNROLLED_FOR(i, 8, { ret.vec_.m256i_u32[i] = std::max(v1.vec_.m256i_u32[i], v2.vec_.m256i_u32[i]); })
#else
        alignas(32) unsigned comp1[8], comp2[8];
        _mm256_store_si256((__m256i *)comp1, v1.vec_);
        _mm256_store_si256((__m256i *)comp2, v2.vec_);
        UNROLLED_FOR(i, 8, { comp1[i] = std::max(comp1[i], comp2[i]); })
        ret.vec_ = _mm256_load_si256((const __m256i *)comp1);
#endif
        return ret;
    }

    friend force_inline simd_vec<unsigned, 8> vectorcall clamp(const simd_vec<unsigned, 8> v1,
                                                               const simd_vec<unsigned, 8> _min,
                                                               const simd_vec<unsigned, 8> _max) {
        return max(_min, min(v1, _max));
    }

    force_inline static simd_vec<unsigned, 8> vectorcall and_not(const simd_vec<unsigned, 8> v1,
                                                                 const simd_vec<unsigned, 8> v2) {
        simd_vec<unsigned, 8> temp;
        temp.vec_ = _mm256_castps_si256(_mm256_andnot_ps(_mm256_castsi256_ps(v1.vec_), _mm256_castsi256_ps(v2.vec_)));
        return temp;
    }

    friend force_inline simd_vec<unsigned, 8> vectorcall operator&(const simd_vec<unsigned, 8> v1,
                                                                   const simd_vec<unsigned, 8> v2) {
        simd_vec<unsigned, 8> temp;
        temp.vec_ = _mm256_castps_si256(_mm256_and_ps(_mm256_castsi256_ps(v1.vec_), _mm256_castsi256_ps(v2.vec_)));
        return temp;
    }

    friend force_inline simd_vec<unsigned, 8> vectorcall operator|(const simd_vec<unsigned, 8> v1,
                                                                   const simd_vec<unsigned, 8> v2) {
        simd_vec<unsigned, 8> temp;
        temp.vec_ = _mm256_castps_si256(_mm256_or_ps(_mm256_castsi256_ps(v1.vec_), _mm256_castsi256_ps(v2.vec_)));
        return temp;
    }

    friend force_inline simd_vec<unsigned, 8> vectorcall operator^(const simd_vec<unsigned, 8> v1,
                                                                   const simd_vec<unsigned, 8> v2) {
        simd_vec<unsigned, 8> temp;
        temp.vec_ = _mm256_castps_si256(_mm256_xor_ps(_mm256_castsi256_ps(v1.vec_), _mm256_castsi256_ps(v2.vec_)));
        return temp;
    }

    friend force_inline simd_vec<unsigned, 8> vectorcall operator+(const simd_vec<unsigned, 8> v1,
                                                                   const simd_vec<unsigned, 8> v2) {
        simd_vec<unsigned, 8> ret;
#if defined(USE_AVX2) || defined(USE_AVX512)
        ret.vec_ = _mm256_add_epi32(v1.vec_, v2.vec_);
#elif defined(_MSC_VER) && !defined(__clang__)
        UNROLLED_FOR(i, 8, { ret.vec_.m256i_u32[i] = v1.vec_.m256i_u32[i] + v2.vec_.m256i_u32[i]; })
#else
        alignas(32) unsigned comp1[8], comp2[8];
        _mm256_store_si256((__m256i *)comp1, v1.vec_);
        _mm256_store_si256((__m256i *)comp2, v2.vec_);
        UNROLLED_FOR(i, 8, { comp1[i] += comp2[i]; })
        ret.vec_ = _mm256_load_si256((const __m256i *)comp1);
#endif
        return ret;
    }

    friend force_inline simd_vec<unsigned, 8> vectorcall operator-(const simd_vec<unsigned, 8> v1,
                                                                   const simd_vec<unsigned, 8> v2) {
        simd_vec<unsigned, 8> ret;
#if defined(USE_AVX2) || defined(USE_AVX512)
        ret.vec_ = _mm256_sub_epi32(v1.vec_, v2.vec_);
#elif defined(_MSC_VER) && !defined(__clang__)
        UNROLLED_FOR(i, 8, { ret.vec_.m256i_u32[i] = v1.vec_.m256i_u32[i] - v2.vec_.m256i_u32[i]; })
#else
        alignas(32) unsigned comp1[8], comp2[8];
        _mm256_store_si256((__m256i *)comp1, v1.vec_);
        _mm256_store_si256((__m256i *)comp2, v2.vec_);
        UNROLLED_FOR(i, 8, { comp1[i] -= comp2[i]; })
        ret.vec_ = _mm256_load_si256((const __m256i *)comp1);
#endif
        return ret;
    }

    friend force_inline simd_vec<unsigned, 8> vectorcall operator*(const simd_vec<unsigned, 8> v1,
                                                                   const simd_vec<unsigned, 8> v2) {
        simd_vec<unsigned, 8> ret;
#if 0 // defined(USE_AVX2) || defined(USE_AVX512)
        ret.vec_ = _mm256_mullo_epi32(v1.vec_, v2.vec_);
#elif defined(_MSC_VER) && !defined(__clang__)
        UNROLLED_FOR(i, 8, { ret.vec_.m256i_u32[i] = v1.vec_.m256i_u32[i] * v2.vec_.m256i_u32[i]; })
#else
        alignas(32) unsigned comp1[8], comp2[8];
        _mm256_store_si256((__m256i *)comp1, v1.vec_);
        _mm256_store_si256((__m256i *)comp2, v2.vec_);
        UNROLLED_FOR(i, 8, { comp1[i] *= comp2[i]; })
        ret.vec_ = _mm256_load_si256((const __m256i *)comp1);
#endif
        return ret;
    }

    friend force_inline simd_vec<unsigned, 8> vectorcall operator/(const simd_vec<unsigned, 8> v1,
                                                                   const simd_vec<unsigned, 8> v2) {
        simd_vec<unsigned, 8> ret;
#if defined(_MSC_VER) && !defined(__clang__)
        UNROLLED_FOR(i, 8, { ret.vec_.m256i_u32[i] = (v1.vec_.m256i_u32[i] / v2.vec_.m256i_u32[i]); })
#else
        alignas(32) unsigned comp1[8], comp2[8];
        _mm256_store_si256((__m256i *)comp1, v1.vec_);
        _mm256_store_si256((__m256i *)comp2, v2.vec_);
        UNROLLED_FOR(i, 8, { comp1[i] /= comp2[i]; })
        ret.vec_ = _mm256_load_si256((const __m256i *)comp1);
#endif
        return ret;
    }

    friend force_inline simd_vec<unsigned, 8> vectorcall operator>>(const simd_vec<unsigned, 8> v1,
                                                                    const simd_vec<unsigned, 8> v2) {
        simd_vec<unsigned, 8> ret;
#if defined(USE_AVX2) || defined(USE_AVX512)
        ret.vec_ = _mm256_srlv_epi32(v1.vec_, v2.vec_);
#elif defined(_MSC_VER) && !defined(__clang__)
        UNROLLED_FOR(i, 8, { ret.vec_.m256i_u32[i] = (v1.vec_.m256i_u32[i] >> v2.vec_.m256i_u32[i]); })
#else
        alignas(32) unsigned comp1[8], comp2[8];
        _mm256_store_si256((__m256i *)comp1, v1.vec_);
        _mm256_store_si256((__m256i *)comp2, v2.vec_);
        UNROLLED_FOR(i, 8, { comp1[i] = reinterpret_cast<const unsigned &>(comp1[i]) >> comp2[i]; })
        ret.vec_ = _mm256_load_si256((const __m256i *)comp1);
#endif
        return ret;
    }

    friend force_inline simd_vec<unsigned, 8> vectorcall operator>>(const simd_vec<unsigned, 8> v1, const unsigned v2) {
        simd_vec<unsigned, 8> ret;
#if defined(USE_AVX2) || defined(USE_AVX512)
        ret.vec_ = _mm256_srli_epi32(v1.vec_, v2);
#elif defined(_MSC_VER) && !defined(__clang__)
        UNROLLED_FOR(i, 8, { ret.vec_.m256i_u32[i] = (v1.vec_.m256i_u32[i] >> v2); })
#else
        alignas(32) unsigned comp1[8];
        _mm256_store_si256((__m256i *)comp1, v1.vec_);
        UNROLLED_FOR(i, 8, { comp1[i] = reinterpret_cast<const unsigned &>(comp1[i]) >> v2; })
        ret.vec_ = _mm256_load_si256((const __m256i *)comp1);
#endif
        return ret;
    }

    friend force_inline simd_vec<unsigned, 8> vectorcall operator<<(const simd_vec<unsigned, 8> v1,
                                                                    const simd_vec<unsigned, 8> v2) {
        simd_vec<unsigned, 8> ret;
#if defined(USE_AVX2) || defined(USE_AVX512)
        ret.vec_ = _mm256_sllv_epi32(v1.vec_, v2.vec_);
#elif defined(_MSC_VER) && !defined(__clang__)
        UNROLLED_FOR(i, 8, { ret.vec_.m256i_u32[i] = (v1.vec_.m256i_u32[i] << v2.vec_.m256i_u32[i]); })
#else
        alignas(32) unsigned comp1[8], comp2[8];
        _mm256_store_si256((__m256i *)comp1, v1.vec_);
        _mm256_store_si256((__m256i *)comp2, v2.vec_);
        UNROLLED_FOR(i, 8, { comp1[i] = comp1[i] << comp2[i]; })
        ret.vec_ = _mm256_load_si256((const __m256i *)comp1);
#endif
        return ret;
    }

    friend force_inline simd_vec<unsigned, 8> vectorcall operator<<(const simd_vec<unsigned, 8> v1, const unsigned v2) {
        simd_vec<unsigned, 8> ret;
#if defined(USE_AVX2) || defined(USE_AVX512)
        ret.vec_ = _mm256_slli_epi32(v1.vec_, v2);
#elif defined(_MSC_VER) && !defined(__clang__)
        UNROLLED_FOR(i, 8, { ret.vec_.m256i_u32[i] = (v1.vec_.m256i_u32[i] << v2); })
#else
        alignas(32) unsigned comp1[8];
        _mm256_store_si256((__m256i *)comp1, v1.vec_);
        UNROLLED_FOR(i, 8, { comp1[i] = comp1[i] << v2; })
        ret.vec_ = _mm256_load_si256((const __m256i *)comp1);
#endif
        return ret;
    }

    force_inline simd_vec<unsigned, 8> operator~() const {
        simd_vec<unsigned, 8> ret;
#if defined(USE_AVX2) || defined(USE_AVX512)
        ret.vec_ = _mm256_andnot_si256(vec_, _mm256_set1_epi32(~0));
#elif defined(_MSC_VER) && !defined(__clang__)
        UNROLLED_FOR(i, 8, { ret.vec_.m256i_u32[i] = ~vec_.m256i_u32[i]; })
#else
        alignas(32) unsigned comp[8];
        _mm256_store_si256((__m256i *)comp, vec_);
        UNROLLED_FOR(i, 8, { comp[i] = ~comp[i]; })
        ret.vec_ = _mm256_load_si256((const __m256i *)comp);
#endif
        return ret;
    }

    friend force_inline bool vectorcall is_equal(const simd_vec<unsigned, 8> v1, const simd_vec<unsigned, 8> v2) {
#if defined(USE_AVX2) || defined(USE_AVX512)
        __m256i vcmp = _mm256_cmpeq_epi32(v1.vec_, v2.vec_);
        return (_mm256_movemask_epi8(vcmp) == 0xffffffff);
#elif defined(_MSC_VER) && !defined(__clang__)
        bool ret = true;

        UNROLLED_FOR(i, 8, { ret &= (v1.vec_.m256i_u32[i] == v2.vec_.m256i_u32[i]); })

        return ret;
#else
        bool ret = true;

        alignas(32) unsigned comp1[8], comp2[8];
        _mm256_store_si256((__m256i *)comp1, v1.vec_);
        _mm256_store_si256((__m256i *)comp2, v2.vec_);
        UNROLLED_FOR(i, 8, { ret &= (comp1[i] == comp2[i]); })

        return ret;
#endif
    }

#if defined(USE_AVX2) || defined(USE_AVX512)
    friend force_inline simd_vec<unsigned, 8> vectorcall inclusive_scan(simd_vec<unsigned, 8> v1) {
        v1.vec_ = _mm256_add_epi32(v1.vec_, _mm256_slli_si256(v1.vec_, 4));
        v1.vec_ = _mm256_add_epi32(v1.vec_, _mm256_slli_si256(v1.vec_, 8));

        __m256i temp = _mm256_shuffle_epi32(v1.vec_, _MM_SHUFFLE(3, 3, 3, 3));
        temp = _mm256_permute2x128_si256(_mm256_setzero_si256(), temp, 0x20);

        v1.vec_ = _mm256_add_epi32(v1.vec_, temp);

        return v1;
    }
#endif

#if defined(USE_AVX2) || defined(USE_AVX512)
    // friend force_inline simd_vec<float, 8> vectorcall gather(const float *base_addr, simd_vec<int, 8> vindex);
    // friend force_inline simd_vec<float, 8> vectorcall gather(simd_vec<float, 8> src, const float *base_addr,
    //                                                          simd_vec<int, 8> mask, simd_vec<int, 8> vindex);
    friend force_inline simd_vec<unsigned, 8> vectorcall gather(const unsigned *base_addr, simd_vec<int, 8> vindex);
    friend force_inline simd_vec<unsigned, 8> vectorcall gather(simd_vec<unsigned, 8> src, const unsigned *base_addr,
                                                                simd_vec<unsigned, 8> mask, simd_vec<int, 8> vindex);
#endif

    template <typename U>
    friend force_inline simd_vec<float, 8> vectorcall select(const simd_vec<U, 8> mask, const simd_vec<float, 8> vec1,
                                                             const simd_vec<float, 8> vec2);
    template <typename U>
    friend force_inline simd_vec<int, 8> vectorcall select(const simd_vec<U, 8> mask, const simd_vec<int, 8> vec1,
                                                           const simd_vec<int, 8> vec2);
    template <typename U>
    friend force_inline simd_vec<unsigned, 8> vectorcall select(const simd_vec<U, 8> mask,
                                                                const simd_vec<unsigned, 8> vec1,
                                                                const simd_vec<unsigned, 8> vec2);

#ifndef NDEBUG
    friend void vectorcall __assert_valid_mask(const simd_vec<unsigned, 8> mask) {
        UNROLLED_FOR(i, 8, {
            const int val = mask.get<i>();
            assert(val == 0 || val == 0xffffffff);
        })
    }
#endif

    friend force_inline const unsigned *value_ptr(const simd_vec<unsigned, 8> &v1) {
        return reinterpret_cast<const unsigned *>(&v1.vec_);
    }
    friend force_inline unsigned *value_ptr(simd_vec<unsigned, 8> &v1) {
        return reinterpret_cast<unsigned *>(&v1.vec_);
    }

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
#if defined(USE_AVX2) || defined(USE_AVX512)
    simd_vec<float, 8> ret;
    ret.vec_ = _mm256_castsi256_ps(_mm256_andnot_si256(_mm256_castps_si256(vec_), _mm256_set1_epi32(~0)));
    return ret;
#else
    alignas(32) uint32_t temp[8];
    _mm256_store_ps((float *)temp, vec_);
    UNROLLED_FOR(i, 8, { temp[i] = ~temp[i]; })
    return simd_vec<float, 8>{(const float *)temp, simd_mem_aligned};
#endif
}

force_inline simd_vec<float, 8> simd_vec<float, 8>::operator-() const {
    simd_vec<float, 8> temp;
    __m256 m = _mm256_set1_ps(-0.0f);
    temp.vec_ = _mm256_xor_ps(vec_, m);
    return temp;
}

force_inline simd_vec<float, 8>::operator simd_vec<int, 8>() const {
    simd_vec<int, 8> ret;
    ret.vec_ = _mm256_cvttps_epi32(vec_);
    return ret;
}

force_inline simd_vec<float, 8>::operator simd_vec<unsigned, 8>() const {
    simd_vec<unsigned, 8> ret;
    ret.vec_ = _mm256_cvttps_epi32(vec_);
    return ret;
}

force_inline simd_vec<int, 8>::operator simd_vec<unsigned, 8>() const {
    simd_vec<unsigned, 8> ret;
    ret.vec_ = vec_;
    return ret;
}

force_inline simd_vec<float, 8> simd_vec<float, 8>::sqrt() const {
    simd_vec<float, 8> temp;
    temp.vec_ = _mm256_sqrt_ps(vec_);
    return temp;
}

force_inline simd_vec<float, 8> simd_vec<float, 8>::log() const {
    alignas(32) float comp[8];
    _mm256_store_ps(comp, vec_);
    UNROLLED_FOR(i, 8, { comp[i] = logf(comp[i]); })
    return simd_vec<float, 8>{comp, simd_mem_aligned};
}

force_inline simd_vec<float, 8> vectorcall min(const simd_vec<float, 8> v1, const simd_vec<float, 8> v2) {
    simd_vec<float, 8> temp;
    temp.vec_ = _mm256_min_ps(v1.vec_, v2.vec_);
    return temp;
}

force_inline simd_vec<float, 8> vectorcall max(const simd_vec<float, 8> v1, const simd_vec<float, 8> v2) {
    simd_vec<float, 8> temp;
    temp.vec_ = _mm256_max_ps(v1.vec_, v2.vec_);
    return temp;
}

force_inline simd_vec<float, 8> vectorcall clamp(const simd_vec<float, 8> v1, const simd_vec<float, 8> min,
                                                 const simd_vec<float, 8> max) {
    simd_vec<float, 8> ret;
    ret.vec_ = _mm256_max_ps(min.vec_, _mm256_min_ps(v1.vec_, max.vec_));
    return ret;
}

force_inline simd_vec<float, 8> vectorcall and_not(const simd_vec<float, 8> v1, const simd_vec<float, 8> v2) {
    simd_vec<float, 8> temp;
    temp.vec_ = _mm256_andnot_ps(v1.vec_, v2.vec_);
    return temp;
}

force_inline simd_vec<float, 8> vectorcall floor(const simd_vec<float, 8> v1) {
    simd_vec<float, 8> temp;
    temp.vec_ = _mm256_floor_ps(v1.vec_);
    return temp;
}

force_inline simd_vec<float, 8> vectorcall ceil(const simd_vec<float, 8> v1) {
    simd_vec<float, 8> temp;
    temp.vec_ = _mm256_ceil_ps(v1.vec_);
    return temp;
}

force_inline simd_vec<float, 8> vectorcall operator&(const simd_vec<float, 8> v1, const simd_vec<float, 8> v2) {
    simd_vec<float, 8> temp;
    temp.vec_ = _mm256_and_ps(v1.vec_, v2.vec_);
    return temp;
}

force_inline simd_vec<float, 8> vectorcall operator|(const simd_vec<float, 8> v1, const simd_vec<float, 8> v2) {
    simd_vec<float, 8> temp;
    temp.vec_ = _mm256_or_ps(v1.vec_, v2.vec_);
    return temp;
}

force_inline simd_vec<float, 8> vectorcall operator^(const simd_vec<float, 8> v1, const simd_vec<float, 8> v2) {
    simd_vec<float, 8> temp;
    temp.vec_ = _mm256_xor_ps(v1.vec_, v2.vec_);
    return temp;
}

force_inline simd_vec<float, 8> vectorcall operator+(const simd_vec<float, 8> v1, const simd_vec<float, 8> v2) {
    simd_vec<float, 8> temp;
    temp.vec_ = _mm256_add_ps(v1.vec_, v2.vec_);
    return temp;
}

force_inline simd_vec<float, 8> vectorcall operator-(const simd_vec<float, 8> v1, const simd_vec<float, 8> v2) {
    simd_vec<float, 8> temp;
    temp.vec_ = _mm256_sub_ps(v1.vec_, v2.vec_);
    return temp;
}

force_inline simd_vec<float, 8> vectorcall operator*(const simd_vec<float, 8> v1, const simd_vec<float, 8> v2) {
    simd_vec<float, 8> temp;
    temp.vec_ = _mm256_mul_ps(v1.vec_, v2.vec_);
    return temp;
}

force_inline simd_vec<float, 8> vectorcall operator/(const simd_vec<float, 8> v1, const simd_vec<float, 8> v2) {
    simd_vec<float, 8> temp;
    temp.vec_ = _mm256_div_ps(v1.vec_, v2.vec_);
    return temp;
}

force_inline simd_vec<float, 8> vectorcall operator<(const simd_vec<float, 8> v1, const simd_vec<float, 8> v2) {
    simd_vec<float, 8> ret;
    ret.vec_ = _mm256_cmp_ps(v1.vec_, v2.vec_, _CMP_LT_OS);
    return ret;
}

force_inline simd_vec<float, 8> vectorcall operator<=(const simd_vec<float, 8> v1, const simd_vec<float, 8> v2) {
    simd_vec<float, 8> ret;
    ret.vec_ = _mm256_cmp_ps(v1.vec_, v2.vec_, _CMP_LE_OS);
    return ret;
}

force_inline simd_vec<float, 8> vectorcall operator>(const simd_vec<float, 8> v1, const simd_vec<float, 8> v2) {
    simd_vec<float, 8> ret;
    ret.vec_ = _mm256_cmp_ps(v1.vec_, v2.vec_, _CMP_GT_OS);
    return ret;
}

force_inline simd_vec<float, 8> vectorcall operator>=(const simd_vec<float, 8> v1, const simd_vec<float, 8> v2) {
    simd_vec<float, 8> ret;
    ret.vec_ = _mm256_cmp_ps(v1.vec_, v2.vec_, _CMP_GE_OS);
    return ret;
}

force_inline simd_vec<float, 8> vectorcall operator==(const simd_vec<float, 8> v1, const simd_vec<float, 8> v2) {
    simd_vec<float, 8> ret;
    ret.vec_ = _mm256_cmp_ps(v1.vec_, v2.vec_, _CMP_EQ_OS);
    return ret;
}

force_inline simd_vec<float, 8> vectorcall operator!=(const simd_vec<float, 8> v1, const simd_vec<float, 8> v2) {
    simd_vec<float, 8> ret;
    ret.vec_ = _mm256_cmp_ps(v1.vec_, v2.vec_, _CMP_NEQ_OS);
    return ret;
}

force_inline simd_vec<float, 8> vectorcall pow(const simd_vec<float, 8> v1, const simd_vec<float, 8> v2) {
    alignas(32) float comp1[8], comp2[8];
    _mm256_store_ps(comp1, v1.vec_);
    _mm256_store_ps(comp2, v2.vec_);
    UNROLLED_FOR(i, 8, { comp1[i] = powf(comp1[i], comp2[i]); })
    return simd_vec<float, 8>{comp1, simd_mem_aligned};
}

force_inline simd_vec<float, 8> vectorcall normalize(const simd_vec<float, 8> v1) { return v1 / v1.length(); }

force_inline simd_vec<float, 8> vectorcall normalize_len(const simd_vec<float, 8> v1, float &out_len) {
    return v1 / (out_len = v1.length());
}

#ifdef USE_FMA
force_inline simd_vec<float, 8> vectorcall fmadd(const simd_vec<float, 8> a, const simd_vec<float, 8> b,
                                                 const simd_vec<float, 8> c) {
    simd_vec<float, 8> ret;
    ret.vec_ = _mm256_fmadd_ps(a.vec_, b.vec_, c.vec_);
    return ret;
}

force_inline simd_vec<float, 8> vectorcall fmsub(const simd_vec<float, 8> a, const simd_vec<float, 8> b,
                                                 const simd_vec<float, 8> c) {
    simd_vec<float, 8> ret;
    ret.vec_ = _mm256_fmsub_ps(a.vec_, b.vec_, c.vec_);
    return ret;
}
#endif // USE_FMA

#if defined(USE_AVX2) || defined(USE_AVX512)
force_inline simd_vec<float, 8> vectorcall gather(const float *base_addr, const simd_vec<int, 8> vindex) {
    simd_vec<float, 8> ret;
    ret.vec_ = _mm256_i32gather_ps(base_addr, vindex.vec_, sizeof(float));
    return ret;
}

force_inline simd_vec<float, 8> vectorcall gather(simd_vec<float, 8> src, const float *base_addr, simd_vec<int, 8> mask,
                                                  simd_vec<int, 8> vindex) {
    simd_vec<float, 8> ret;
    ret.vec_ =
        _mm256_mask_i32gather_ps(src.vec_, base_addr, vindex.vec_, _mm256_castsi256_ps(mask.vec_), sizeof(float));
    return ret;
}

force_inline simd_vec<int, 8> vectorcall gather(const int *base_addr, const simd_vec<int, 8> vindex) {
    simd_vec<int, 8> ret;
    ret.vec_ = _mm256_i32gather_epi32(base_addr, vindex.vec_, sizeof(int));
    return ret;
}

force_inline simd_vec<int, 8> vectorcall gather(simd_vec<int, 8> src, const int *base_addr, simd_vec<int, 8> mask,
                                                simd_vec<int, 8> vindex) {
    simd_vec<int, 8> ret;
    ret.vec_ = _mm256_mask_i32gather_epi32(src.vec_, base_addr, vindex.vec_, mask.vec_, sizeof(int));
    return ret;
}

force_inline simd_vec<unsigned, 8> vectorcall gather(const unsigned *base_addr, const simd_vec<int, 8> vindex) {
    simd_vec<unsigned, 8> ret;
    ret.vec_ = _mm256_i32gather_epi32(reinterpret_cast<const int *>(base_addr), vindex.vec_, sizeof(int));
    return ret;
}

force_inline simd_vec<unsigned, 8> vectorcall gather(simd_vec<unsigned, 8> src, const unsigned *base_addr,
                                                     simd_vec<unsigned, 8> mask, simd_vec<int, 8> vindex) {
    simd_vec<unsigned, 8> ret;
    ret.vec_ = _mm256_mask_i32gather_epi32(src.vec_, reinterpret_cast<const int *>(base_addr), vindex.vec_, mask.vec_,
                                           sizeof(unsigned));
    return ret;
}
#endif

template <typename U>
force_inline simd_vec<float, 8> vectorcall select(const simd_vec<U, 8> mask, const simd_vec<float, 8> vec1,
                                                  const simd_vec<float, 8> vec2) {
    validate_mask(mask);
    simd_vec<float, 8> ret;
    ret.vec_ = _mm256_blendv_ps(vec2.vec_, vec1.vec_, _mm_cast<__m256>(mask.vec_));
    return ret;
}

template <typename U>
force_inline simd_vec<int, 8> vectorcall select(const simd_vec<U, 8> mask, const simd_vec<int, 8> vec1,
                                                const simd_vec<int, 8> vec2) {
    validate_mask(mask);
    simd_vec<int, 8> ret;
    ret.vec_ = _mm256_castps_si256(
        _mm256_blendv_ps(_mm256_castsi256_ps(vec2.vec_), _mm256_castsi256_ps(vec1.vec_), _mm_cast<__m256>(mask.vec_)));
    return ret;
}

template <typename U>
force_inline simd_vec<unsigned, 8> vectorcall select(const simd_vec<U, 8> mask, const simd_vec<unsigned, 8> vec1,
                                                     const simd_vec<unsigned, 8> vec2) {
    validate_mask(mask);
    simd_vec<unsigned, 8> ret;
    ret.vec_ = _mm256_castps_si256(
        _mm256_blendv_ps(_mm256_castsi256_ps(vec2.vec_), _mm256_castsi256_ps(vec1.vec_), _mm_cast<__m256>(mask.vec_)));
    return ret;
}

} // namespace NS
} // namespace Ray

#pragma warning(pop)

#undef validate_mask

#ifdef __GNUC__
#pragma GCC pop_options
#pragma clang attribute pop
#endif
