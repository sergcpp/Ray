#pragma once

#include "simd_vec.h"

#include <type_traits>

#include <immintrin.h>
#include <xmmintrin.h>

#ifdef __GNUC__
#pragma GCC push_options
#pragma GCC target ("sse2")
#endif

template <int S>
class simd_vec<typename std::enable_if<S % 4 == 0
#if defined(USE_AVX)
    && S % 8 != 0
#endif
    , float>::type, S> {
    union {
        __m128 vec_[S/4];
        float comp_[S];
    };
public:
    simd_vec() = default;
    simd_vec(float f) {
        for (int i = 0; i < S/4; i++) {
            vec_[i] = _mm_set1_ps(f);
        }
    }
    template <typename... Tail>
    simd_vec(typename std::enable_if<sizeof...(Tail)+1 == S, float>::type head, Tail... tail) {
        const float _tail[] = { tail... };
        vec_[0] = _mm_setr_ps(head, _tail[0], _tail[1], _tail[2]);
        for (int i = 3; i < S - 1; i += 4) {
            vec_[(i+1)/4] = _mm_setr_ps(_tail[i], _tail[i+1], _tail[i+2], _tail[i+3]);
        }
    }
    simd_vec(const float *f) {
        for (int i = 0; i < S/4; i++) {
            vec_[i] = _mm_loadu_ps(f);
            f += 4;
        }
    }
    simd_vec(const float *f, simd_mem_aligned_tag) {
        for (int i = 0; i < S/4; i++) {
            vec_[i] = _mm_load_ps(f);
            f += 4;
        }
    }

    float &operator[](int i) { return comp_[i]; }
    float operator[](int i) const { return comp_[i]; }

    simd_vec<float, S> &operator+=(const simd_vec<float, S> &rhs) {
        for (int i = 0; i < S/4; i++) {
            vec_[i] = _mm_add_ps(vec_[i], rhs.vec_[i]);
        }
        return *this;
    }

    simd_vec<float, S> &operator-=(const simd_vec<float, S> &rhs) {
        for (int i = 0; i < S/4; i++) {
            vec_[i] = _mm_sub_ps(vec_[i], rhs.vec_[i]);
        }
        return *this;
    }

    simd_vec<float, S> &operator*=(const simd_vec<float, S> &rhs) {
        for (int i = 0; i < S/4; i++) {
            vec_[i] = _mm_mul_ps(vec_[i], rhs.vec_[i]);
        }
        return *this;
    }

    simd_vec<float, S> &operator/=(const simd_vec<float, S> &rhs) {
        for (int i = 0; i < S/4; i++) {
            vec_[i] = _mm_div_ps(vec_[i], rhs.vec_[i]);
        }
        return *this;
    }

    void copy_to(float *f) const {
        for (int i = 0; i < S/4; i++) {
            _mm_storeu_ps(f, vec_[i]);
            f += 4;
        }
    }

    void copy_to(float *f, simd_mem_aligned_tag) const {
        for (int i = 0; i < S/4; i++) {
            _mm_store_ps(f, vec_[i]);
            f += 4;
        }
    }

    static const size_t alignment = alignof(__m128);

    static int size() { return S; }
    static int native_count() { return S/4; }
    static bool is_native() { return native_count() == 1; }
};

template <int S>
class simd_vec<typename std::enable_if<S % 4 == 0
#if defined(USE_AVX)
    && S % 8 != 0
#endif
    , int>::type, S> {
    union {
        __m128i vec_[S / 4];
        int comp_[S];
    };
public:
    simd_vec() = default;
    simd_vec(int f) {
        for (int i = 0; i < S / 4; i++) {
            vec_[i] = _mm_set1_epi32(f);
        }
    }
    template <typename... Tail>
    simd_vec(typename std::enable_if<sizeof...(Tail)+1 == S, int>::type head, Tail... tail) {
        const int _tail[] = { tail... };
        vec_[0] = _mm_setr_epi32(head, _tail[0], _tail[1], _tail[2]);
        for (int i = 3; i < S - 1; i += 4) {
            vec_[(i + 1) / 4] = _mm_setr_epi32(_tail[i], _tail[i + 1], _tail[i + 2], _tail[i + 3]);
        }
    }
    simd_vec(const int *f) {
        for (int i = 0; i < S / 4; i++) {
            vec_[i] = _mm_loadu_si128((const __m128i *)f);
            f += 4;
        }
    }
    simd_vec(const int *f, simd_mem_aligned_tag) {
        for (int i = 0; i < S / 4; i++) {
            vec_[i] = _mm_load_si128((const __m128i *)f);
            f += 4;
        }
    }

    int &operator[](int i) { return comp_[i]; }
    int operator[](int i) const { return comp_[i]; }

    simd_vec<int, S> &operator+=(const simd_vec<int, S> &rhs) {
        for (int i = 0; i < S / 4; i++) {
            vec_[i] = _mm_add_epi32(vec_[i], rhs.vec_[i]);
        }
        return *this;
    }

    simd_vec<int, S> &operator-=(const simd_vec<int, S> &rhs) {
        for (int i = 0; i < S / 4; i++) {
            vec_[i] = _mm_sub_epi32(vec_[i], rhs.vec_[i]);
        }
        return *this;
    }

    simd_vec<int, S> &operator*=(const simd_vec<int, S> &rhs) {
        for (int i = 0; i < S; i += 4) {
            comp_[i + 0] = comp_[i + 0] * rhs.comp_[i + 0];
            comp_[i + 1] = comp_[i + 1] * rhs.comp_[i + 1];
            comp_[i + 2] = comp_[i + 2] * rhs.comp_[i + 2];
            comp_[i + 3] = comp_[i + 3] * rhs.comp_[i + 3];
        }
        return *this;
    }

    simd_vec<int, S> &operator/=(const simd_vec<int, S> &rhs) {
        for (int i = 0; i < S; i += 4) {
            comp_[i + 0] = comp_[i + 0] / rhs.comp_[i + 0];
            comp_[i + 1] = comp_[i + 1] / rhs.comp_[i + 1];
            comp_[i + 2] = comp_[i + 2] / rhs.comp_[i + 2];
            comp_[i + 3] = comp_[i + 3] / rhs.comp_[i + 3];
        }
        return *this;
    }

    void copy_to(int *f) const {
        for (int i = 0; i < S / 4; i++) {
            _mm_storeu_si128((__m128i *)f, vec_[i]);
            f += 4;
        }
    }

    void copy_to(int *f, simd_mem_aligned_tag) const {
        for (int i = 0; i < S / 4; i++) {
            _mm_store_si128((__m128i *)f, vec_[i]);
            f += 4;
        }
    }

    static const size_t alignment = alignof(__m128i);

    static int size() { return S; }
    static int native_count() { return S / 4; }
    static bool is_native() { return native_count() == 1; }
};

#if defined(USE_SSE)
using native_simd_fvec = simd_fvec<4>;
using native_simd_ivec = simd_ivec<4>;
#endif

#ifdef __GNUC__
#pragma GCC pop_options
#endif