#pragma once

#include "simd_vec_sse.h"

#include <immintrin.h>

#ifdef __GNUC__
#pragma GCC push_options
#pragma GCC target ("avx")
#endif

template <int S>
class simd_vec<typename std::enable_if<S % 8 == 0, float>::type, S> {
    union {
        __m256 vec_[S / 8];
        float comp_[S];
    };
public:
    simd_vec() = default;
    simd_vec(float f) {
        for (int i = 0; i < S/8; i++) {
            vec_[i] = _mm256_set1_ps(f);
        }
    }
    template <typename... Tail>
    simd_vec(typename std::enable_if<sizeof...(Tail)+1 == S, float>::type head, Tail... tail) {
        const float _tail[] = { tail... };
        vec_[0] = _mm256_setr_ps(head, _tail[0], _tail[1], _tail[2], _tail[3], _tail[4], _tail[5], _tail[6]);
        for (int i = 7; i < S - 1; i += 8) {
            vec_[(i + 1) / 8] = _mm256_setr_ps(_tail[i], _tail[i + 1], _tail[i + 2], _tail[i + 3], _tail[i + 4], _tail[i + 5], _tail[i + 6], _tail[i + 7]);
        }
    }
    simd_vec(const float *f) {
        for (int i = 0; i < S / 8; i++) {
            vec_[i] = _mm256_loadu_ps(f);
            f += 8;
        }
    }
    simd_vec(const float *f, simd_mem_aligned_tag) {
        for (int i = 0; i < S / 8; i++) {
            vec_[i] = _mm256_load_ps(f);
            f += 8;
        }
    }

    float &operator[](int i) { return comp_[i]; }
    float operator[](int i) const { return comp_[i]; }

    simd_vec<float, S> &operator+=(const simd_vec<float, S> &rhs) {
        for (int i = 0; i < S / 8; i++) {
            vec_[i] = _mm256_add_ps(vec_[i], rhs.vec_[i]);
        }
        return *this;
    }

    simd_vec<float, S> &operator-=(const simd_vec<float, S> &rhs) {
        for (int i = 0; i < S / 8; i++) {
            vec_[i] = _mm256_sub_ps(vec_[i], rhs.vec_[i]);
        }
        return *this;
    }

    simd_vec<float, S> &operator*=(const simd_vec<float, S> &rhs) {
        for (int i = 0; i < S / 8; i++) {
            vec_[i] = _mm256_mul_ps(vec_[i], rhs.vec_[i]);
        }
        return *this;
        }

    simd_vec<float, S> &operator/=(const simd_vec<float, S> &rhs) {
        for (int i = 0; i < S / 8; i++) {
            vec_[i] = _mm256_div_ps(vec_[i], rhs.vec_[i]);
        }
        return *this;
    }

    void copy_to(float *f) const {
        for (int i = 0; i < S / 8; i++) {
            _mm256_storeu_ps(f, vec_[i]);
            f += 8;
        }
    }

    void copy_to(float *f, simd_mem_aligned_tag) const {
        for (int i = 0; i < S / 8; i++) {
            _mm256_store_ps(f, vec_[i]);
            f += 8;
        }
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
public:
    simd_vec() = default;
    simd_vec(int f) {
        for (int i = 0; i < S / 8; i++) {
            vec_[i] = _mm256_set1_epi32(f);
        }
    }
    template <typename... Tail>
    simd_vec(typename std::enable_if<sizeof...(Tail)+1 == S, int>::type head, Tail... tail) {
        const int _tail[] = { tail... };
        vec_[0] = _mm256_setr_epi32(head, _tail[0], _tail[1], _tail[2], _tail[3], _tail[4], _tail[5], _tail[6]);
        for (int i = 7; i < S - 1; i += 8) {
            vec_[(i + 1) / 8] = _mm256_setr_epi32(_tail[i], _tail[i + 1], _tail[i + 2], _tail[i + 3], _tail[i + 4], _tail[i + 5], _tail[i + 6], _tail[i + 7]);
        }
    }
    simd_vec(const int *f) {
        for (int i = 0; i < S / 8; i++) {
            vec_[i] = _mm256_loadu_si128((const __m256i *)f);
            f += 4;
        }
    }
    simd_vec(const int *f, simd_mem_aligned_tag) {
        for (int i = 0; i < S / 8; i++) {
            vec_[i] = _mm256_load_si256((const __m256i *)f);
            f += 4;
        }
    }

    int &operator[](int i) { return comp_[i]; }
    int operator[](int i) const { return comp_[i]; }

    simd_vec<int, S> &operator+=(const simd_vec<int, S> &rhs) {
        for (int i = 0; i < S / 8; i++) {
            vec_[i] = _mm256_add_epi32(vec_[i], rhs.vec_[i]);
        }
        return *this;
    }

    simd_vec<int, S> &operator-=(const simd_vec<int, S> &rhs) {
        for (int i = 0; i < S / 8; i++) {
            vec_[i] = _mm256_sub_epi32(vec_[i], rhs.vec_[i]);
        }
        return *this;
    }

    simd_vec<int, S> &operator*=(const simd_vec<int, S> &rhs) {
        for (int i = 0; i < S; i += 8) {
            comp_[i + 0] = comp_[i + 0] * rhs.comp_[i + 0];
            comp_[i + 1] = comp_[i + 1] * rhs.comp_[i + 1];
            comp_[i + 2] = comp_[i + 2] * rhs.comp_[i + 2];
            comp_[i + 3] = comp_[i + 3] * rhs.comp_[i + 3];
            comp_[i + 4] = comp_[i + 4] * rhs.comp_[i + 4];
            comp_[i + 5] = comp_[i + 5] * rhs.comp_[i + 5];
            comp_[i + 6] = comp_[i + 6] * rhs.comp_[i + 6];
            comp_[i + 7] = comp_[i + 7] * rhs.comp_[i + 7];
        }
        return *this;
    }

    simd_vec<int, S> &operator/=(const simd_vec<int, S> &rhs) {
        for (int i = 0; i < S; i += 8) {
            comp_[i + 0] = comp_[i + 0] / rhs.comp_[i + 0];
            comp_[i + 1] = comp_[i + 1] / rhs.comp_[i + 1];
            comp_[i + 2] = comp_[i + 2] / rhs.comp_[i + 2];
            comp_[i + 3] = comp_[i + 3] / rhs.comp_[i + 3];
            comp_[i + 4] = comp_[i + 4] / rhs.comp_[i + 4];
            comp_[i + 5] = comp_[i + 5] / rhs.comp_[i + 5];
            comp_[i + 6] = comp_[i + 6] / rhs.comp_[i + 6];
            comp_[i + 7] = comp_[i + 7] / rhs.comp_[i + 7];
        }
        return *this;
    }

    void copy_to(int *f) const {
        for (int i = 0; i < S / 8; i++) {
            _mm256_storeu_si256((__m256i *)f, vec_[i]);
            f += 8;
        }
    }

    void copy_to(int *f, simd_mem_aligned_tag) const {
        for (int i = 0; i < S / 8; i++) {
            _mm256_store_si256((__m256i *)f, vec_[i]);
            f += 8;
        }
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

#ifdef __GNUC__
#pragma GCC pop_options
#endif