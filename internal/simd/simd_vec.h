#pragma once

#include <cstring>

#ifndef INSTANTIATION_ID
#define INSTANTIATION_ID 0
#endif

enum simd_mem_aligned_tag { simd_mem_aligned };

template <typename T, int S, int I = INSTANTIATION_ID>
class simd_vec {
    T vec_[S];
public:
    simd_vec() = default;
    simd_vec(T f) {
        for (auto &v : vec_) {
            v = f;
        }
    }
    template <typename... Tail>
    simd_vec(typename std::enable_if<sizeof...(Tail)+1 == S, T>::type head, Tail... tail)
        : vec_{ head, T(tail)... } {
    }
    simd_vec(const T *f) {
        memcpy(&vec_, f, S * sizeof(T));
    }
    simd_vec(const T *f, simd_mem_aligned_tag) {
        memcpy(&vec_, f, S * sizeof(T));
    }

    T &operator[](int i) { return vec_[i]; }
    T operator[](int i) const { return vec_[i]; }

    simd_vec<T, S> &operator+=(const simd_vec<T, S> &rhs) {
        for (int i = 0; i < S; i++) {
            vec_[i] += rhs.vec_[i];
        }
        return *this;
    }

    simd_vec<T, S> &operator-=(const simd_vec<T, S> &rhs) {
        for (int i = 0; i < S; i++) {
            vec_[i] -= rhs.vec_[i];
        }
        return *this;
    }

    simd_vec<T, S> &operator*=(const simd_vec<T, S> &rhs) {
        for (int i = 0; i < S; i++) {
            vec_[i] *= rhs.vec_[i];
        }
        return *this;
    }

    simd_vec<T, S> &operator/=(const simd_vec<T, S> &rhs) {
        for (int i = 0; i < S; i++) {
            vec_[i] /= rhs.vec_[i];
        }
        return *this;
    }

    simd_vec<T, S> operator<(const simd_vec<T, S> &rhs) const {
        T set;
        memset(&set, 0xFF, sizeof(T));
        simd_vec<T, S> ret;
        for (int i = 0; i < S; i++) {
            ret.vec_[i] = vec_[i] < rhs.vec_[i] ? set : 0;
        }
        return ret;
    }

    void copy_to(T *f) const {
        memcpy(f, &vec_[0], S * sizeof(T));
    }

    void copy_to(T *f, simd_mem_aligned_tag) const {
        memcpy(f, &vec_[0], S * sizeof(T));
    }

    static const size_t alignment = 1;

    static int size() { return S; }
    static int native_count() { return S; }
    static bool is_native() { return native_count() == 1; }
};

template <typename T, int S, int I = INSTANTIATION_ID>
inline simd_vec<T, S> operator+(const simd_vec<T, S> &v1, const simd_vec<T, S> &v2) { simd_vec<T, S> temp = v1; temp += v2; return temp; }

template <typename T, int S, int I = INSTANTIATION_ID>
inline simd_vec<T, S> operator-(const simd_vec<T, S> &v1, const simd_vec<T, S> &v2) { simd_vec<T, S> temp = v1; temp -= v2; return temp; }

template <typename T, int S, int I = INSTANTIATION_ID>
inline simd_vec<T, S> operator*(const simd_vec<T, S> &v1, const simd_vec<T, S> &v2) { simd_vec<T, S> temp = v1; temp *= v2; return temp; }

template <typename T, int S, int I = INSTANTIATION_ID>
inline simd_vec<T, S> operator/(const simd_vec<T, S> &v1, const simd_vec<T, S> &v2) { simd_vec<T, S> temp = v1; temp /= v2; return temp; }

template <typename T, int S, int I = INSTANTIATION_ID>
inline simd_vec<T, S> sqrt(const simd_vec<T, S> &v1) {
    simd_vec<T, S> temp;
    for (int i = 0; i < S; i++) {
        temp[i] = sqrt(v1[i]);
    }
    return temp;
}

template <int S>
using simd_fvec = simd_vec<float, S>;
using simd_fvec4 = simd_fvec<4>;
using simd_fvec8 = simd_fvec<8>;
using simd_fvec16 = simd_fvec<16>;

template <int S>
using simd_ivec = simd_vec<int, S>;
using simd_ivec4 = simd_ivec<4>;
using simd_ivec8 = simd_ivec<8>;
using simd_ivec16 = simd_ivec<16>;

#if defined(USE_SSE)
#include "simd_vec_sse.h"
#elif defined (USE_AVX)
#include "simd_vec_avx.h"
#else
using native_simd_fvec = simd_fvec<1>;
using native_simd_ivec = simd_ivec<1>;
#endif
