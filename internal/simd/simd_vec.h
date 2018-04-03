
#include <cstring>

#define ITERATE(n, exp)  \
    if (n <= 16) {                              \
        switch (n) {                            \
        case 16: { const int i = n - 16; exp }  \
        case 15: { const int i = n - 15; exp }  \
        case 14: { const int i = n - 14; exp }  \
        case 13: { const int i = n - 13; exp }  \
        case 12: { const int i = n - 12; exp }  \
        case 11: { const int i = n - 11; exp }  \
        case 10: { const int i = n - 10; exp }  \
        case 9: { const int i = n - 9; exp }    \
        case 8: { const int i = n - 8; exp }    \
        case 7: { const int i = n - 7; exp }    \
        case 6: { const int i = n - 6; exp }    \
        case 5: { const int i = n - 5; exp }    \
        case 4: { const int i = n - 4; exp }    \
        case 3: { const int i = n - 3; exp }    \
        case 2: { const int i = n - 2; exp }    \
        case 1: { const int i = n - 1; exp }    \
        }                                       \
    } else                                      \
    for (int i = 0; i < n; i++) {               \
        exp                                     \
    }

namespace ray {
namespace NS {

enum simd_mem_aligned_tag { simd_mem_aligned };

template <typename T, int S>
class simd_vec {
    T vec_[S];
public:
    force_inline simd_vec() = default;
    force_inline simd_vec(T f) {
        ITERATE(S, { vec_[i] = f; })
    }
    template <typename... Tail>
    force_inline simd_vec(typename std::enable_if<sizeof...(Tail)+1 == S, T>::type head, Tail... tail)
        : vec_{ head, T(tail)... } {
    }
    force_inline simd_vec(const T *f) {
        memcpy(&vec_, f, S * sizeof(T));
    }
    force_inline simd_vec(const T *f, simd_mem_aligned_tag) {
        memcpy(&vec_, f, S * sizeof(T));
    }

    force_inline T &operator[](int i) { return vec_[i]; }
    force_inline T operator[](int i) const { return vec_[i]; }

    force_inline simd_vec<T, S> &operator+=(const simd_vec<T, S> &rhs) {
        ITERATE(S, { vec_[i] += rhs.vec_[i]; })
        return *this;
    }

    force_inline simd_vec<T, S> &operator-=(const simd_vec<T, S> &rhs) {
        ITERATE(S, { vec_[i] -= rhs.vec_[i]; })
        return *this;
    }

    force_inline simd_vec<T, S> &operator*=(const simd_vec<T, S> &rhs) {
        ITERATE(S, { vec_[i] *= rhs.vec_[i]; })
        return *this;
    }

    force_inline simd_vec<T, S> &operator/=(const simd_vec<T, S> &rhs) {
        ITERATE(S, { vec_[i] /= rhs.vec_[i]; })
        return *this;
    }

    force_inline simd_vec<T, S> &operator+=(T rhs) {
        ITERATE(S, { vec_[i] += rhs; })
        return *this;
    }

    force_inline simd_vec<T, S> &operator-=(T rhs) {
        ITERATE(S, { vec_[i] -= rhs; })
        return *this;
    }

    force_inline simd_vec<T, S> &operator*=(T rhs) {
        ITERATE(S, { vec_[i] *= rhs; })
        return *this;
    }

    force_inline simd_vec<T, S> &operator/=(T rhs) {
        ITERATE(S, { vec_[i] /= rhs; })
        return *this;
    }

    force_inline simd_vec<T, S> operator-() const {
        simd_vec<T, S> temp;
        ITERATE(S, { temp.vec_[i] = -vec_[i]; })
        return temp;
    }

    force_inline simd_vec<T, S> operator<(const simd_vec<T, S> &rhs) const {
        T set, not_set = T(0);
        memset(&set, 0xFF, sizeof(T));
        simd_vec<T, S> ret;
        ITERATE(S, { ret.vec_[i] = vec_[i] < rhs.vec_[i] ? set : not_set; })
        return ret;
    }

    force_inline simd_vec<T, S> operator<=(const simd_vec<T, S> &rhs) const {
        T set, not_set = T(0);
        memset(&set, 0xFF, sizeof(T));
        simd_vec<T, S> ret;
        ITERATE(S, { ret.vec_[i] = vec_[i] <= rhs.vec_[i] ? set : not_set; })
        return ret;
    }

    force_inline simd_vec<T, S> operator>(const simd_vec<T, S> &rhs) const {
        T set, not_set = T(0);
        memset(&set, 0xFF, sizeof(T));
        simd_vec<T, S> ret;
        ITERATE(S, { ret.vec_[i] = vec_[i] > rhs.vec_[i] ? set : not_set; })
        return ret;
    }

    force_inline simd_vec<T, S> operator>=(const simd_vec<T, S> &rhs) const {
        T set, not_set = T(0);
        memset(&set, 0xFF, sizeof(T));
        simd_vec<T, S> ret;
        ITERATE(S, { ret.vec_[i] = vec_[i] >= rhs.vec_[i] ? set : not_set; })
        return ret;
    }

    force_inline simd_vec<T, S> operator<(T rhs) const {
        T set, not_set = T(0);
        memset(&set, 0xFF, sizeof(T));
        simd_vec<T, S> ret;
        ITERATE(S, { ret.vec_[i] = vec_[i] < rhs ? set : not_set; })
            return ret;
    }

    force_inline simd_vec<T, S> operator<=(T rhs) const {
        T set, not_set = T(0);
        memset(&set, 0xFF, sizeof(T));
        simd_vec<T, S> ret;
        ITERATE(S, { ret.vec_[i] = vec_[i] <= rhs ? set : not_set; })
            return ret;
    }

    force_inline simd_vec<T, S> operator>(T rhs) const {
        T set, not_set = T(0);
        memset(&set, 0xFF, sizeof(T));
        simd_vec<T, S> ret;
        ITERATE(S, { ret.vec_[i] = vec_[i] > rhs ? set : not_set; })
            return ret;
    }

    force_inline simd_vec<T, S> operator>=(T rhs) const {
        T set, not_set = T(0);
        memset(&set, 0xFF, sizeof(T));
        simd_vec<T, S> ret;
        ITERATE(S, { ret.vec_[i] = vec_[i] >= rhs ? set : not_set; })
            return ret;
    }

    force_inline simd_vec<T, S> sqrt() const {
        simd_vec<T, S> temp;
        ITERATE(S, { temp[i] = ::sqrt(vec_[i]); })
        return temp;
    }

    force_inline void copy_to(T *f) const {
        memcpy(f, &vec_[0], S * sizeof(T));
    }

    force_inline void copy_to(T *f, simd_mem_aligned_tag) const {
        memcpy(f, &vec_[0], S * sizeof(T));
    }

    force_inline bool all_zeros() const {
        ITERATE(S, { if (vec_[i] != 0) return false; })
        return true;
    }

    force_inline bool all_zeros(const simd_vec<int, S> &mask) const {
        const auto *src1 = reinterpret_cast<const uint8_t*>(&vec_[0]);
        const auto *src2 = reinterpret_cast<const uint8_t*>(&mask.vec_[0]);

        for (int i = 0; i < S * sizeof(T); i++) {
            if ((src1[i] & src2[i]) != 0) return false;
        }

        return true;
    }

    force_inline bool not_all_zeros() const {
        ITERATE(S, { if (vec_[i] != 0) return true; })
        return false;
    }

    force_inline void blend_to(const simd_vec<T, S> &mask, const simd_vec<T, S> &v1) {
        ITERATE(S, { if (mask.vec_[i] != T(0)) vec_[i] = v1.vec_[i]; })
    }

    force_inline static simd_vec<T, S> min(const simd_vec<T, S> &v1, const simd_vec<T, S> &v2) {
        simd_vec<T, S> temp;
        ITERATE(S, { temp.vec_[i] = std::min(v1.vec_[i], v2.vec_[i]); })
        return temp;
    }

    force_inline static simd_vec<T, S> max(const simd_vec<T, S> &v1, const simd_vec<T, S> &v2) {
        simd_vec<T, S> temp;
        ITERATE(S, { temp.vec_[i] = std::max(v1.vec_[i], v2.vec_[i]); })
        return temp;
    }

    force_inline static simd_vec<T, S> and_not(const simd_vec<T, S> &v1, const simd_vec<T, S> &v2) {
        const auto *src1 = reinterpret_cast<const uint8_t*>(&v1.vec_[0]);
        const auto *src2 = reinterpret_cast<const uint8_t*>(&v2.vec_[0]);

        simd_vec<T, S> ret;

        auto *dst = reinterpret_cast<uint8_t*>(&ret.vec_[0]);

        for (int i = 0; i < S * sizeof(T); i++) {
            dst[i] = (~src1[i]) & src2[i];
        }

        return ret;
    }

    friend force_inline simd_vec<T, S> operator&(const simd_vec<T, S> &v1, const simd_vec<T, S> &v2) {
        const auto *src1 = reinterpret_cast<const uint8_t*>(&v1.vec_[0]);
        const auto *src2 = reinterpret_cast<const uint8_t*>(&v2.vec_[0]);

        simd_vec<T, S> ret;

        auto *dst = reinterpret_cast<uint8_t*>(&ret.vec_[0]);

        for (int i = 0; i < S * sizeof(T); i++) {
            dst[i] = src1[i] & src2[i];
        }

        return ret;
    }

    friend force_inline simd_vec<T, S> operator|(const simd_vec<T, S> &v1, const simd_vec<T, S> &v2) {
        const auto *src1 = reinterpret_cast<const uint8_t*>(&v1.vec_[0]);
        const auto *src2 = reinterpret_cast<const uint8_t*>(&v2.vec_[0]);

        simd_vec<T, S> ret;

        auto *dst = reinterpret_cast<uint8_t*>(&ret.vec_[0]);

        for (int i = 0; i < S * sizeof(T); i++) {
            dst[i] = src1[i] | src2[i];
        }

        return ret;
    }

    friend force_inline simd_vec<T, S> operator^(const simd_vec<T, S> &v1, const simd_vec<T, S> &v2) {
        const auto *src1 = reinterpret_cast<const uint8_t*>(&v1.vec_[0]);
        const auto *src2 = reinterpret_cast<const uint8_t*>(&v2.vec_[0]);

        simd_vec<T, S> ret;

        auto *dst = reinterpret_cast<uint8_t*>(&ret.vec_[0]);

        for (int i = 0; i < S * sizeof(T); i++) {
            dst[i] = src1[i] ^ src2[i];
        }

        return ret;
    }

    friend force_inline simd_vec<T, S> operator+(const simd_vec<T, S> &v1, const simd_vec<T, S> &v2) {
        simd_vec<T, S> ret;
        ITERATE(S, { ret.vec_[i] = v1.vec_[i] + v2.vec_[i]; })
        return ret;
    }

    friend force_inline simd_vec<T, S> operator-(const simd_vec<T, S> &v1, const simd_vec<T, S> &v2) {
        simd_vec<T, S> ret;
        ITERATE(S, { ret.vec_[i] = v1.vec_[i] - v2.vec_[i]; })
        return ret;
    }

    friend force_inline simd_vec<T, S> operator*(const simd_vec<T, S> &v1, const simd_vec<T, S> &v2) {
        simd_vec<T, S> ret;
        ITERATE(S, { ret.vec_[i] = v1.vec_[i] * v2.vec_[i]; })
        return ret;
    }

    friend force_inline simd_vec<T, S> operator/(const simd_vec<T, S> &v1, const simd_vec<T, S> &v2) {
        simd_vec<T, S> ret;
        ITERATE(S, { ret.vec_[i] = v1.vec_[i] / v2.vec_[i]; })
        return ret;
    }

    friend force_inline simd_vec<T, S> operator+(const simd_vec<T, S> &v1, T v2) {
        simd_vec<T, S> ret;
        ITERATE(S, { ret.vec_[i] = v1.vec_[i] + v2; })
        return ret;
    }

    friend force_inline simd_vec<T, S> operator-(const simd_vec<T, S> &v1, T v2) {
        simd_vec<T, S> ret;
        ITERATE(S, { ret.vec_[i] = v1.vec_[i] - v2; })
        return ret;
    }

    friend force_inline simd_vec<T, S> operator*(const simd_vec<T, S> &v1, T v2) {
        simd_vec<T, S> ret;
        ITERATE(S, { ret.vec_[i] = v1.vec_[i] * v2; })
        return ret;
    }

    friend force_inline simd_vec<T, S> operator/(const simd_vec<T, S> &v1, T v2) {
        simd_vec<T, S> ret;
        ITERATE(S, { ret.vec_[i] = v1.vec_[i] / v2; })
        return ret;
    }

    friend force_inline simd_vec<T, S> operator+(T v1, const simd_vec<T, S> &v2) {
        return operator+(v2, v1);
    }

    friend force_inline simd_vec<T, S> operator-(T v1, const simd_vec<T, S> &v2) {
        simd_vec<T, S> ret;
        ITERATE(S, { ret.vec_[i] = v1 - v2.vec_[i]; })
        return ret;
    }

    friend force_inline simd_vec<T, S> operator*(T v1, const simd_vec<T, S> &v2) {
        simd_vec<T, S> ret;
        ITERATE(S, { ret.vec_[i] = v1 * v2.vec_[i]; })
        return ret;
    }

    friend force_inline simd_vec<T, S> operator/(T v1, const simd_vec<T, S> &v2) {
        simd_vec<T, S> ret;
        ITERATE(S, { ret.vec_[i] = v1 / v2.vec_[i]; })
        return ret;
    }

    static const size_t alignment = 1;

    static int size() { return S; }
    static int native_count() { return S; }
    static bool is_native() { return native_count() == 1; }
};

template <typename T, int S>
force_inline simd_vec<T, S> and_not(const simd_vec<T, S> &v1, const simd_vec<T, S> &v2) { return simd_vec<T, S>::and_not(v1, v2); }

template <typename T, int S>
force_inline simd_vec<T, S> sqrt(const simd_vec<T, S> &v1) { return v1.sqrt(); }

template <typename T, int S>
force_inline simd_vec<T, S> min(const simd_vec<T, S> &v1, const simd_vec<T, S> &v2) { return simd_vec<T, S>::min(v1, v2); }

template <typename T, int S>
force_inline simd_vec<T, S> max(const simd_vec<T, S> &v1, const simd_vec<T, S> &v2) { return simd_vec<T, S>::max(v1, v2); }

template <typename T, typename U>
force_inline const T &cast(const U &v1) { return reinterpret_cast<const T&>(v1); }

template <typename T, int S>
class simd_vec_where_helper {
    const simd_vec<T, S> &mask_;
    simd_vec<T, S> &vec_;
public:
    force_inline simd_vec_where_helper(const simd_vec<T, S> &mask, simd_vec<T, S> &vec) : mask_(mask), vec_(vec) {}

    force_inline void operator=(const simd_vec<T, S> &vec) {
        vec_.blend_to(mask_, vec);
    }
};

template <typename T, int S>
force_inline simd_vec_where_helper<T, S> where(const simd_vec<T, S> &mask, simd_vec<T, S> &vec) {
    return { mask, vec };
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

}
}

#if defined(USE_SSE)
#include "simd_vec_sse.h"
#elif defined (USE_AVX)
#include "simd_vec_avx.h"
#else
namespace ray {
namespace NS {
using native_simd_fvec = simd_fvec<1>;
using native_simd_ivec = simd_ivec<1>;
}
}
#endif

#undef ITERATE