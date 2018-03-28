
#include <cstring>

namespace ray {
namespace NS {

enum simd_mem_aligned_tag { simd_mem_aligned };

template <typename T, int S>
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

    simd_vec<T, S> &operator+=(T rhs) {
        for (int i = 0; i < S; i++) {
            vec_[i] += rhs;
        }
        return *this;
    }

    simd_vec<T, S> &operator-=(T rhs) {
        for (int i = 0; i < S; i++) {
            vec_[i] -= rhs;
        }
        return *this;
    }

    simd_vec<T, S> &operator*=(T rhs) {
        for (int i = 0; i < S; i++) {
            vec_[i] *= rhs;
        }
        return *this;
    }

    simd_vec<T, S> &operator/=(T rhs) {
        for (int i = 0; i < S; i++) {
            vec_[i] /= rhs;
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

    simd_vec<T, S> operator<=(const simd_vec<T, S> &rhs) const {
        T set;
        memset(&set, 0xFF, sizeof(T));
        simd_vec<T, S> ret;
        for (int i = 0; i < S; i++) {
            ret.vec_[i] = vec_[i] <= rhs.vec_[i] ? set : 0;
        }
        return ret;
    }

    simd_vec<T, S> operator>(const simd_vec<T, S> &rhs) const {
        T set;
        memset(&set, 0xFF, sizeof(T));
        simd_vec<T, S> ret;
        for (int i = 0; i < S; i++) {
            ret.vec_[i] = vec_[i] > rhs.vec_[i] ? set : 0;
        }
        return ret;
    }

    simd_vec<T, S> operator>=(const simd_vec<T, S> &rhs) const {
        T set;
        memset(&set, 0xFF, sizeof(T));
        simd_vec<T, S> ret;
        for (int i = 0; i < S; i++) {
            ret.vec_[i] = vec_[i] >= rhs.vec_[i] ? set : 0;
        }
        return ret;
    }

    simd_vec<T, S> sqrt() const {
        simd_vec<T, S> temp;
        for (int i = 0; i < S; i++) {
            temp[i] = ::sqrt(vec_[i]);
        }
        return temp;
    }

    void copy_to(T *f) const {
        memcpy(f, &vec_[0], S * sizeof(T));
    }

    void copy_to(T *f, simd_mem_aligned_tag) const {
        memcpy(f, &vec_[0], S * sizeof(T));
    }

    bool all_zeros() const {
        for (int i = 0; i < S; i++) {
            if (vec_[i] != 0) return false;
        }
        return true;
    }

    bool all_zeros(const simd_vec<int, S> &mask) const {
        for (int i = 0; i < S; i++) {
            if (vec_[i] != 0 && mask.vec_ == 0) return false;
        }
        return true;
    }

    bool not_all_zeros() const {
        for (int i = 0; i < S; i++) {
            if (vec_[i] != 0) return true;
        }
        return false;
    }

    void blend_to(const simd_vec<T, S> &mask, const simd_vec<T, S> &v1) {
        for (int i = 0; i < S; i++) {
            if (mask.vec_[i] != T(0)) vec_[i] = v1.vec_[i];
        }
    }

    static simd_vec<T, S> min(const simd_vec<T, S> &v1, const simd_vec<T, S> &v2) {
        simd_vec<T, S> temp;
        for (int i = 0; i < S; i++) {
            temp.vec_[i] = std::min(v1.vec_[i], v2.vec_[i]);
        }
        return temp;
    }

    static simd_vec<T, S> max(const simd_vec<T, S> &v1, const simd_vec<T, S> &v2) {
        simd_vec<T, S> temp;
        for (int i = 0; i < S; i++) {
            temp.vec_[i] = std::max(v1.vec_[i], v2.vec_[i]);
        }
        return temp;
    }

    static simd_vec<T, S> and(const simd_vec<T, S> &v1, const simd_vec<T, S> &v2) {
        const auto *src1 = reinterpret_cast<const uint8_t*>(&v1.vec_[0]);
        const auto *src2 = reinterpret_cast<const uint8_t*>(&v2.vec_[0]);

        simd_vec<T, S> ret;

        auto *dst = reinterpret_cast<uint8_t*>(&ret.vec_[0]);

        for (int i = 0; i < S * sizeof(T); i++) {
            dst[i] = src1[i] & src2[i];
        }
        return ret;
    }

    static simd_vec<T, S> and_not(const simd_vec<T, S> &v1, const simd_vec<T, S> &v2) {
        const auto *src1 = reinterpret_cast<const uint8_t*>(&v1.vec_[0]);
        const auto *src2 = reinterpret_cast<const uint8_t*>(&v2.vec_[0]);

        simd_vec<T, S> ret;

        auto *dst = reinterpret_cast<uint8_t*>(&ret.vec_[0]);

        for (int i = 0; i < S * sizeof(T); i++) {
            dst[i] = ~src1[i] & src2[i];
        }
        return ret;
    }

    static simd_vec<T, S> or(const simd_vec<T, S> &v1, const simd_vec<T, S> &v2) {
        const auto *src1 = reinterpret_cast<const uint8_t*>(&v1.vec_[0]);
        const auto *src2 = reinterpret_cast<const uint8_t*>(&v2.vec_[0]);

        simd_vec<T, S> ret;

        auto *dst = reinterpret_cast<uint8_t*>(&ret.vec_[0]);

        for (int i = 0; i < S * sizeof(T); i++) {
            dst[i] = src1[i] | src2[i];
        }
        return ret;
    }

    static simd_vec<T, S> xor(const simd_vec<T, S> &v1, const simd_vec<T, S> &v2) {
        const auto *src1 = reinterpret_cast<const uint8_t*>(&v1.vec_[0]);
        const auto *src2 = reinterpret_cast<const uint8_t*>(&v2.vec_[0]);

        simd_vec<T, S> ret;

        auto *dst = reinterpret_cast<uint8_t*>(&ret.vec_[0]);

        for (int i = 0; i < S * sizeof(T); i++) {
            dst[i] = src1[i] ^ src2[i];
        }
        return ret;
    }

    static const size_t alignment = 1;

    static int size() { return S; }
    static int native_count() { return S; }
    static bool is_native() { return native_count() == 1; }
};

template <typename T, int S>
inline simd_vec<T, S> operator+(const simd_vec<T, S> &v1, const simd_vec<T, S> &v2) { simd_vec<T, S> temp = v1; temp += v2; return temp; }

template <typename T, int S>
inline simd_vec<T, S> operator-(const simd_vec<T, S> &v1, const simd_vec<T, S> &v2) { simd_vec<T, S> temp = v1; temp -= v2; return temp; }

template <typename T, int S>
inline simd_vec<T, S> operator*(const simd_vec<T, S> &v1, const simd_vec<T, S> &v2) { simd_vec<T, S> temp = v1; temp *= v2; return temp; }

template <typename T, int S>
inline simd_vec<T, S> operator/(const simd_vec<T, S> &v1, const simd_vec<T, S> &v2) { simd_vec<T, S> temp = v1; temp /= v2; return temp; }

template <typename T, int S>
inline simd_vec<T, S> operator+(const simd_vec<T, S> &v1, T v2) { simd_vec<T, S> temp = v1; temp += v2; return temp; }

template <typename T, int S>
inline simd_vec<T, S> operator-(const simd_vec<T, S> &v1, T v2) { simd_vec<T, S> temp = v1; temp -= v2; return temp; }

template <typename T, int S>
inline simd_vec<T, S> operator*(const simd_vec<T, S> &v1, T v2) { simd_vec<T, S> temp = v1; temp *= v2; return temp; }

template <typename T, int S>
inline simd_vec<T, S> operator/(const simd_vec<T, S> &v1, T v2) { simd_vec<T, S> temp = v1; temp /= v2; return temp; }

template <typename T, int S>
inline simd_vec<T, S> operator<(const simd_vec<T, S> &v1, const simd_vec<T, S> &v2) { return simd_vec<T, S>::operator<(v1, v2); }

template <typename T, int S>
inline simd_vec<T, S> operator<=(const simd_vec<T, S> &v1, const simd_vec<T, S> &v2) { return simd_vec<T, S>::operator<=(v1, v2); }

template <typename T, int S>
inline simd_vec<T, S> operator>(const simd_vec<T, S> &v1, const simd_vec<T, S> &v2) { return simd_vec<T, S>::operator>(v1, v2); }

template <typename T, int S>
inline simd_vec<T, S> operator>=(const simd_vec<T, S> &v1, const simd_vec<T, S> &v2) { return simd_vec<T, S>::operator>=(v1, v2); }

template <typename T, int S>
inline simd_vec<T, S> operator&(const simd_vec<T, S> &v1, const simd_vec<T, S> &v2) { return simd_vec<T, S>::and(v1, v2); }

template <typename T, int S>
inline simd_vec<T, S> and_not(const simd_vec<T, S> &v1, const simd_vec<T, S> &v2) { return simd_vec<T, S>::and_not(v1, v2); }

template <typename T, int S>
inline simd_vec<T, S> operator|(const simd_vec<T, S> &v1, const simd_vec<T, S> &v2) { return simd_vec<T, S>::or(v1, v2); }

template <typename T, int S>
inline simd_vec<T, S> operator^(const simd_vec<T, S> &v1, const simd_vec<T, S> &v2) { return simd_vec<T, S>::xor(v1, v2); }

template <typename T, int S>
inline simd_vec<T, S> sqrt(const simd_vec<T, S> &v1) { return v1.sqrt(); }

template <typename T, int S>
inline simd_vec<T, S> min(const simd_vec<T, S> &v1, const simd_vec<T, S> &v2) { return simd_vec<T, S>::min(v1, v2); }

template <typename T, int S>
inline simd_vec<T, S> max(const simd_vec<T, S> &v1, const simd_vec<T, S> &v2) { return simd_vec<T, S>::max(v1, v2); }

template <typename T, int S>
class simd_vec_where_helper {
    const simd_vec<T, S> &mask_;
    simd_vec<T, S> &vec_;
public:
    simd_vec_where_helper(const simd_vec<T, S> &mask, simd_vec<T, S> &vec) : mask_(mask), vec_(vec) {}

    void operator=(const simd_vec<T, S> &vec) {
        vec_.blend_to(mask_, vec);
    }
};

template <typename T, int S>
inline simd_vec_where_helper<T, S> where(const simd_vec<T, S> &mask, simd_vec<T, S> &vec) {
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
