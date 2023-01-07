
#include <cmath>
#include <cstring>

#include <algorithm> // for std::max

#pragma warning(push)
#pragma warning(disable : 4789) // buffer overrun
#pragma warning(disable : 4127) // conditional expression is constant

// Used to force loop unroll in release mode
// clang-format off
#define ITERATE_2(exp)  \
        { const int i = 0; exp }    \
        { const int i = 1; exp }

#define ITERATE_3(exp)  \
        { const int i = 0; exp }    \
        { const int i = 1; exp }    \
        { const int i = 2; exp }

#define ITERATE_4(exp)  \
        { const int i = 0; exp }    \
        { const int i = 1; exp }    \
        { const int i = 2; exp }    \
        { const int i = 3; exp }

#define ITERATE_8(exp)  \
        { const int i = 0; exp }    \
        { const int i = 1; exp }    \
        { const int i = 2; exp }    \
        { const int i = 3; exp }    \
        { const int i = 4; exp }    \
        { const int i = 5; exp }    \
        { const int i = 6; exp }    \
        { const int i = 7; exp }

#define ITERATE_16(exp)  \
        { const int i = 0; exp }    \
        { const int i = 1; exp }    \
        { const int i = 2; exp }    \
        { const int i = 3; exp }    \
        { const int i = 4; exp }    \
        { const int i = 5; exp }    \
        { const int i = 6; exp }    \
        { const int i = 7; exp }    \
        { const int i = 8; exp }    \
        { const int i = 9; exp }    \
        { const int i = 10; exp }   \
        { const int i = 11; exp }   \
        { const int i = 12; exp }   \
        { const int i = 13; exp }   \
        { const int i = 14; exp }   \
        { const int i = 15; exp }

#define ITERATE(n, exp)  \
    if ((n) == 16) {                \
        ITERATE_16(exp)             \
    } else if ((n) == 8) {          \
        ITERATE_8(exp)              \
    } else if ((n) == 4) {          \
        ITERATE_4(exp)              \
    } else if ((n) == 3) {          \
        ITERATE_3(exp)              \
    } else if ((n) == 2) {          \
        ITERATE_2(exp)              \
    } else if ((n) == 1) {          \
        { const int i = 0; exp }    \
    }

#define ITERATE_R(n, exp)  \
    if ((n) == 16) {                \
        { const int i = 15; exp }   \
        { const int i = 14; exp }   \
        { const int i = 13; exp }   \
        { const int i = 12; exp }   \
        { const int i = 11; exp }   \
        { const int i = 10; exp }   \
        { const int i = 9; exp }    \
        { const int i = 8; exp }    \
        { const int i = 7; exp }    \
        { const int i = 6; exp }    \
        { const int i = 5; exp }    \
        { const int i = 4; exp }    \
        { const int i = 3; exp }    \
        { const int i = 2; exp }    \
        { const int i = 1; exp }    \
        { const int i = 0; exp }    \
    } else if ((n) == 8) {          \
        { const int i = 7; exp }    \
        { const int i = 6; exp }    \
        { const int i = 5; exp }    \
        { const int i = 4; exp }    \
        { const int i = 3; exp }    \
        { const int i = 2; exp }    \
        { const int i = 1; exp }    \
        { const int i = 0; exp }    \
    } else if ((n) == 4) {          \
        { const int i = 3; exp }    \
        { const int i = 2; exp }    \
        { const int i = 1; exp }    \
        { const int i = 0; exp }    \
    } else if ((n) == 3) {          \
        { const int i = 2; exp }    \
        { const int i = 1; exp }    \
        { const int i = 0; exp }    \
    } else if ((n) == 2) {          \
        { const int i = 1; exp }    \
        { const int i = 0; exp }    \
    } else if ((n) == 1) {          \
        { const int i = 0; exp }    \
    }

#define ITERATE_2_R(exp)  \
        { const int i = 1; exp }    \
        { const int i = 0; exp }
// clang-format on

#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Warray-bounds"
#endif

namespace Ray {
namespace NS {

enum simd_mem_aligned_tag { simd_mem_aligned };

template <typename T, int S> class simd_vec {
    T comp_[S];

    friend class simd_vec<int, S>;
    friend class simd_vec<float, S>;

  public:
    force_inline simd_vec() = default;
    force_inline simd_vec(T f) {
        ITERATE(S, { comp_[i] = f; })
    }
    template <typename... Tail>
    force_inline simd_vec(typename std::enable_if<sizeof...(Tail) + 1 == S, T>::type head, Tail... tail)
        : comp_{head, T(tail)...} {}
    force_inline explicit simd_vec(const T *f) { memcpy(&comp_, f, S * sizeof(T)); }
    force_inline simd_vec(const T *f, simd_mem_aligned_tag) { memcpy(&comp_, f, S * sizeof(T)); }

    force_inline T &operator[](const int i) { return comp_[i]; }
    force_inline T operator[](const int i) const { return comp_[i]; }

    template <int i> force_inline T get() const { return comp_[i]; }

    force_inline simd_vec<T, S> &operator+=(const simd_vec<T, S> &rhs) {
        ITERATE(S, { comp_[i] += rhs.comp_[i]; })
        return *this;
    }

    force_inline simd_vec<T, S> &operator-=(const simd_vec<T, S> &rhs) {
        ITERATE(S, { comp_[i] -= rhs.comp_[i]; })
        return *this;
    }

    force_inline simd_vec<T, S> &operator*=(const simd_vec<T, S> &rhs) {
        ITERATE(S, { comp_[i] *= rhs.comp_[i]; })
        return *this;
    }

    force_inline simd_vec<T, S> &operator/=(const simd_vec<T, S> &rhs) {
        ITERATE(S, { comp_[i] /= rhs.comp_[i]; })
        return *this;
    }

    force_inline simd_vec<T, S> &operator|=(const simd_vec<T, S> &rhs) {
        const auto *src2 = reinterpret_cast<const uint8_t *>(&rhs.comp_[0]);

        auto *dst = reinterpret_cast<uint8_t *>(&comp_[0]);

        for (int i = 0; i < S * sizeof(T); i++) {
            dst[i] |= src2[i];
        }

        return *this;
    }

    force_inline simd_vec<T, S> &operator+=(T rhs) {
        ITERATE(S, { comp_[i] += rhs; })
        return *this;
    }

    force_inline simd_vec<T, S> &operator-=(T rhs) {
        ITERATE(S, { comp_[i] -= rhs; })
        return *this;
    }

    force_inline simd_vec<T, S> &operator*=(T rhs) {
        ITERATE(S, { comp_[i] *= rhs; })
        return *this;
    }

    force_inline simd_vec<T, S> &operator/=(T rhs) {
        ITERATE(S, { comp_[i] /= rhs; })
        return *this;
    }

    force_inline simd_vec<T, S> operator-() const {
        simd_vec<T, S> temp;
        ITERATE(S, { temp.comp_[i] = -comp_[i]; })
        return temp;
    }

    force_inline simd_vec<T, S> operator==(T rhs) const {
        T set, not_set = T(0);
        memset(&set, 0xFF, sizeof(T));
        simd_vec<T, S> ret;
        ITERATE(S, { ret.comp_[i] = comp_[i] == rhs ? set : not_set; })
        return ret;
    }

    force_inline simd_vec<T, S> operator==(const simd_vec<T, S> &rhs) const {
        T set, not_set = T(0);
        memset(&set, 0xFF, sizeof(T));
        simd_vec<T, S> ret;
        ITERATE(S, { ret.comp_[i] = comp_[i] == rhs.comp_[i] ? set : not_set; })
        return ret;
    }

    force_inline simd_vec<T, S> operator!=(T rhs) const {
        T set, not_set = T(0);
        memset(&set, 0xFF, sizeof(T));
        simd_vec<T, S> ret;
        ITERATE(S, { ret.comp_[i] = comp_[i] != rhs ? set : not_set; })
        return ret;
    }

    force_inline simd_vec<T, S> operator!=(const simd_vec<T, S> &rhs) const {
        T set, not_set = T(0);
        memset(&set, 0xFF, sizeof(T));
        simd_vec<T, S> ret;
        ITERATE(S, { ret.comp_[i] = comp_[i] != rhs.comp_[i] ? set : not_set; })
        return ret;
    }

    force_inline simd_vec<T, S> operator<(const simd_vec<T, S> &rhs) const {
        T set, not_set = T(0);
        memset(&set, 0xFF, sizeof(T));
        simd_vec<T, S> ret;
        ITERATE(S, { ret.comp_[i] = comp_[i] < rhs.comp_[i] ? set : not_set; })
        return ret;
    }

    force_inline simd_vec<T, S> operator<=(const simd_vec<T, S> &rhs) const {
        T set, not_set = T(0);
        memset(&set, 0xFF, sizeof(T));
        simd_vec<T, S> ret;
        ITERATE(S, { ret.comp_[i] = comp_[i] <= rhs.comp_[i] ? set : not_set; })
        return ret;
    }

    force_inline simd_vec<T, S> operator>(const simd_vec<T, S> &rhs) const {
        T set, not_set = T(0);
        memset(&set, 0xFF, sizeof(T));
        simd_vec<T, S> ret;
        ITERATE(S, { ret.comp_[i] = comp_[i] > rhs.comp_[i] ? set : not_set; })
        return ret;
    }

    force_inline simd_vec<T, S> operator>=(const simd_vec<T, S> &rhs) const {
        T set, not_set = T(0);
        memset(&set, 0xFF, sizeof(T));
        simd_vec<T, S> ret;
        ITERATE(S, { ret.comp_[i] = comp_[i] >= rhs.comp_[i] ? set : not_set; })
        return ret;
    }

    force_inline simd_vec<T, S> &operator&=(const simd_vec<T, S> &rhs) {
        ITERATE(S, { comp_[i] &= rhs.comp_[i]; })
        return *this;
    }

    force_inline simd_vec<T, S> operator<(T rhs) const {
        T set, not_set = T(0);
        memset(&set, 0xFF, sizeof(T));
        simd_vec<T, S> ret;
        ITERATE(S, { ret.comp_[i] = comp_[i] < rhs ? set : not_set; })
        return ret;
    }

    force_inline simd_vec<T, S> operator<=(T rhs) const {
        T set, not_set = T(0);
        memset(&set, 0xFF, sizeof(T));
        simd_vec<T, S> ret;
        ITERATE(S, { ret.comp_[i] = comp_[i] <= rhs ? set : not_set; })
        return ret;
    }

    force_inline simd_vec<T, S> operator>(T rhs) const {
        T set, not_set = T(0);
        memset(&set, 0xFF, sizeof(T));
        simd_vec<T, S> ret;
        ITERATE(S, { ret.comp_[i] = comp_[i] > rhs ? set : not_set; })
        return ret;
    }

    force_inline simd_vec<T, S> operator>=(T rhs) const {
        T set, not_set = T(0);
        memset(&set, 0xFF, sizeof(T));
        simd_vec<T, S> ret;
        ITERATE(S, { ret.comp_[i] = comp_[i] >= rhs ? set : not_set; })
        return ret;
    }

    force_inline simd_vec<T, S> &operator&=(const T rhs) {
        ITERATE(S, { comp_[i] &= rhs; })
        return *this;
    }

    force_inline simd_vec<T, S> operator~() const {
        simd_vec<T, S> ret;
        ITERATE(S, {
            const uint32_t temp = ~reinterpret_cast<const uint32_t &>(comp_[i]);
            ret.comp_[i] = reinterpret_cast<const T &>(temp);
        })
        return ret;
    }

    force_inline explicit operator simd_vec<int, S>() const {
        simd_vec<int, S> ret;
        ITERATE(S, { ret.comp_[i] = (int)comp_[i]; })
        return ret;
    }

    force_inline explicit operator simd_vec<float, S>() const {
        simd_vec<float, S> ret;
        ITERATE(S, { ret.comp_[i] = (float)comp_[i]; })
        return ret;
    }

    force_inline simd_vec<T, S> sqrt() const {
        simd_vec<T, S> temp;
        ITERATE(S, { temp[i] = std::sqrt(comp_[i]); })
        return temp;
    }

    force_inline simd_vec<T, S> log() const {
        simd_vec<T, S> temp;
        ITERATE(S, { temp[i] = std::log(comp_[i]); })
        return temp;
    }

    force_inline T length() const {
        T temp = {0};
        ITERATE(S, { temp += comp_[i] * comp_[i]; })
        return std::sqrt(temp);
    }

    force_inline T length2() const {
        T temp = {0};
        ITERATE(S, { temp += comp_[i] * comp_[i]; })
        return temp;
    }

    force_inline void copy_to(T *f) const { memcpy(f, &comp_[0], S * sizeof(T)); }

    force_inline void copy_to(T *f, simd_mem_aligned_tag) const { memcpy(f, &comp_[0], S * sizeof(T)); }

    force_inline bool all_zeros() const {
        ITERATE(S, {
            if (comp_[i] != 0)
                return false;
        })
        return true;
    }

    force_inline bool all_zeros(const simd_vec<int, S> &mask) const {
        const auto *src1 = reinterpret_cast<const uint8_t *>(&comp_[0]);
        const auto *src2 = reinterpret_cast<const uint8_t *>(&mask.comp_[0]);

        for (int i = 0; i < S * sizeof(T); i++) {
            if ((src1[i] & src2[i]) != 0)
                return false;
        }

        return true;
    }

    force_inline bool not_all_zeros() const {
        ITERATE(S, {
            if (comp_[i] != 0)
                return true;
        })
        return false;
    }

    // clang-format off
    force_inline void blend_to(const simd_vec<T, S> &mask, const simd_vec<T, S> &v1) {
        ITERATE(S, {
            if (mask.comp_[i] != T(0)) {
                comp_[i] = v1.comp_[i];
            }
        })
    }

    force_inline
        void blend_inv_to(const simd_vec<T, S> &mask, const simd_vec<T, S> &v1) {
        ITERATE(S, {
            if (mask.comp_[i] == T(0)) {
                comp_[i] = v1.comp_[i];
            }
        })
    } // clang-format on

    force_inline int movemask() const {
        int res = 0;
        ITERATE(S, {
            if (comp_[i] != T(0)) {
                res |= (1 << i);
            }
        })
        return res;
    }

    force_inline static simd_vec<T, S> min(const simd_vec<T, S> &v1, const simd_vec<T, S> &v2) {
        simd_vec<T, S> temp;
        ITERATE(S, { temp.comp_[i] = std::min(v1.comp_[i], v2.comp_[i]); })
        return temp;
    }

    force_inline static simd_vec<T, S> min(const simd_vec<T, S> &v1, const T v2) {
        simd_vec<T, S> temp;
        ITERATE(S, { temp.comp_[i] = std::min(v1.comp_[i], v2); })
        return temp;
    }

    force_inline static simd_vec<T, S> min(const T v1, const simd_vec<T, S> &v2) {
        simd_vec<T, S> temp;
        ITERATE(S, { temp.comp_[i] = std::min(v1, v2.comp_[i]); })
        return temp;
    }

    force_inline static simd_vec<T, S> max(const simd_vec<T, S> &v1, const simd_vec<T, S> &v2) {
        simd_vec<T, S> temp;
        ITERATE(S, { temp.comp_[i] = std::max(v1.comp_[i], v2.comp_[i]); })
        return temp;
    }

    force_inline static simd_vec<T, S> max(const simd_vec<T, S> &v1, const T v2) {
        simd_vec<T, S> temp;
        ITERATE(S, { temp.comp_[i] = std::max(v1.comp_[i], v2); })
        return temp;
    }

    force_inline static simd_vec<T, S> max(const T v1, const simd_vec<T, S> &v2) {
        simd_vec<T, S> temp;
        ITERATE(S, { temp.comp_[i] = std::max(v1, v2.comp_[i]); })
        return temp;
    }

    force_inline static simd_vec<T, S> and_not(const simd_vec<T, S> &v1, const simd_vec<T, S> &v2) {
        const auto *src1 = reinterpret_cast<const uint8_t *>(&v1.comp_[0]);
        const auto *src2 = reinterpret_cast<const uint8_t *>(&v2.comp_[0]);

        simd_vec<T, S> ret;

        auto *dst = reinterpret_cast<uint8_t *>(&ret.comp_[0]);

        for (int i = 0; i < S * sizeof(T); i++) {
            dst[i] = (~src1[i]) & src2[i];
        }

        return ret;
    }

    force_inline static simd_vec<float, S> floor(const simd_vec<float, S> &v1) {
        simd_vec<float, S> temp;
        ITERATE(S, { temp.comp_[i] = (float)((int)v1.comp_[i] - (v1.comp_[i] < 0.0f)); })
        return temp;
    }

    force_inline static simd_vec<float, S> ceil(const simd_vec<float, S> &v1) {
        simd_vec<float, S> temp;
        ITERATE(S, {
            int _v = (int)v1.comp_[i];
            temp.comp_[i] = (float)(_v + (v1.comp_[i] != _v));
        })
        return temp;
    }

    friend force_inline simd_vec<T, S> operator&(const simd_vec<T, S> &v1, const simd_vec<T, S> &v2) {
        const auto *src1 = reinterpret_cast<const uint8_t *>(&v1.comp_[0]);
        const auto *src2 = reinterpret_cast<const uint8_t *>(&v2.comp_[0]);

        simd_vec<T, S> ret;

        auto *dst = reinterpret_cast<uint8_t *>(&ret.comp_[0]);

        for (int i = 0; i < S * sizeof(T); i++) {
            dst[i] = src1[i] & src2[i];
        }

        return ret;
    }

    friend force_inline simd_vec<T, S> operator|(const simd_vec<T, S> &v1, const simd_vec<T, S> &v2) {
        const auto *src1 = reinterpret_cast<const uint8_t *>(&v1.comp_[0]);
        const auto *src2 = reinterpret_cast<const uint8_t *>(&v2.comp_[0]);

        simd_vec<T, S> ret;

        auto *dst = reinterpret_cast<uint8_t *>(&ret.comp_[0]);

        for (int i = 0; i < S * sizeof(T); i++) {
            dst[i] = src1[i] | src2[i];
        }

        return ret;
    }

    friend force_inline simd_vec<T, S> operator^(const simd_vec<T, S> &v1, const simd_vec<T, S> &v2) {
        const auto *src1 = reinterpret_cast<const uint8_t *>(&v1.comp_[0]);
        const auto *src2 = reinterpret_cast<const uint8_t *>(&v2.comp_[0]);

        simd_vec<T, S> ret;

        auto *dst = reinterpret_cast<uint8_t *>(&ret.comp_[0]);

        for (int i = 0; i < S * sizeof(T); i++) {
            dst[i] = src1[i] ^ src2[i];
        }

        return ret;
    }

    friend force_inline simd_vec<T, S> operator+(const simd_vec<T, S> &v1, const simd_vec<T, S> &v2) {
        simd_vec<T, S> ret;
        ITERATE(S, { ret.comp_[i] = v1.comp_[i] + v2.comp_[i]; })
        return ret;
    }

    friend force_inline simd_vec<T, S> operator-(const simd_vec<T, S> &v1, const simd_vec<T, S> &v2) {
        simd_vec<T, S> ret;
        ITERATE(S, { ret.comp_[i] = v1.comp_[i] - v2.comp_[i]; })
        return ret;
    }

    friend force_inline simd_vec<T, S> operator*(const simd_vec<T, S> &v1, const simd_vec<T, S> &v2) {
        simd_vec<T, S> ret;
        ITERATE(S, { ret.comp_[i] = v1.comp_[i] * v2.comp_[i]; })
        return ret;
    }

    friend force_inline simd_vec<T, S> operator/(const simd_vec<T, S> &v1, const simd_vec<T, S> &v2) {
        simd_vec<T, S> ret;
        ITERATE(S, { ret.comp_[i] = v1.comp_[i] / v2.comp_[i]; })
        return ret;
    }

    friend force_inline simd_vec<T, S> operator+(const simd_vec<T, S> &v1, T v2) {
        simd_vec<T, S> ret;
        ITERATE(S, { ret.comp_[i] = v1.comp_[i] + v2; })
        return ret;
    }

    friend force_inline simd_vec<T, S> operator-(const simd_vec<T, S> &v1, T v2) {
        simd_vec<T, S> ret;
        ITERATE(S, { ret.comp_[i] = v1.comp_[i] - v2; })
        return ret;
    }

    friend force_inline simd_vec<T, S> operator*(const simd_vec<T, S> &v1, T v2) {
        simd_vec<T, S> ret;
        ITERATE(S, { ret.comp_[i] = v1.comp_[i] * v2; })
        return ret;
    }

    friend force_inline simd_vec<T, S> operator/(const simd_vec<T, S> &v1, T v2) {
        simd_vec<T, S> ret;
        ITERATE(S, { ret.comp_[i] = v1.comp_[i] / v2; })
        return ret;
    }

    friend force_inline simd_vec<T, S> operator+(T v1, const simd_vec<T, S> &v2) { return operator+(v2, v1); }

    friend force_inline simd_vec<T, S> operator-(T v1, const simd_vec<T, S> &v2) {
        simd_vec<T, S> ret;
        ITERATE(S, { ret.comp_[i] = v1 - v2.comp_[i]; })
        return ret;
    }

    friend force_inline simd_vec<T, S> operator*(T v1, const simd_vec<T, S> &v2) {
        simd_vec<T, S> ret;
        ITERATE(S, { ret.comp_[i] = v1 * v2.comp_[i]; })
        return ret;
    }

    friend force_inline simd_vec<T, S> operator/(T v1, const simd_vec<T, S> &v2) {
        simd_vec<T, S> ret;
        ITERATE(S, { ret.comp_[i] = v1 / v2.comp_[i]; })
        return ret;
    }

    friend force_inline simd_vec<T, S> operator>>(const simd_vec<T, S> &v1, const simd_vec<T, S> &v2) {
        simd_vec<T, S> ret;
        ITERATE(S, { ret.comp_[i] = v1.comp_[i] >> v2.comp_[i]; })
        return ret;
    }

    friend force_inline simd_vec<T, S> operator>>(const simd_vec<T, S> &v1, T v2) {
        simd_vec<T, S> ret;
        ITERATE(S, { ret.comp_[i] = v1.comp_[i] >> v2; })
        return ret;
    }

    friend force_inline simd_vec<T, S> operator<<(const simd_vec<T, S> &v1, const simd_vec<T, S> &v2) {
        simd_vec<T, S> ret;
        ITERATE(S, { ret.comp_[i] = v1.comp_[i] << v2.comp_[i]; })
        return ret;
    }

    friend force_inline simd_vec<T, S> operator<<(const simd_vec<T, S> &v1, T v2) {
        simd_vec<T, S> ret;
        ITERATE(S, { ret.comp_[i] = v1.comp_[i] << v2; })
        return ret;
    }

    friend force_inline simd_vec<T, S> srai(const simd_vec<T, S> &v1, int v2) {
        simd_vec<T, S> ret;
        ITERATE(S, { ret.comp_[i] = v1.comp_[i] >> v2; })
        return ret;
    }

    friend force_inline T dot(const simd_vec<T, S> &v1, const simd_vec<T, S> &v2) {
        T ret = {0};
        ITERATE(S, { ret += v1.comp_[i] * v2.comp_[i]; })
        return ret;
    }

    friend force_inline simd_vec<T, S> clamp(const simd_vec<T, S> &v1, T min, T max) {
        simd_vec<T, S> ret;
        ITERATE(S, { ret.comp_[i] = v1.comp_[i] < min ? min : (v1.comp_[i] > max ? max : v1.comp_[i]); })
        return ret;
    }

    friend force_inline simd_vec<T, S> pow(const simd_vec<T, S> &v1, const simd_vec<T, S> &v2) {
        simd_vec<T, S> ret;
        ITERATE(S, { ret.comp_[i] = std::pow(v1.comp_[i], v2.comp_[i]); })
        return ret;
    }

    friend force_inline simd_vec<T, S> normalize(const simd_vec<T, S> &v1) { return v1 / v1.length(); }

    friend force_inline bool is_equal(const simd_vec<T, S> &v1, const simd_vec<T, S> &v2) {
        bool res = true;
        ITERATE(S, { res = res && (v1.comp_[i] == v2.comp_[i]); })
        return res;
    }

    friend force_inline const T *value_ptr(const simd_vec<T, S> &v1) { return &v1.comp_[0]; }

    static int size() { return S; }
    static bool is_native() { return false; }
};

template <typename T, int S> force_inline simd_vec<T, S> and_not(const simd_vec<T, S> &v1, const simd_vec<T, S> &v2) {
    return simd_vec<T, S>::and_not(v1, v2);
}

template <typename T, int S> force_inline simd_vec<T, S> floor(const simd_vec<T, S> &v1) {
    return simd_vec<T, S>::floor(v1);
}

template <typename T, int S> force_inline simd_vec<T, S> ceil(const simd_vec<T, S> &v1) {
    return simd_vec<T, S>::ceil(v1);
}

template <typename T, int S> force_inline simd_vec<T, S> sqrt(const simd_vec<T, S> &v1) { return v1.sqrt(); }
template <typename T, int S> force_inline simd_vec<T, S> log(const simd_vec<T, S> &v1) { return v1.log(); }

template <typename T, int S> force_inline T length(const simd_vec<T, S> &v1) { return v1.length(); }

template <typename T, int S> force_inline T length2(const simd_vec<T, S> &v1) { return v1.length2(); }

template <typename T, int S> force_inline simd_vec<T, S> fract(const simd_vec<T, S> &v1) { return v1 - floor(v1); }

template <typename T, int S> force_inline simd_vec<T, S> min(const simd_vec<T, S> &v1, const simd_vec<T, S> &v2) {
    return simd_vec<T, S>::min(v1, v2);
}

template <typename T, int S> force_inline simd_vec<T, S> min(const simd_vec<T, S> &v1, const T v2) {
    return simd_vec<T, S>::min(v1, v2);
}

template <typename T, int S> force_inline simd_vec<T, S> max(const simd_vec<T, S> &v1, const simd_vec<T, S> &v2) {
    return simd_vec<T, S>::max(v1, v2);
}

template <typename T, int S> force_inline simd_vec<T, S> max(const simd_vec<T, S> &v1, const T v2) {
    return simd_vec<T, S>::max(v1, v2);
}

template <typename T, int S> force_inline simd_vec<T, S> max(const T v1, const simd_vec<T, S> &v2) {
    return simd_vec<T, S>::max(v1, v2);
}

template <typename T, int S> force_inline simd_vec<T, S> abs(const simd_vec<T, S> &v) {
    // TODO: find faster implementation
    return max(v, -v);
}

template <typename T, int S>
force_inline simd_vec<T, S> fmadd(const simd_vec<T, S> &a, const simd_vec<T, S> &b, const simd_vec<T, S> &c) {
    return a * b + c;
}

template <typename T, int S>
force_inline simd_vec<T, S> fmadd(const simd_vec<T, S> &a, const float b, const simd_vec<T, S> &c) {
    return a * b + c;
}

template <typename T, int S> force_inline simd_vec<T, S> fmadd(const float a, const simd_vec<T, S> &b, const float c) {
    return a * b + c;
}

template <typename T, int S>
force_inline simd_vec<T, S> fmsub(const simd_vec<T, S> &a, const simd_vec<T, S> &b, const simd_vec<T, S> &c) {
    return a * b - c;
}

template <typename T, int S>
force_inline simd_vec<T, S> fmsub(const simd_vec<T, S> &a, const float b, const simd_vec<T, S> &c) {
    return a * b - c;
}

template <int Scale = 1, typename T, int S>
force_inline simd_vec<T, S> gather(const T *base_addr, const simd_vec<int, S> &vindex) {
    simd_vec<T, S> res;
    ITERATE(S, { res[i] = base_addr[vindex[i] * Scale]; });
    return res;
}

template <typename T, int S> force_inline simd_vec<T, S> fmsub(const float a, const simd_vec<T, S> &b, const float c) {
    return a * b - c;
}

template <typename T, int S> force_inline simd_vec<T, S> mix(const simd_vec<T, S> &v1, const simd_vec<T, S> &v2, T k) {
    return (1.0f - k) * v1 + k * v2;
}

template <typename T, int S>
force_inline simd_vec<T, S> mix(const simd_vec<T, S> &v1, const simd_vec<T, S> &v2, simd_vec<T, S> k) {
    return (simd_vec<T, S>{1} - k) * v1 + k * v2;
}

template <typename T, typename U, int S, bool Inv> class simd_comp_where_helper {
    const simd_vec<T, S> &mask_;
    simd_vec<T, S> &comp_;

  public:
    force_inline simd_comp_where_helper(const simd_vec<U, S> &mask, simd_vec<T, S> &vec)
        : mask_(reinterpret_cast<const simd_vec<T, S> &>(mask)), comp_(vec) {}

    force_inline void operator=(const simd_vec<T, S> &vec) {
        if (!Inv) {
            comp_.blend_to(mask_, vec);
        } else {
            comp_.blend_inv_to(mask_, vec);
        }
    }

    force_inline void operator+=(const simd_vec<T, S> &vec) {
        if (!Inv) {
            comp_.blend_to(mask_, comp_ + vec);
        } else {
            comp_.blend_inv_to(mask_, comp_ + vec);
        }
    }

    force_inline void operator-=(const simd_vec<T, S> &vec) {
        if (!Inv) {
            comp_.blend_to(mask_, comp_ - vec);
        } else {
            comp_.blend_inv_to(mask_, comp_ - vec);
        }
    }

    force_inline void operator*=(const simd_vec<T, S> &vec) {
        if (!Inv) {
            comp_.blend_to(mask_, comp_ * vec);
        } else {
            comp_.blend_inv_to(mask_, comp_ * vec);
        }
    }

    force_inline void operator/=(const simd_vec<T, S> &vec) {
        if (!Inv) {
            comp_.blend_to(mask_, comp_ / vec);
        } else {
            comp_.blend_inv_to(mask_, comp_ / vec);
        }
    }

    force_inline void operator|=(const simd_vec<T, S> &vec) {
        if (!Inv) {
            comp_.blend_to(mask_, comp_ | vec);
        } else {
            comp_.blend_inv_to(mask_, comp_ | vec);
        }
    }

    force_inline void operator&=(const simd_vec<T, S> &vec) {
        if (!Inv) {
            comp_.blend_to(mask_, comp_ & vec);
        } else {
            comp_.blend_inv_to(mask_, comp_ & vec);
        }
    }
};

template <typename T, typename U, int S>
force_inline simd_comp_where_helper<T, U, S, false> where(const simd_vec<U, S> &mask, simd_vec<T, S> &vec) {
    return {mask, vec};
}

template <typename T, typename U, int S>
force_inline simd_comp_where_helper<T, U, S, true> where_not(const simd_vec<U, S> &mask, simd_vec<T, S> &vec) {
    return {mask, vec};
}

template <int S> force_inline const simd_vec<int, S> &simd_cast(const simd_vec<float, S> &vec) {
    return reinterpret_cast<const simd_vec<int, S> &>(vec);
}

template <int S> force_inline const simd_vec<float, S> &simd_cast(const simd_vec<int, S> &vec) {
    return reinterpret_cast<const simd_vec<float, S> &>(vec);
}

} // namespace NS
} // namespace Ray

#if defined(USE_SSE2) || defined(USE_SSE41)
#include "simd_vec_sse.h"
#elif defined(USE_AVX) || defined(USE_AVX2)
#include "simd_vec_avx.h"
#elif defined(USE_AVX512)
#include "simd_vec_avx512.h"
#elif defined(USE_NEON)
#include "simd_vec_neon.h"
#endif

namespace Ray {
namespace NS {
template <int S> using simd_fvec = simd_vec<float, S>;
using simd_fvec2 = simd_fvec<2>;
using simd_fvec3 = simd_fvec<3>;
using simd_fvec4 = simd_fvec<4>;
using simd_fvec8 = simd_fvec<8>;
using simd_fvec16 = simd_fvec<16>;

template <int S> using simd_ivec = simd_vec<int, S>;
using simd_ivec2 = simd_ivec<2>;
using simd_ivec3 = simd_ivec<3>;
using simd_ivec4 = simd_ivec<4>;
using simd_ivec8 = simd_ivec<8>;
using simd_ivec16 = simd_ivec<16>;

template <int S> using simd_dvec = simd_vec<double, S>;
using simd_dvec2 = simd_dvec<2>;
using simd_dvec3 = simd_dvec<3>;
using simd_dvec4 = simd_dvec<4>;
using simd_dvec8 = simd_dvec<8>;
using simd_dvec16 = simd_dvec<16>;
} // namespace NS
} // namespace Ray

#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif

#pragma warning(pop)

// #undef ITERATE
