
#include <cmath>
#include <cstring>

#include <algorithm> // for std::max

#pragma warning(push)
#pragma warning(disable : 4789) // buffer overrun
#pragma warning(disable : 4127) // conditional expression is constant

// Used to force loop unroll and make index compile-time constant
// clang-format off
#define UNROLLED_FOR_0(I, N, C)
#define UNROLLED_FOR_1(I, N, C) { const int I =  0 % N; C }
#define UNROLLED_FOR_2(I, N, C) { const int I =  0 % N; C } { const int I =  1 % N; C }
#define UNROLLED_FOR_3(I, N, C) { const int I =  0 % N; C } { const int I =  1 % N; C } { const int I =  2 % N; C }
#define UNROLLED_FOR_4(I, N, C) { const int I =  0 % N; C } { const int I =  1 % N; C } { const int I =  2 % N; C } { const int I =  3 % N; C }
#define UNROLLED_FOR_5(I, N, C) { const int I =  0 % N; C } { const int I =  1 % N; C } { const int I =  2 % N; C } { const int I =  3 % N; C } \
                                { const int I =  4 % N; C }
#define UNROLLED_FOR_6(I, N, C) { const int I =  0 % N; C } { const int I =  1 % N; C } { const int I =  2 % N; C } { const int I =  3 % N; C } \
                                { const int I =  4 % N; C } { const int I =  5 % N; C }
#define UNROLLED_FOR_7(I, N, C) { const int I =  0 % N; C } { const int I =  1 % N; C } { const int I =  2 % N; C } { const int I =  3 % N; C } \
                                { const int I =  4 % N; C } { const int I =  5 % N; C } { const int I =  6 % N; C }
#define UNROLLED_FOR_8(I, N, C) { const int I =  0 % N; C } { const int I =  1 % N; C } { const int I =  2 % N; C } { const int I =  3 % N; C } \
                                { const int I =  4 % N; C } { const int I =  5 % N; C } { const int I =  6 % N; C } { const int I =  7 % N; C }
#define UNROLLED_FOR_9(I, N, C) { const int I =  0 % N; C } { const int I =  1 % N; C } { const int I =  2 % N; C } { const int I =  3 % N; C } \
                                { const int I =  4 % N; C } { const int I =  5 % N; C } { const int I =  6 % N; C } { const int I =  7 % N; C } \
                                { const int I =  8 % N; C }
#define UNROLLED_FOR_10(I, N, C){ const int I =  0 % N; C } { const int I =  1 % N; C } { const int I =  2 % N; C } { const int I =  3 % N; C } \
                                { const int I =  4 % N; C } { const int I =  5 % N; C } { const int I =  6 % N; C } { const int I =  7 % N; C } \
                                { const int I =  8 % N; C } { const int I =  9 % N; C }
#define UNROLLED_FOR_11(I, N, C){ const int I =  0 % N; C } { const int I =  1 % N; C } { const int I =  2 % N; C } { const int I =  3 % N; C } \
                                { const int I =  4 % N; C } { const int I =  5 % N; C } { const int I =  6 % N; C } { const int I =  7 % N; C } \
                                { const int I =  8 % N; C } { const int I =  9 % N; C } { const int I = 10 % N; C }
#define UNROLLED_FOR_12(I, N, C){ const int I =  0 % N; C } { const int I =  1 % N; C } { const int I =  2 % N; C } { const int I =  3 % N; C } \
                                { const int I =  4 % N; C } { const int I =  5 % N; C } { const int I =  6 % N; C } { const int I =  7 % N; C } \
                                { const int I =  8 % N; C } { const int I =  9 % N; C } { const int I = 10 % N; C } { const int I = 11 % N; C }
#define UNROLLED_FOR_13(I, N, C){ const int I =  0 % N; C } { const int I =  1 % N; C } { const int I =  2 % N; C } { const int I =  3 % N; C } \
                                { const int I =  4 % N; C } { const int I =  5 % N; C } { const int I =  6 % N; C } { const int I =  7 % N; C } \
                                { const int I =  8 % N; C } { const int I =  9 % N; C } { const int I = 10 % N; C } { const int I = 11 % N; C } \
                                { const int I = 12 % N; C }
#define UNROLLED_FOR_14(I, N, C){ const int I =  0 % N; C } { const int I =  1 % N; C } { const int I =  2 % N; C } { const int I =  3 % N; C } \
                                { const int I =  4 % N; C } { const int I =  5 % N; C } { const int I =  6 % N; C } { const int I =  7 % N; C } \
                                { const int I =  8 % N; C } { const int I =  9 % N; C } { const int I = 10 % N; C } { const int I = 11 % N; C } \
                                { const int I = 12 % N; C } { const int I = 13 % N; C }
#define UNROLLED_FOR_15(I, N, C){ const int I =  0 % N; C } { const int I =  1 % N; C } { const int I =  2 % N; C } { const int I =  3 % N; C } \
                                { const int I =  4 % N; C } { const int I =  5 % N; C } { const int I =  6 % N; C } { const int I =  7 % N; C } \
                                { const int I =  8 % N; C } { const int I =  9 % N; C } { const int I = 10 % N; C } { const int I = 11 % N; C } \
                                { const int I = 12 % N; C } { const int I = 13 % N; C } { const int I = 14 % N; C }
#define UNROLLED_FOR_16(I, N, C){ const int I =  0 % N; C } { const int I =  1 % N; C } { const int I =  2 % N; C } { const int I =  3 % N; C } \
                                { const int I =  4 % N; C } { const int I =  5 % N; C } { const int I =  6 % N; C } { const int I =  7 % N; C } \
                                { const int I =  8 % N; C } { const int I =  9 % N; C } { const int I = 10 % N; C } { const int I = 11 % N; C } \
                                { const int I = 12 % N; C } { const int I = 13 % N; C } { const int I = 14 % N; C } { const int I = 15 % N; C }

#define UNROLLED_FOR_WRAPPER(INDEX, COUNT, CODE) UNROLLED_FOR_##COUNT(INDEX, COUNT, CODE)
#define UNROLLED_FOR(INDEX, COUNT, CODE) UNROLLED_FOR_WRAPPER(INDEX, COUNT, CODE)

// Needed for cases when count is not a literal constant (e.g. template parameter)
#define UNROLLED_FOR_S(INDEX, COUNT, CODE)                  \
    switch(COUNT) {                                         \
        case  0: UNROLLED_FOR_0(INDEX, COUNT, CODE); break; \
        case  1: UNROLLED_FOR_1(INDEX, COUNT, CODE); break; \
        case  2: UNROLLED_FOR_2(INDEX, COUNT, CODE); break; \
        case  3: UNROLLED_FOR_3(INDEX, COUNT, CODE); break; \
        case  4: UNROLLED_FOR_4(INDEX, COUNT, CODE); break; \
        case  5: UNROLLED_FOR_5(INDEX, COUNT, CODE); break; \
        case  6: UNROLLED_FOR_6(INDEX, COUNT, CODE); break; \
        case  7: UNROLLED_FOR_7(INDEX, COUNT, CODE); break; \
        case  8: UNROLLED_FOR_8(INDEX, COUNT, CODE); break; \
        case  9: UNROLLED_FOR_9(INDEX, COUNT, CODE); break; \
        case 10: UNROLLED_FOR_10(INDEX, COUNT, CODE); break;\
        case 11: UNROLLED_FOR_11(INDEX, COUNT, CODE); break;\
        case 12: UNROLLED_FOR_12(INDEX, COUNT, CODE); break;\
        case 13: UNROLLED_FOR_13(INDEX, COUNT, CODE); break;\
        case 14: UNROLLED_FOR_14(INDEX, COUNT, CODE); break;\
        case 15: UNROLLED_FOR_15(INDEX, COUNT, CODE); break;\
        case 16: UNROLLED_FOR_16(INDEX, COUNT, CODE); break;\
        default:                                      break;\
    }

#define UNROLLED_FOR_R(INDEX, COUNT, CODE)              \
    switch(COUNT) {                                     \
        case 16: { const int INDEX = 15 % COUNT; CODE } \
        case 15: { const int INDEX = 14 % COUNT; CODE } \
        case 14: { const int INDEX = 13 % COUNT; CODE } \
        case 13: { const int INDEX = 12 % COUNT; CODE } \
        case 12: { const int INDEX = 11 % COUNT; CODE } \
        case 11: { const int INDEX = 10 % COUNT; CODE } \
        case 10: { const int INDEX =  9 % COUNT; CODE } \
        case  9: { const int INDEX =  8 % COUNT; CODE } \
        case  8: { const int INDEX =  7 % COUNT; CODE } \
        case  7: { const int INDEX =  6 % COUNT; CODE } \
        case  6: { const int INDEX =  5 % COUNT; CODE } \
        case  5: { const int INDEX =  4 % COUNT; CODE } \
        case  4: { const int INDEX =  3 % COUNT; CODE } \
        case  3: { const int INDEX =  2 % COUNT; CODE } \
        case  2: { const int INDEX =  1 % COUNT; CODE } \
        case  1: { const int INDEX =  0 % COUNT; CODE } \
    }

#define ITERATE_R(n, exp)               \
    if ((n) == 16) {                    \
        { const int i = 15 % n; exp }   \
        { const int i = 14 % n; exp }   \
        { const int i = 13 % n; exp }   \
        { const int i = 12 % n; exp }   \
        { const int i = 11 % n; exp }   \
        { const int i = 10 % n; exp }   \
        { const int i = 9 % n; exp }    \
        { const int i = 8 % n; exp }    \
        { const int i = 7 % n; exp }    \
        { const int i = 6 % n; exp }    \
        { const int i = 5 % n; exp }    \
        { const int i = 4 % n; exp }    \
        { const int i = 3 % n; exp }    \
        { const int i = 2 % n; exp }    \
        { const int i = 1 % n; exp }    \
        { const int i = 0 % n; exp }    \
    } else if ((n) == 8) {              \
        { const int i = 7 % n; exp }    \
        { const int i = 6 % n; exp }    \
        { const int i = 5 % n; exp }    \
        { const int i = 4 % n; exp }    \
        { const int i = 3 % n; exp }    \
        { const int i = 2 % n; exp }    \
        { const int i = 1 % n; exp }    \
        { const int i = 0 % n; exp }    \
    } else if ((n) == 4) {              \
        { const int i = 3 % n; exp }    \
        { const int i = 2 % n; exp }    \
        { const int i = 1 % n; exp }    \
        { const int i = 0 % n; exp }    \
    } else if ((n) == 3) {              \
        { const int i = 2 % n; exp }    \
        { const int i = 1 % n; exp }    \
        { const int i = 0 % n; exp }    \
    } else if ((n) == 2) {              \
        { const int i = 1 % n; exp }    \
        { const int i = 0 % n; exp }    \
    } else if ((n) == 1) {              \
        { const int i = 0; exp }        \
    }
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
        UNROLLED_FOR_S(i, S, { comp_[i] = f; })
    }
    template <typename... Tail>
    force_inline simd_vec(typename std::enable_if<sizeof...(Tail) + 1 == S, T>::type head, Tail... tail)
        : comp_{head, T(tail)...} {}
    force_inline explicit simd_vec(const T *f) { memcpy(&comp_, f, S * sizeof(T)); }
    force_inline simd_vec(const T *_f, simd_mem_aligned_tag) {
        const auto *f = (const T *)assume_aligned(_f, sizeof(T));
        memcpy(&comp_, f, S * sizeof(T));
    }

    force_inline T operator[](const int i) const { return comp_[i]; }

    template <int i> force_inline T get() const { return comp_[i]; }
    template <int i> force_inline void set(const T f) { comp_[i] = f; }
    force_inline void set(const int i, const T f) { comp_[i] = f; }

    force_inline simd_vec<T, S> &operator+=(const simd_vec<T, S> &rhs) {
        UNROLLED_FOR_S(i, S, { comp_[i] += rhs.comp_[i]; })
        return *this;
    }

    force_inline simd_vec<T, S> &operator-=(const simd_vec<T, S> &rhs) {
        UNROLLED_FOR_S(i, S, { comp_[i] -= rhs.comp_[i]; })
        return *this;
    }

    force_inline simd_vec<T, S> &operator*=(const simd_vec<T, S> &rhs) {
        UNROLLED_FOR_S(i, S, { comp_[i] *= rhs.comp_[i]; })
        return *this;
    }

    force_inline simd_vec<T, S> &operator/=(const simd_vec<T, S> &rhs) {
        UNROLLED_FOR_S(i, S, { comp_[i] /= rhs.comp_[i]; })
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
        UNROLLED_FOR_S(i, S, { comp_[i] += rhs; })
        return *this;
    }

    force_inline simd_vec<T, S> &operator-=(T rhs) {
        UNROLLED_FOR_S(i, S, { comp_[i] -= rhs; })
        return *this;
    }

    force_inline simd_vec<T, S> &operator*=(T rhs) {
        UNROLLED_FOR_S(i, S, { comp_[i] *= rhs; })
        return *this;
    }

    force_inline simd_vec<T, S> &operator/=(T rhs) {
        UNROLLED_FOR_S(i, S, { comp_[i] /= rhs; })
        return *this;
    }

    force_inline simd_vec<T, S> operator-() const {
        simd_vec<T, S> temp;
        UNROLLED_FOR_S(i, S, { temp.comp_[i] = -comp_[i]; })
        return temp;
    }

    force_inline simd_vec<T, S> operator==(T rhs) const {
        simd_vec<int, S> temp;
        UNROLLED_FOR_S(i, S, { temp.comp_[i] = comp_[i] == rhs ? -1 : 0; })

        static_assert(sizeof(simd_vec<T, S>) == sizeof(simd_vec<int, S>), "!");

        simd_vec<T, S> ret;
        memcpy(&ret, &temp, sizeof(simd_vec<T, S>));

        return ret;
    }

    force_inline simd_vec<T, S> operator==(const simd_vec<T, S> &rhs) const {
        simd_vec<int, S> temp;
        UNROLLED_FOR_S(i, S, { temp.comp_[i] = comp_[i] == rhs.comp_[i] ? -1 : 0; })

        static_assert(sizeof(simd_vec<T, S>) == sizeof(simd_vec<int, S>), "!");

        simd_vec<T, S> ret;
        memcpy(&ret, &temp, sizeof(simd_vec<T, S>));

        return ret;
    }

    force_inline simd_vec<T, S> operator!=(T rhs) const {
        simd_vec<int, S> temp;
        UNROLLED_FOR_S(i, S, { temp.comp_[i] = comp_[i] != rhs ? -1 : 0; })

        static_assert(sizeof(simd_vec<T, S>) == sizeof(simd_vec<int, S>), "!");

        simd_vec<T, S> ret;
        memcpy(&ret, &temp, sizeof(simd_vec<T, S>));

        return ret;
    }

    force_inline simd_vec<T, S> operator!=(const simd_vec<T, S> &rhs) const {
        simd_vec<int, S> temp;
        UNROLLED_FOR_S(i, S, { temp.comp_[i] = comp_[i] != rhs.comp_[i] ? -1 : 0; })

        static_assert(sizeof(simd_vec<T, S>) == sizeof(simd_vec<int, S>), "!");

        simd_vec<T, S> ret;
        memcpy(&ret, &temp, sizeof(simd_vec<T, S>));

        return ret;
    }

    force_inline simd_vec<T, S> operator<(const simd_vec<T, S> &rhs) const {
        simd_vec<int, S> temp;
        UNROLLED_FOR_S(i, S, { temp.comp_[i] = comp_[i] < rhs.comp_[i] ? -1 : 0; })

        static_assert(sizeof(simd_vec<T, S>) == sizeof(simd_vec<int, S>), "!");

        simd_vec<T, S> ret;
        memcpy(&ret, &temp, sizeof(simd_vec<T, S>));

        return ret;
    }

    force_inline simd_vec<T, S> operator<=(const simd_vec<T, S> &rhs) const {
        simd_vec<int, S> temp;
        UNROLLED_FOR_S(i, S, { temp.comp_[i] = comp_[i] <= rhs.comp_[i] ? -1 : 0; })

        static_assert(sizeof(simd_vec<T, S>) == sizeof(simd_vec<int, S>), "!");

        simd_vec<T, S> ret;
        memcpy(&ret, &temp, sizeof(simd_vec<T, S>));

        return ret;
    }

    force_inline simd_vec<T, S> operator>(const simd_vec<T, S> &rhs) const {
        simd_vec<int, S> temp;
        UNROLLED_FOR_S(i, S, { temp.comp_[i] = comp_[i] > rhs.comp_[i] ? -1 : 0; })

        static_assert(sizeof(simd_vec<T, S>) == sizeof(simd_vec<int, S>), "!");

        simd_vec<T, S> ret;
        memcpy(&ret, &temp, sizeof(simd_vec<T, S>));

        return ret;
    }

    force_inline simd_vec<T, S> operator>=(const simd_vec<T, S> &rhs) const {
        simd_vec<int, S> temp;
        UNROLLED_FOR_S(i, S, { temp.comp_[i] = comp_[i] >= rhs.comp_[i] ? -1 : 0; })

        static_assert(sizeof(simd_vec<T, S>) == sizeof(simd_vec<int, S>), "!");

        simd_vec<T, S> ret;
        memcpy(&ret, &temp, sizeof(simd_vec<T, S>));

        return ret;
    }

    force_inline simd_vec<T, S> &operator&=(const simd_vec<T, S> &rhs) {
        UNROLLED_FOR_S(i, S,
                       { reinterpret_cast<uint32_t &>(comp_[i]) &= reinterpret_cast<const uint32_t &>(rhs.comp_[i]); })
        return *this;
    }

    force_inline simd_vec<T, S> operator<(T rhs) const {
        simd_vec<int, S> temp;
        UNROLLED_FOR_S(i, S, { temp.comp_[i] = comp_[i] < rhs ? -1 : 0; })

        static_assert(sizeof(simd_vec<T, S>) == sizeof(simd_vec<int, S>), "!");

        simd_vec<T, S> ret;
        memcpy(&ret, &temp, sizeof(simd_vec<T, S>));

        return ret;
    }

    force_inline simd_vec<T, S> operator<=(T rhs) const {
        simd_vec<int, S> temp;
        UNROLLED_FOR_S(i, S, { temp.comp_[i] = comp_[i] <= rhs ? -1 : 0; })

        static_assert(sizeof(simd_vec<T, S>) == sizeof(simd_vec<int, S>), "!");

        simd_vec<T, S> ret;
        memcpy(&ret, &temp, sizeof(simd_vec<T, S>));

        return ret;
    }

    force_inline simd_vec<T, S> operator>(T rhs) const {
        simd_vec<int, S> temp;
        UNROLLED_FOR_S(i, S, { temp.comp_[i] = comp_[i] > rhs ? -1 : 0; })

        static_assert(sizeof(simd_vec<T, S>) == sizeof(simd_vec<int, S>), "!");

        simd_vec<T, S> ret;
        memcpy(&ret, &temp, sizeof(simd_vec<T, S>));

        return ret;
    }

    force_inline simd_vec<T, S> operator>=(T rhs) const {
        simd_vec<int, S> temp;
        UNROLLED_FOR_S(i, S, { temp.comp_[i] = comp_[i] >= rhs ? -1 : 0; })

        static_assert(sizeof(simd_vec<T, S>) == sizeof(simd_vec<int, S>), "!");

        simd_vec<T, S> ret;
        memcpy(&ret, &temp, sizeof(simd_vec<T, S>));

        return ret;
    }

    force_inline simd_vec<T, S> &operator&=(const T rhs) {
        UNROLLED_FOR_S(i, S, { comp_[i] &= rhs; })
        return *this;
    }

    force_inline simd_vec<T, S> operator~() const {
        simd_vec<T, S> ret;
        UNROLLED_FOR_S(i, S, {
            const uint32_t temp = ~reinterpret_cast<const uint32_t &>(comp_[i]);
            ret.comp_[i] = reinterpret_cast<const T &>(temp);
        })
        return ret;
    }

    force_inline explicit operator simd_vec<int, S>() const {
        simd_vec<int, S> ret;
        UNROLLED_FOR_S(i, S, { ret.comp_[i] = int(comp_[i]); })
        return ret;
    }

    force_inline explicit operator simd_vec<float, S>() const {
        simd_vec<float, S> ret;
        UNROLLED_FOR_S(i, S, { ret.comp_[i] = float(comp_[i]); })
        return ret;
    }

    force_inline simd_vec<T, S> sqrt() const {
        simd_vec<T, S> temp;
        UNROLLED_FOR_S(i, S, { temp.set<i>(std::sqrt(comp_[i])); })
        return temp;
    }

    force_inline simd_vec<T, S> log() const {
        simd_vec<T, S> temp;
        UNROLLED_FOR_S(i, S, { temp.set<i>(std::log(comp_[i])); })
        return temp;
    }

    force_inline T length() const {
        T temp = {0};
        UNROLLED_FOR_S(i, S, { temp += comp_[i] * comp_[i]; })
        return std::sqrt(temp);
    }

    force_inline T length2() const {
        T temp = {0};
        UNROLLED_FOR_S(i, S, { temp += comp_[i] * comp_[i]; })
        return temp;
    }

    force_inline void copy_to(T *f) const { memcpy(f, &comp_[0], S * sizeof(T)); }

    force_inline void copy_to(T *_f, simd_mem_aligned_tag) const {
        auto *f = (T *)assume_aligned(_f, sizeof(T));
        memcpy(f, &comp_[0], S * sizeof(T));
    }

    force_inline bool all_zeros() const {
        UNROLLED_FOR_S(i, S, {
            if (comp_[i] != 0) {
                return false;
            }
        })
        return true;
    }

    force_inline bool all_zeros(const simd_vec<int, S> &mask) const {
        const auto *src1 = reinterpret_cast<const uint8_t *>(&comp_[0]);
        const auto *src2 = reinterpret_cast<const uint8_t *>(&mask.comp_[0]);

        for (int i = 0; i < S * sizeof(T); i++) {
            if ((src1[i] & src2[i]) != 0) {
                return false;
            }
        }

        return true;
    }

    force_inline bool not_all_zeros() const {
        UNROLLED_FOR_S(i, S, {
            if (comp_[i] != 0) {
                return true;
            }
        })
        return false;
    }

    // clang-format off
    force_inline void blend_to(const simd_vec<T, S> &mask, const simd_vec<T, S> &v1) {
        UNROLLED_FOR_S(i, S, {
            if (mask.comp_[i] != T(0)) {
                comp_[i] = v1.comp_[i];
            }
        })
    }

    force_inline void blend_inv_to(const simd_vec<T, S> &mask, const simd_vec<T, S> &v1) {
        UNROLLED_FOR_S(i, S, {
            if (mask.comp_[i] == T(0)) {
                comp_[i] = v1.comp_[i];
            }
        })
    } // clang-format on

    force_inline int movemask() const {
        int res = 0;
        UNROLLED_FOR_S(i, S, {
            if (comp_[i] != T(0)) {
                res |= (1 << i);
            }
        })
        return res;
    }

    force_inline static simd_vec<T, S> min(const simd_vec<T, S> &v1, const simd_vec<T, S> &v2) {
        simd_vec<T, S> temp;
        UNROLLED_FOR_S(i, S, { temp.comp_[i] = std::min(v1.comp_[i], v2.comp_[i]); })
        return temp;
    }

    force_inline static simd_vec<T, S> min(const simd_vec<T, S> &v1, const T v2) {
        simd_vec<T, S> temp;
        UNROLLED_FOR_S(i, S, { temp.comp_[i] = std::min(v1.comp_[i], v2); })
        return temp;
    }

    force_inline static simd_vec<T, S> min(const T v1, const simd_vec<T, S> &v2) {
        simd_vec<T, S> temp;
        UNROLLED_FOR_S(i, S, { temp.comp_[i] = std::min(v1, v2.comp_[i]); })
        return temp;
    }

    force_inline static simd_vec<T, S> max(const simd_vec<T, S> &v1, const simd_vec<T, S> &v2) {
        simd_vec<T, S> temp;
        UNROLLED_FOR_S(i, S, { temp.comp_[i] = std::max(v1.comp_[i], v2.comp_[i]); })
        return temp;
    }

    force_inline static simd_vec<T, S> max(const simd_vec<T, S> &v1, const T v2) {
        simd_vec<T, S> temp;
        UNROLLED_FOR_S(i, S, { temp.comp_[i] = std::max(v1.comp_[i], v2); })
        return temp;
    }

    force_inline static simd_vec<T, S> max(const T v1, const simd_vec<T, S> &v2) {
        simd_vec<T, S> temp;
        UNROLLED_FOR_S(i, S, { temp.comp_[i] = std::max(v1, v2.comp_[i]); })
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
        UNROLLED_FOR_S(i, S, { temp.comp_[i] = float(int(v1.comp_[i]) - (v1.comp_[i] < 0.0f)); })
        return temp;
    }

    force_inline static simd_vec<float, S> ceil(const simd_vec<float, S> &v1) {
        simd_vec<float, S> temp;
        UNROLLED_FOR_S(i, S, {
            int _v = int(v1.comp_[i]);
            temp.comp_[i] = float(_v + (v1.comp_[i] != _v));
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
        UNROLLED_FOR_S(i, S, { ret.comp_[i] = v1.comp_[i] + v2.comp_[i]; })
        return ret;
    }

    friend force_inline simd_vec<T, S> operator-(const simd_vec<T, S> &v1, const simd_vec<T, S> &v2) {
        simd_vec<T, S> ret;
        UNROLLED_FOR_S(i, S, { ret.comp_[i] = v1.comp_[i] - v2.comp_[i]; })
        return ret;
    }

    friend force_inline simd_vec<T, S> operator*(const simd_vec<T, S> &v1, const simd_vec<T, S> &v2) {
        simd_vec<T, S> ret;
        UNROLLED_FOR_S(i, S, { ret.comp_[i] = v1.comp_[i] * v2.comp_[i]; })
        return ret;
    }

    friend force_inline simd_vec<T, S> operator/(const simd_vec<T, S> &v1, const simd_vec<T, S> &v2) {
        simd_vec<T, S> ret;
        UNROLLED_FOR_S(i, S, { ret.comp_[i] = v1.comp_[i] / v2.comp_[i]; })
        return ret;
    }

    friend force_inline simd_vec<T, S> operator+(const simd_vec<T, S> &v1, T v2) {
        simd_vec<T, S> ret;
        UNROLLED_FOR_S(i, S, { ret.comp_[i] = v1.comp_[i] + v2; })
        return ret;
    }

    friend force_inline simd_vec<T, S> operator-(const simd_vec<T, S> &v1, T v2) {
        simd_vec<T, S> ret;
        UNROLLED_FOR_S(i, S, { ret.comp_[i] = v1.comp_[i] - v2; })
        return ret;
    }

    friend force_inline simd_vec<T, S> operator*(const simd_vec<T, S> &v1, T v2) {
        simd_vec<T, S> ret;
        UNROLLED_FOR_S(i, S, { ret.comp_[i] = v1.comp_[i] * v2; })
        return ret;
    }

    friend force_inline simd_vec<T, S> operator/(const simd_vec<T, S> &v1, T v2) {
        simd_vec<T, S> ret;
        UNROLLED_FOR_S(i, S, { ret.comp_[i] = v1.comp_[i] / v2; })
        return ret;
    }

    friend force_inline simd_vec<T, S> operator+(T v1, const simd_vec<T, S> &v2) { return operator+(v2, v1); }

    friend force_inline simd_vec<T, S> operator-(T v1, const simd_vec<T, S> &v2) {
        simd_vec<T, S> ret;
        UNROLLED_FOR_S(i, S, { ret.comp_[i] = v1 - v2.comp_[i]; })
        return ret;
    }

    friend force_inline simd_vec<T, S> operator*(T v1, const simd_vec<T, S> &v2) {
        simd_vec<T, S> ret;
        UNROLLED_FOR_S(i, S, { ret.comp_[i] = v1 * v2.comp_[i]; })
        return ret;
    }

    friend force_inline simd_vec<T, S> operator/(T v1, const simd_vec<T, S> &v2) {
        simd_vec<T, S> ret;
        UNROLLED_FOR_S(i, S, { ret.comp_[i] = v1 / v2.comp_[i]; })
        return ret;
    }

    friend force_inline simd_vec<T, S> operator>>(const simd_vec<T, S> &v1, const simd_vec<T, S> &v2) {
        simd_vec<T, S> ret;
        UNROLLED_FOR_S(i, S, { ret.comp_[i] = v1.comp_[i] >> v2.comp_[i]; })
        return ret;
    }

    friend force_inline simd_vec<T, S> operator>>(const simd_vec<T, S> &v1, T v2) {
        simd_vec<T, S> ret;
        UNROLLED_FOR_S(i, S, { ret.comp_[i] = v1.comp_[i] >> v2; })
        return ret;
    }

    friend force_inline simd_vec<T, S> operator<<(const simd_vec<T, S> &v1, const simd_vec<T, S> &v2) {
        simd_vec<T, S> ret;
        UNROLLED_FOR_S(i, S, { ret.comp_[i] = v1.comp_[i] << v2.comp_[i]; })
        return ret;
    }

    friend force_inline simd_vec<T, S> operator<<(const simd_vec<T, S> &v1, T v2) {
        simd_vec<T, S> ret;
        UNROLLED_FOR_S(i, S, { ret.comp_[i] = v1.comp_[i] << v2; })
        return ret;
    }

    friend force_inline simd_vec<T, S> srai(const simd_vec<T, S> &v1, int v2) {
        simd_vec<T, S> ret;
        UNROLLED_FOR_S(i, S, { ret.comp_[i] = v1.comp_[i] >> v2; })
        return ret;
    }

    friend force_inline T dot(const simd_vec<T, S> &v1, const simd_vec<T, S> &v2) {
        T ret = {0};
        UNROLLED_FOR_S(i, S, { ret += v1.comp_[i] * v2.comp_[i]; })
        return ret;
    }

    friend force_inline simd_vec<T, S> clamp(const simd_vec<T, S> &v1, T min, T max) {
        simd_vec<T, S> ret;
        UNROLLED_FOR_S(i, S, { ret.comp_[i] = v1.comp_[i] < min ? min : (v1.comp_[i] > max ? max : v1.comp_[i]); })
        return ret;
    }

    friend force_inline simd_vec<T, S> pow(const simd_vec<T, S> &v1, const simd_vec<T, S> &v2) {
        simd_vec<T, S> ret;
        UNROLLED_FOR_S(i, S, { ret.comp_[i] = std::pow(v1.comp_[i], v2.comp_[i]); })
        return ret;
    }

    friend force_inline simd_vec<T, S> normalize(const simd_vec<T, S> &v1) { return v1 / v1.length(); }

    friend force_inline bool is_equal(const simd_vec<T, S> &v1, const simd_vec<T, S> &v2) {
        bool res = true;
        UNROLLED_FOR_S(i, S, { res = res && (v1.comp_[i] == v2.comp_[i]); })
        return res;
    }

    friend force_inline const T *value_ptr(const simd_vec<T, S> &v1) { return &v1.comp_[0]; }
    friend force_inline T *value_ptr(simd_vec<T, S> &v1) { return &v1.comp_[0]; }

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

template <typename T, int S> force_inline simd_vec<T, S> gather(const T *base_addr, const simd_vec<int, S> &vindex) {
    simd_vec<T, S> res;
    UNROLLED_FOR_S(i, S, { res.template set<i>(base_addr[vindex[i]]); });
    return res;
}

template <typename T, int S>
force_inline void scatter(T *base_addr, const simd_vec<int, S> &vindex, const simd_vec<T, S> &v) {
    UNROLLED_FOR_S(i, S, { base_addr[vindex[i]] = v.template get<i>(); });
}

template <typename T, typename U, int S> class simd_comp_where_helper {
    const simd_vec<T, S> &mask_;
    simd_vec<T, S> &comp_;

  public:
    force_inline simd_comp_where_helper(const simd_vec<U, S> &mask, simd_vec<T, S> &vec)
        : mask_(reinterpret_cast<const simd_vec<T, S> &>(mask)), comp_(vec) {}

    force_inline void operator=(const simd_vec<T, S> &vec) { comp_.blend_to(mask_, vec); }
    force_inline void operator+=(const simd_vec<T, S> &vec) { comp_.blend_to(mask_, comp_ + vec); }
    force_inline void operator-=(const simd_vec<T, S> &vec) { comp_.blend_to(mask_, comp_ - vec); }
    force_inline void operator*=(const simd_vec<T, S> &vec) { comp_.blend_to(mask_, comp_ * vec); }
    force_inline void operator/=(const simd_vec<T, S> &vec) { comp_.blend_to(mask_, comp_ / vec); }
    force_inline void operator|=(const simd_vec<T, S> &vec) { comp_.blend_to(mask_, comp_ | vec); }
    force_inline void operator&=(const simd_vec<T, S> &vec) { comp_.blend_to(mask_, comp_ & vec); }
};

template <typename T, typename U, int S> class simd_comp_where_inv_helper {
    const simd_vec<T, S> &mask_;
    simd_vec<T, S> &comp_;

  public:
    force_inline simd_comp_where_inv_helper(const simd_vec<U, S> &mask, simd_vec<T, S> &vec)
        : mask_(reinterpret_cast<const simd_vec<T, S> &>(mask)), comp_(vec) {}

    force_inline void operator=(const simd_vec<T, S> &vec) { comp_.blend_inv_to(mask_, vec); }
    force_inline void operator+=(const simd_vec<T, S> &vec) { comp_.blend_inv_to(mask_, comp_ + vec); }
    force_inline void operator-=(const simd_vec<T, S> &vec) { comp_.blend_inv_to(mask_, comp_ - vec); }
    force_inline void operator*=(const simd_vec<T, S> &vec) { comp_.blend_inv_to(mask_, comp_ * vec); }
    force_inline void operator/=(const simd_vec<T, S> &vec) { comp_.blend_inv_to(mask_, comp_ / vec); }
    force_inline void operator|=(const simd_vec<T, S> &vec) { comp_.blend_inv_to(mask_, comp_ | vec); }
    force_inline void operator&=(const simd_vec<T, S> &vec) { comp_.blend_inv_to(mask_, comp_ & vec); }
};

template <typename T, typename U, int S>
force_inline simd_comp_where_helper<T, U, S> where(const simd_vec<U, S> &mask, simd_vec<T, S> &vec) {
    return {mask, vec};
}

template <typename T, typename U, int S>
force_inline simd_comp_where_inv_helper<T, U, S> where_not(const simd_vec<U, S> &mask, simd_vec<T, S> &vec) {
    return {mask, vec};
}

template <int S> force_inline simd_vec<int, S> simd_cast(const simd_vec<float, S> &vec) {
    simd_vec<int, S> ret;
    memcpy(&ret, &vec, sizeof(simd_vec<int, S>));
    return ret;
}

template <int S> force_inline const simd_vec<float, S> simd_cast(const simd_vec<int, S> &vec) {
    simd_vec<float, S> ret;
    memcpy(&ret, &vec, sizeof(simd_vec<float, S>));
    return ret;
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
